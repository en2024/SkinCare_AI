import logging
import os
import re

import json
import cv2
import numpy as np
import pytesseract
import torch
import torch.nn.functional as F
import torch.nn as nn
from django.conf import settings
from django.http import JsonResponse
from django.views.decorators.http import require_POST
from PIL import Image, ImageEnhance
from torchvision import transforms, models

from ..models import Product

logger = logging.getLogger('analyzer')

# ─── إعداد مسار Tesseract OCR ───────────────────────────────────────────────
pytesseract.pytesseract.tesseract_cmd = getattr(settings, 'TESSERACT_CMD', r'C:\Program Files\Tesseract-OCR\tesseract.exe')

# ─── 1. Face Analysis Model (Vision AI) ──────────────────────────────────────
CLASS_NAMES = ['combination', 'dry', 'normal', 'oily', 'sensitive']
face_model = models.resnet50(weights=None)
face_model.fc = nn.Linear(face_model.fc.in_features, 5)
face_model_path = os.path.join(settings.BASE_DIR, 'analyzer', 'ml_models', 'model_resnet50_5class.pth')

try:
    face_model.load_state_dict(torch.load(face_model_path, map_location=torch.device('cpu')), strict=False)
    face_model.eval()
    logger.info('Face analysis model loaded successfully.')
except Exception as e:
    logger.error(f'Face Model Loading Error: {e}')

# ─── Load YOLOv8 Acne Detection Model ───
try:
    from ultralytics import YOLO
    yolo_model_path = os.path.join(settings.BASE_DIR, 'analyzer', 'ml_models', 'acne_yolov8.pt')
    if os.path.exists(yolo_model_path):
        acne_model = YOLO(yolo_model_path)
        logger.info('YOLO acne detection model loaded.')
    else:
        acne_model = None
except ImportError:
    acne_model = None
    logger.warning("Ultralytics not installed. YOLO detection disabled.")

# ─── Face Detection Gate (OpenCV Haar Cascade) ───────────────────────────────
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# ─── 2. Ingredients AI Logic (The Hybrid System) ─────────────────────────────
# Unified model loaded via predictor helper (lazy singleton)
from ..ml_models.ingredients.predictor import predict_safety


# قاعدة بيانات المكونات المحظورة (Safety Shield)
HARMFUL_INGREDIENTS = {
    'oily': ['alcohol denat', 'isopropyl myristate', 'coconut oil', 'lanolin', 'mineral oil', 'cocoa butter', 'sodium lauryl sulfate'],
    'dry': ['alcohol denat', 'benzoyl peroxide', 'salicylic acid', 'fragrance', 'sulfates', 'isopropyl alcohol'],
    'sensitive': ['alcohol denat', 'fragrance', 'parfum', 'parabens', 'linalool', 'limonene', 'essential oil', 'sls', 'sodium lauryl sulfate', 'phenoxyethanol'],
    'normal': ['alcohol denat', 'fragrance', 'parabens', 'sodium lauryl sulfate'],
    'combination': ['alcohol denat', 'coconut oil', 'mineral oil', 'isopropyl myristate'],
}

SAFE_INGREDIENTS = {
    'oily': ['salicylic acid', 'niacinamide', 'hyaluronic acid', 'zinc', 'tea tree'],
    'dry': ['hyaluronic acid', 'glycerin', 'ceramide', 'squalane', 'shea butter'],
    'sensitive': ['aloe vera', 'chamomile', 'oat', 'ceramide', 'panthenol', 'allantoin', 'centella asiatica'],
    'normal': ['hyaluronic acid', 'glycerin', 'vitamin c', 'niacinamide'],
    'combination': ['hyaluronic acid', 'niacinamide', 'ceramide', 'salicylic acid'],
}


# ─── 3. OCR Helper (Multi-pass pre-processing for product labels) ─────────────
def _ocr_single_pass(pil_img, psm=6):
    """Run Tesseract on a single pre-processed PIL image."""
    config = f'--oem 3 --psm {psm} -l eng'
    try:
        return pytesseract.image_to_string(pil_img, config=config)
    except Exception:
        return ""


def extract_text_from_image(pil_image):
    """
    Multi-pass OCR extraction optimised for ingredient labels.
    Tries 3 different pre-processing strategies and picks the longest
    (most informative) result.
    """
    orig = pil_image.convert('RGB')
    w, h = orig.size
    # Up-scale small images so Tesseract gets more detail
    scale = max(1, 2000 // max(w, h))
    if scale > 1:
        orig = orig.resize((w * scale, h * scale), Image.LANCZOS)

    gray = orig.convert('L')

    candidates = []

    # Pass 1: High-contrast grayscale
    enhanced = ImageEnhance.Contrast(gray).enhance(2.2)
    enhanced = ImageEnhance.Sharpness(enhanced).enhance(2.0)
    candidates.append(_ocr_single_pass(enhanced, psm=6))

    # Pass 2: Adaptive threshold via OpenCV
    try:
        arr = np.array(gray)
        thresh = cv2.adaptiveThreshold(
            arr, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 10
        )
        candidates.append(_ocr_single_pass(Image.fromarray(thresh), psm=6))
    except Exception:
        pass

    # Pass 3: Denoised + Otsu threshold
    try:
        arr = np.array(gray)
        denoised = cv2.fastNlMeansDenoising(arr, None, 12, 7, 21)
        _, otsu = cv2.threshold(denoised, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        candidates.append(_ocr_single_pass(Image.fromarray(otsu), psm=4))
    except Exception:
        pass

    # Pick the longest useful result
    best = ""
    for raw in candidates:
        cleaned = re.sub(r'[^a-zA-Z,\s\-&()./]', ' ', raw)
        cleaned = re.sub(r'\s+', ' ', cleaned).strip()
        if len(cleaned) > len(best):
            best = cleaned

    if best:
        logger.info(f'OCR extracted {len(best)} chars from image.')
    else:
        logger.warning('OCR could not extract any text from the uploaded image.')

    return best


# ─── Ingredient-text validation gate ──────────────────────────────────────────
# Cosmetic / INCI keywords that signal an authentic ingredient list
_INGREDIENT_KEYWORDS = [
    # Common header words
    'ingredients', 'active ingredients', 'inactive ingredients', 'composition',
    # Top INCI ingredients (appear on almost every label)
    'aqua', 'water', 'glycerin', 'glycerine', 'dimethicone', 'tocopherol',
    'panthenol', 'niacinamide', 'retinol', 'hyaluronic', 'squalane',
    'cetearyl', 'stearic', 'palmitic', 'caprylic', 'cetyl',
    'sodium', 'potassium', 'magnesium', 'zinc', 'titanium dioxide',
    # Functional groups
    'acid', 'sulfate', 'sulphate', 'paraben', 'phenoxyethanol',
    'extract', 'oil', 'butter', 'wax', 'alcohol',
    'fragrance', 'parfum', 'ci ',  # colour-index prefix
    # Botanical / active markers
    'aloe', 'chamomile', 'centella', 'ceramide', 'peptide',
    'collagen', 'keratin', 'salicylic', 'benzoyl', 'kojic',
    'ascorbic', 'lactic', 'citric', 'glycolic',
    'shea', 'jojoba', 'argan', 'rosehip', 'tea tree',
]
_MIN_KEYWORD_HITS = 3  # At least this many must match


def _validate_ingredients_text(text: str) -> tuple:
    """
    Check whether *text* looks like a genuine skincare ingredient list.

    Returns
    -------
    (is_valid: bool, hits: list[str])
        is_valid is True when >= _MIN_KEYWORD_HITS keywords are found.
        hits contains the matched keywords (for logging / debugging).
    """
    lower = text.lower()
    hits = [kw for kw in _INGREDIENT_KEYWORDS if kw in lower]
    return len(hits) >= _MIN_KEYWORD_HITS, hits


# ─── 4. AI Endpoints ─────────────────────────────────────────────────────────

@require_POST
def analyze_skin(request):
    """
    Multi-stage skin analysis pipeline:
      1. YOLO detects acne bounding boxes
      2. Acne regions are blurred out → ResNet sees clean skin texture
      3. ResNet classifies skin type from the cleaned image
      4. Acne count applies a minor secondary adjustment
    """
    if request.FILES.get('image'):
        try:
            img = Image.open(request.FILES.get('image')).convert('RGB')

            # ── Step 0: Face Detection & Patch Extraction ───────────────
            img_cv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
            gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60))

            if len(faces) == 0:
                logger.info('Face detection gate: no face found in image.')
                return JsonResponse({
                    'error': 'No face detected. Please upload a clear photo of your face.',
                    'no_face': True
                }, status=200)

            # Get largest face
            (x, y, w, h) = max(faces, key=lambda f: f[2] * f[3])
            logger.info('Face detected. Proceeding with analysis.')

            # The ResNet model was trained on pure skin patches (zoomed in textures), 
            # NOT full faces. We must crop a patch of the cheek to match the training data.
            cheek_x = x + int(w * 0.6)
            cheek_y = y + int(h * 0.5)
            cheek_w = int(w * 0.3)
            cheek_h = int(h * 0.3)
            
            # Ensure it's within image bounds
            cheek_x = min(max(0, cheek_x), img.width - cheek_w)
            cheek_y = min(max(0, cheek_y), img.height - cheek_h)
            
            skin_patch = img.crop((cheek_x, cheek_y, cheek_x + cheek_w, cheek_y + cheek_h))
            logger.info('Extracted pure skin patch for ResNet analysis.')

            # ── Step 1: YOLO detects acne locations on FULL face ─────────
            acne_count = 0
            acne_boxes = []
            if acne_model is not None:
                results = acne_model(img, verbose=False)
                for r in results:
                    for box, cls_id in zip(r.boxes.xyxy, r.boxes.cls):
                        if acne_model.names[int(cls_id)] == 'acne':
                            acne_count += 1
                            acne_boxes.append(box.cpu().numpy().astype(int))
                logger.info(f'YOLO detected {acne_count} acne spot(s).')

            # ── Step 2: Create a "clean" patch for ResNet ────────────────
            # Blur out any acne that happens to fall inside our cheek patch
            clean_patch = skin_patch.copy()
            if acne_boxes:
                from PIL import ImageFilter
                blurred_patch = skin_patch.filter(ImageFilter.GaussianBlur(radius=15))
                # Adjust acne boxes to patch coordinates
                for (ax1, ay1, ax2, ay2) in acne_boxes:
                    # Check if acne overlaps with our cheek patch
                    if not (ax2 < cheek_x or ax1 > cheek_x + cheek_w or ay2 < cheek_y or ay1 > cheek_y + cheek_h):
                        px1 = max(0, ax1 - cheek_x)
                        py1 = max(0, ay1 - cheek_y)
                        px2 = min(cheek_w, ax2 - cheek_x)
                        py2 = min(cheek_h, ay2 - cheek_y)
                        
                        if px2 > px1 and py2 > py1:
                            acne_crop = blurred_patch.crop((px1, py1, px2, py2))
                            clean_patch.paste(acne_crop, (px1, py1, px2, py2))

            # ── Step 3: ResNet classifies the CLEANED PATCH ──────────────
            output = face_model(transform(clean_patch).unsqueeze(0))
            probabilities = F.softmax(output, dim=1).squeeze()  # shape: [5]

            # ── Step 4: Minor secondary adjustment from acne evidence ────
            # This is a SMALL nudge, NOT the main factor.
            # The main classification already happened on clean skin above.
            # CLASS_NAMES = ['combination', 'dry', 'normal', 'oily', 'sensitive']
            # Indices:        0               1      2         3       4
            if acne_model is not None and acne_count > 0:
                adjusted = probabilities.clone().detach()

                if acne_count >= 3:
                    # Noticeable acne presence → small boost to oily/sensitive
                    adjusted[3] *= 1.08  # oily
                    adjusted[4] *= 1.06  # sensitive
                    adjusted[0] *= 1.04  # combination
                elif acne_count >= 1:
                    # Minimal acne → very slight nudge
                    adjusted[3] *= 1.04  # oily
                    adjusted[4] *= 1.03  # sensitive

                # Re-normalize
                adjusted = adjusted / adjusted.sum()
                probabilities = adjusted

            # ── Pick the final winner ────────────────────────────────────
            confidence_val, predicted_idx = torch.max(probabilities, dim=0)
            confidence_pct = round(confidence_val.item() * 100, 1)
            skin_type = CLASS_NAMES[predicted_idx.item()]

            # Confidence threshold check
            if confidence_pct < 55.0:
                logger.info(f'Low confidence scan: {confidence_pct}% for {skin_type}')
                return JsonResponse({
                    'low_confidence': True,
                    'confidence': confidence_pct,
                    'error': 'Upload a clearer photo following the guidelines.'
                }, status=200)

            recs = Product.objects.filter(skin_type__in=[skin_type, 'all'])[:4]
            products_list = [{'name': p.name, 'category': p.category, 'image': p.image.url if p.image else None} for p in recs]
            return JsonResponse({
                'skin_type': skin_type,
                'confidence': confidence_pct,
                'acne_detected': acne_count,
                'products': products_list
            })
        except Exception as e:
            logger.error(f'Skin analysis error: {e}')
            return JsonResponse({'error': str(e)}, status=500)
    return JsonResponse({'error': 'Invalid request'}, status=400)


@require_POST
def analyze_product(request):
    """Ingredient analysis using the unified ML model + rule-based safety shield."""
    skin_type = request.POST.get('skin_type', 'normal').lower()
    ingredients_text = request.POST.get('ingredients_text', '').strip()
    input_source = 'text'  # Track whether input came from text or image

    # 1. OCR — extract text from uploaded product-label image
    if request.FILES.get('image'):
        input_source = 'image'
        try:
            pil_img = Image.open(request.FILES.get('image')).convert('RGB')
            ingredients_text = extract_text_from_image(pil_img)
        except Exception as e:
            logger.error(f'Image processing error: {e}')
            return JsonResponse({
                'error': 'Unable to process the uploaded image. Please try a different photo.',
                'ocr_failed': True,
                'input_source': input_source,
            }, status=400)

    # 2. Validate that we have readable text
    if not ingredients_text:
        if input_source == 'image':
            return JsonResponse({
                'error': ('We couldn\u2019t detect any readable text in this image. '
                          'Please ensure the product label is clearly visible and try again.'),
                'ocr_failed': True,
                'not_ingredients': True,
                'input_source': input_source,
            }, status=400)
        return JsonResponse({
            'error': 'No ingredients provided. Please enter or upload ingredients.',
            'input_source': input_source,
        }, status=400)

    # 2b. Keyword validation — only process genuine ingredient lists
    if input_source == 'image':
        is_valid, keyword_hits = _validate_ingredients_text(ingredients_text)
        if not is_valid:
            logger.info(
                f'Ingredient validation failed — only {len(keyword_hits)} keyword(s) '
                f'matched: {keyword_hits}. Extracted text: "{ingredients_text[:120]}"'
            )
            return JsonResponse({
                'error': ('We couldn\u2019t detect a valid ingredients list in this image. '
                          'Please ensure the label is clear and readable, '
                          'or paste the ingredients manually below.'),
                'not_ingredients': True,
                'input_source': input_source,
                'ocr_text': ingredients_text[:500],
            }, status=400)

    # 3. AI Model Prediction via unified model (cached singleton)
    final_score = 50.0
    found_harmful = []
    found_safe = []
    try:
        result = predict_safety(ingredients_text, skin_type)
        prediction = result['prediction']   # 1 = safe, 0 = unsafe
        prob = result['probability']         # P(safe)

        # 4. Rule-Based Safety Shield
        clean_txt = ingredients_text.lower()
        found_harmful = [ing.title() for ing in HARMFUL_INGREDIENTS.get(skin_type, []) if ing in clean_txt]
        found_safe = [ing.title() for ing in SAFE_INGREDIENTS.get(skin_type, []) if ing in clean_txt]

        if found_harmful:
            final_score = 40.0 - (len(found_harmful) * 5)
        elif prediction == 1:
            final_score = 75 + (prob * 25) + (len(found_safe) * 2)
        else:
            final_score = prob * 70

    except Exception as e:
        logger.error(f'Prediction System Error: {e}')
        final_score = 50.0

    final_score = round(max(5, min(100, final_score)), 1)

    # Recommendations
    recs = Product.objects.filter(skin_type__in=[skin_type, 'all'])[:3] if final_score < 70 else []
    recommendations = [{'name': p.name, 'category': p.category, 'image': p.image.url if p.image else None} for p in recs]

    return JsonResponse({
        'safety_score': final_score,
        'harmful_ingredients': found_harmful,
        'safe_ingredients': found_safe,
        'skin_type': skin_type,
        'recommendations': recommendations,
        'input_source': input_source,
        'ocr_text': ingredients_text[:500],
    })


# ─── 5. Hybrid Skin Analysis (AI + Questionnaire) ────────────────────────────

# Questionnaire answer → skin-type point mapping
QUIZ_SCORING = {
    # Q1: How does your skin feel 30 min after washing?
    'q1': {'a': 'oily', 'b': 'dry', 'c': 'normal', 'd': 'sensitive', 'e': 'combination'},
    # Q2: How often does your face get shiny by midday?
    'q2': {'a': 'oily', 'b': 'dry', 'c': 'normal', 'd': 'sensitive', 'e': 'combination'},
    # Q3: How does your skin react to new products?
    'q3': {'a': 'oily', 'b': 'dry', 'c': 'normal', 'd': 'sensitive', 'e': 'combination'},
}

SKIN_DESCRIPTIONS = {
    'oily': 'Your skin produces excess sebum, especially in the T-zone. Lightweight, oil-free products are ideal.',
    'dry': 'Your skin tends to feel tight and may flake. Rich, hydrating products with ceramides work best.',
    'normal': 'Your skin is well-balanced. A simple, consistent routine will keep it healthy.',
    'sensitive': 'Your skin is easily irritated. Look for gentle, fragrance-free products with soothing ingredients.',
    'combination': 'Your skin is oily in the T-zone but dry elsewhere. You may need to treat different areas differently.',
}


@require_POST
def analyze_skin_hybrid(request):
    """الجمع بين نتيجة الذكاء الاصطناعي وإجابات الاستبيان للحصول على نتيجة نهائية"""
    try:
        data = json.loads(request.body)
    except json.JSONDecodeError:
        return JsonResponse({'error': 'Invalid JSON'}, status=400)

    ai_skin_type = data.get('ai_skin_type', '').lower()
    ai_confidence = float(data.get('ai_confidence', 0))
    answers = data.get('answers', {})  # {'q1': 'a', 'q2': 'b', 'q3': 'c'}

    if not ai_skin_type:
        return JsonResponse({'error': 'Missing AI result.'}, status=400)

    # ── Questionnaire scoring ────────────────────────────────────────────
    quiz_scores = {'oily': 0, 'dry': 0, 'normal': 0, 'sensitive': 0, 'combination': 0}
    for q_key, mapping in QUIZ_SCORING.items():
        answer = answers.get(q_key, '')
        skin_type_vote = mapping.get(answer, '')
        if skin_type_vote:
            quiz_scores[skin_type_vote] += 2  # Each answer worth 2 points

    quiz_winner = max(quiz_scores, key=quiz_scores.get)
    quiz_max_score = quiz_scores[quiz_winner]

    # ── Hybrid combination logic ─────────────────────────────────────────
    # High AI confidence (>85%): AI wins unless quiz is unanimous against
    # Medium AI confidence (75-85%): quiz wins on disagreement
    if ai_confidence >= 85.0:
        # AI is very confident — trust it unless quiz unanimously disagrees
        if quiz_winner != ai_skin_type and quiz_max_score == 6:
            # All 3 answers point to a different type — quiz overrides
            final_type = quiz_winner
            method = 'questionnaire_override'
        else:
            final_type = ai_skin_type
            method = 'ai_dominant'
    else:
        # AI is moderately confident (75-84%) — quiz has more influence
        if quiz_winner != ai_skin_type:
            final_type = quiz_winner
            method = 'questionnaire_preferred'
        else:
            final_type = ai_skin_type
            method = 'consensus'

    # Final confidence = weighted average (AI weight based on its confidence)
    ai_weight = ai_confidence / 100.0
    quiz_confidence = (quiz_max_score / 6.0) * 100
    final_confidence = round((ai_weight * ai_confidence) + ((1 - ai_weight) * quiz_confidence), 1)
    final_confidence = min(99.0, max(50.0, final_confidence))

    logger.info(
        f'Hybrid analysis: AI={ai_skin_type}({ai_confidence}%) '
        f'Quiz={quiz_winner}({quiz_confidence}%) → Final={final_type} via {method}'
    )

    # Recommendations
    recs = Product.objects.filter(skin_type__in=[final_type, 'all'])[:4]
    products_list = [{
        'name': p.name, 'category': p.category,
        'image': p.image.url if p.image else None
    } for p in recs]

    return JsonResponse({
        'skin_type': final_type,
        'confidence': final_confidence,
        'method': method,
        'ai_result': ai_skin_type,
        'quiz_result': quiz_winner,
        'quiz_scores': quiz_scores,
        'description': SKIN_DESCRIPTIONS.get(final_type, ''),
        'products': products_list,
    })


@require_POST
def toggle_favorite(request):
    """Toggle a product in the user's favorites list."""
    if not request.user.is_authenticated:
        return JsonResponse({'error': 'Login required.'}, status=401)

    from ..models import Product, UserProfile
    try:
        data = json.loads(request.body)
        product_id = data.get('product_id')
        product = Product.objects.get(id=product_id)
        profile, _ = UserProfile.objects.get_or_create(user=request.user)

        if product in profile.favorite_products.all():
            profile.favorite_products.remove(product)
            return JsonResponse({'status': 'removed', 'product_id': product_id})
        else:
            profile.favorite_products.add(product)
            return JsonResponse({'status': 'added', 'product_id': product_id})
    except Product.DoesNotExist:
        return JsonResponse({'error': 'Product not found.'}, status=404)
    except Exception as e:
        return JsonResponse({'error': str(e)}, status=400)
