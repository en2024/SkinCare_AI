"""
API endpoints for AI-powered skin and product analysis.

Endpoints:
  POST /analyze/          → Face scan: detects skin type using ResNet50 + acne detection with YOLOv8
  POST /analyze-product/  → Formula check: OCR + ingredient safety analysis
  POST /analyze-hybrid/   → Hybrid analysis: merges AI scan results with user questionnaire
  POST /toggle-favorite/  → Add or remove a product from the user's favorites
"""
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
from PIL import Image, ImageEnhance, ImageFilter
from torchvision import transforms, models
from ..models import Product, UserProfile
from ..validators import validate_face_image, validate_ingredient_image
from ..ml_models.ingredients.predictor import predict_safety

logger = logging.getLogger('analyzer')

# ── Tesseract OCR setup ──────────────────────────────────────────────────────

pytesseract.pytesseract.tesseract_cmd = getattr(settings, 'TESSERACT_CMD', r'C:\Program Files\Tesseract-OCR\tesseract.exe')

# ── ResNet50 skin type classifier ────────────────────────────────────────────
# Classifies face images into one of 5 skin types using a fine-tuned ResNet50.

CLASS_NAMES = ['combination', 'dry', 'normal', 'oily', 'sensitive']
face_model = models.resnet50(weights=None)
face_model.fc = nn.Linear(face_model.fc.in_features, 5)
face_model_path = os.path.join(settings.BASE_DIR, 'analyzer', 'ml_models', 'model_resnet50_5class.pth')

try:
    face_model.load_state_dict(torch.load(face_model_path, map_location=torch.device('cpu')), strict=False)
    face_model.eval()
    logger.info('Face model loaded.')
except Exception as e:
    logger.error(f'Face model error: {e}')

# ── YOLOv8 acne detector ────────────────────────────────────────────────────
# Draws bounding boxes around acne spots to adjust skin type confidence.

try:
    from ultralytics import YOLO
    yolo_model_path = os.path.join(settings.BASE_DIR, 'analyzer', 'ml_models', 'acne_yolov8.pt')
    if os.path.exists(yolo_model_path):
        acne_model = YOLO(yolo_model_path)
        logger.info('YOLO loaded.')
    else:
        acne_model = None
except ImportError:
    acne_model = None
    logger.warning("YOLO disabled.")

# OpenCV face detector (used as a pre-check before running the ResNet)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Standard ImageNet preprocessing for the ResNet50 model
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# ── Ingredient safety lists ──────────────────────────────────────────────────
# These are used by the formula checker to flag risky or beneficial ingredients
# for each skin type.

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

# ── OCR helpers ──────────────────────────────────────────────────────────────


def _ocr_single_pass(pil_img, psm=6):
    """Run Tesseract OCR on a single image with the given page segmentation mode."""
    config = f'--oem 3 --psm {psm} -l eng'
    try:
        return pytesseract.image_to_string(pil_img, config=config)
    except Exception:
        return ""


def extract_text_from_image(pil_image):
    """
    Extract text from a product label image using multiple OCR strategies.
    Tries several preprocessing approaches (contrast, adaptive threshold, denoising)
    and picks the result with the most readable characters.
    """
    orig = pil_image.convert('RGB')
    w, h = orig.size
    scale = max(1, 2000 // max(w, h))
    if scale > 1:
        orig = orig.resize((w * scale, h * scale), Image.LANCZOS)

    gray = orig.convert('L')
    candidates = []

    # Strategy 1: high contrast + sharpening
    enhanced = ImageEnhance.Contrast(gray).enhance(2.2)
    enhanced = ImageEnhance.Sharpness(enhanced).enhance(2.0)
    candidates.append(_ocr_single_pass(enhanced, psm=6))

    # Strategy 2: adaptive threshold (good for uneven lighting)
    try:
        arr = np.array(gray)
        thresh = cv2.adaptiveThreshold(
            arr, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 10
        )
        candidates.append(_ocr_single_pass(Image.fromarray(thresh), psm=6))
    except Exception:
        pass

    # Strategy 3: denoising + Otsu threshold (good for noisy photos)
    try:
        arr = np.array(gray)
        denoised = cv2.fastNlMeansDenoising(arr, None, 12, 7, 21)
        _, otsu = cv2.threshold(denoised, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        candidates.append(_ocr_single_pass(Image.fromarray(otsu), psm=4))
    except Exception:
        pass

    # Pick the longest cleaned result — more text usually means better OCR
    best = ""
    for raw in candidates:
        cleaned = re.sub(r'[^a-zA-Z,\s\-&()./]', ' ', raw)
        cleaned = re.sub(r'\s+', ' ', cleaned).strip()
        if len(cleaned) > len(best):
            best = cleaned

    if best:
        logger.info(f'OCR extracted {len(best)} chars.')
    else:
        logger.warning('OCR failed.')

    return best


# Keywords used to verify that OCR text actually contains skincare ingredients
_INGREDIENT_KEYWORDS = [
    'ingredients', 'active ingredients', 'inactive ingredients', 'composition',
    'aqua', 'water', 'glycerin', 'glycerine', 'dimethicone', 'tocopherol',
    'panthenol', 'niacinamide', 'retinol', 'hyaluronic', 'squalane',
    'cetearyl', 'stearic', 'palmitic', 'caprylic', 'cetyl',
    'sodium', 'potassium', 'magnesium', 'zinc', 'titanium dioxide',
    'acid', 'sulfate', 'sulphate', 'paraben', 'phenoxyethanol',
    'extract', 'oil', 'butter', 'wax', 'alcohol',
    'fragrance', 'parfum', 'ci ',
    'aloe', 'chamomile', 'centella', 'ceramide', 'peptide',
    'collagen', 'keratin', 'salicylic', 'benzoyl', 'kojic',
    'ascorbic', 'lactic', 'citric', 'glycolic',
    'shea', 'jojoba', 'argan', 'rosehip', 'tea tree',
]
_MIN_KEYWORD_HITS = 3


def _validate_ingredients_text(text: str) -> tuple:
    """Check if the extracted text contains enough skincare-related keywords to be valid."""
    lower = text.lower()
    hits = [kw for kw in _INGREDIENT_KEYWORDS if kw in lower]
    return len(hits) >= _MIN_KEYWORD_HITS, hits


# ── Face scan endpoint ───────────────────────────────────────────────────────

def analyze_skin(request):
    """
    POST /analyze/ — Upload a face photo to detect skin type.

    Pipeline:
      1. Validate that the image contains a real face
      2. Detect face region using OpenCV cascade (fallback: center crop)
      3. Run YOLOv8 to count acne spots
      4. Blur acne from the skin patch so it doesn't confuse the classifier
      5. Classify the clean patch with ResNet50
      6. Adjust probabilities based on acne count
      7. Return skin type, confidence, and product recommendations
    """
    if request.FILES.get('image'):
        try:
            img = Image.open(request.FILES.get('image')).convert('RGB')

            # Step 1: Make sure this is actually a face photo
            is_valid, rejection = validate_face_image(img)
            if not is_valid:
                return JsonResponse({
                    'no_face': True,
                    'error': rejection.get('message', 'Invalid image.'),
                }, status=400)

            img_cv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
            gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
            gray_eq = cv2.equalizeHist(gray)

            # Step 2: Find the face bounding box
            faces = face_cascade.detectMultiScale(gray_eq, scaleFactor=1.1, minNeighbors=4, minSize=(60, 60))

            if len(faces) == 0:
                # No face detected — fall back to a center crop of the image
                logger.info('Cascade missed. Using crop.')
                img_w, img_h = img.size
                cheek_w = int(img_w * 0.7)
                cheek_h = int(img_h * 0.7)
                cheek_x = (img_w - cheek_w) // 2
                cheek_y = (img_h - cheek_h) // 2
                skin_patch = img.crop((cheek_x, cheek_y, cheek_x + cheek_w, cheek_y + cheek_h))
            else:
                # Use the largest detected face, then crop the cheek area
                (x, y, w, h) = max(faces, key=lambda f: f[2] * f[3])
                logger.info('Face detected.')

                cheek_x = x + int(w * 0.6)
                cheek_y = y + int(h * 0.5)
                cheek_w = int(w * 0.3)
                cheek_h = int(h * 0.3)

                cheek_x = min(max(0, cheek_x), img.width - cheek_w)
                cheek_y = min(max(0, cheek_y), img.height - cheek_h)

                skin_patch = img.crop((cheek_x, cheek_y, cheek_x + cheek_w, cheek_y + cheek_h))

            # Step 3: Detect acne spots with YOLOv8
            acne_count = 0
            acne_boxes = []
            if acne_model is not None:
                results = acne_model(img, verbose=False)
                for r in results:
                    for box, cls_id in zip(r.boxes.xyxy, r.boxes.cls):
                        if acne_model.names[int(cls_id)] == 'acne':
                            acne_count += 1
                            acne_boxes.append(box.cpu().numpy().astype(int))
                logger.info(f'Detected {acne_count} acne.')

            # Step 4: Blur acne regions so they don't mislead the skin type classifier
            clean_patch = skin_patch.copy()
            if acne_boxes:
                blurred_patch = skin_patch.filter(ImageFilter.GaussianBlur(radius=15))
                for (ax1, ay1, ax2, ay2) in acne_boxes:
                    if not (ax2 < cheek_x or ax1 > cheek_x + cheek_w or ay2 < cheek_y or ay1 > cheek_y + cheek_h):
                        px1 = max(0, ax1 - cheek_x)
                        py1 = max(0, ay1 - cheek_y)
                        px2 = min(cheek_w, ax2 - cheek_x)
                        py2 = min(cheek_h, ay2 - cheek_y)

                        if px2 > px1 and py2 > py1:
                            acne_crop = blurred_patch.crop((px1, py1, px2, py2))
                            clean_patch.paste(acne_crop, (px1, py1, px2, py2))

            # Step 5: Run the ResNet50 classifier on the cleaned skin patch
            output = face_model(transform(clean_patch).unsqueeze(0))
            probabilities = F.softmax(output, dim=1).squeeze()

            # Step 6: Nudge probabilities if acne was detected
            # (acne correlates with oily/sensitive/combination skin)
            if acne_model is not None and acne_count > 0:
                adjusted = probabilities.clone().detach()

                if acne_count >= 3:
                    adjusted[3] *= 1.08   # oily
                    adjusted[4] *= 1.06   # sensitive
                    adjusted[0] *= 1.04   # combination
                elif acne_count >= 1:
                    adjusted[3] *= 1.04   # oily
                    adjusted[4] *= 1.03   # sensitive

                adjusted = adjusted / adjusted.sum()
                probabilities = adjusted

            confidence_val, predicted_idx = torch.max(probabilities, dim=0)
            confidence_pct = round(confidence_val.item() * 100, 1)
            skin_type = CLASS_NAMES[predicted_idx.item()]

            # Low confidence → ask user to retake the photo
            if confidence_pct < 55.0:
                logger.info('Low confidence scan.')
                return JsonResponse({
                    'low_confidence': True,
                    'confidence': confidence_pct,
                    'error': 'Upload a clearer photo following the guidelines.'
                }, status=200)

            # Step 7: Recommend products that match the detected skin type
            recs = Product.objects.filter(skin_type__in=[skin_type, 'all'])[:4]
            products_list = [{'name': p.name, 'category': p.category, 'image': p.image.url if p.image else None} for p in recs]
            return JsonResponse({
                'skin_type': skin_type,
                'confidence': confidence_pct,
                'acne_detected': acne_count,
                'products': products_list
            })
        except Exception as e:
            logger.error(f'Skin error: {e}')
            return JsonResponse({'error': str(e)}, status=500)
    return JsonResponse({'error': 'Invalid request'}, status=400)


# ── Formula check endpoint ───────────────────────────────────────────────────

@require_POST
def analyze_product(request):
    """
    POST /analyze-product/ — Check if a product's ingredients are safe for a skin type.

    Accepts either:
      - A photo of the ingredient label (OCR extracts text)
      - Pasted ingredient text directly

    Returns a safety score (0-100), flagged harmful/safe ingredients, and
    product recommendations if the score is low.
    """
    skin_type = request.POST.get('skin_type', 'normal').lower()
    ingredients_text = request.POST.get('ingredients_text', '').strip()
    input_source = 'text'

    # If user uploaded an image, extract text from it using OCR
    if request.FILES.get('image'):
        input_source = 'image'
        try:
            pil_img = Image.open(request.FILES.get('image')).convert('RGB')

            is_valid, rejection = validate_ingredient_image(pil_img)
            if not is_valid:
                return JsonResponse(rejection, status=400)

            ingredients_text = extract_text_from_image(pil_img)
        except Exception as e:
            logger.error(f'Image error: {e}')
            return JsonResponse({
                'error': 'Unable to process the uploaded image. Please try a different photo.',
                'ocr_failed': True,
                'input_source': input_source,
            }, status=400)

    # No text at all — can't analyze
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

    # For image uploads, verify the text actually looks like an ingredient list
    if input_source == 'image':
        is_valid, keyword_hits = _validate_ingredients_text(ingredients_text)
        if not is_valid:
            logger.info('Validation failed.')
            return JsonResponse({
                'error': ('We couldn\u2019t detect a valid ingredients list in this image. '
                          'Please ensure the label is clear and readable, '
                          'or paste the ingredients manually below.'),
                'not_ingredients': True,
                'input_source': input_source,
                'ocr_text': ingredients_text[:500],
            }, status=400)

    # ── Score calculation ────────────────────────────────────────────────
    final_score = 50.0
    found_harmful = []
    found_safe = []
    try:
        # Run the ML model for a base safe/unsafe prediction
        result = predict_safety(ingredients_text, skin_type)
        prediction = result['prediction']
        prob = result['probability']

        # Match known harmful and safe ingredients against the text
        clean_txt = ingredients_text.lower()
        found_harmful = [ing.title() for ing in HARMFUL_INGREDIENTS.get(skin_type, []) if ing in clean_txt]
        found_safe = [ing.title() for ing in SAFE_INGREDIENTS.get(skin_type, []) if ing in clean_txt]

        # Extra check: flag pore-clogging ingredients for oily/combination skin
        if skin_type in ['oily', 'combination']:
            comedogenic_list = [
                'beeswax', 'cera alba', 'mineral oil', 'paraffinum liquidum',
                'coconut oil', 'cocos nucifera', 'olive oil', 'olea europaea',
                'cocoa butter', 'theobroma cacao', 'isopropyl myristate',
                'isopropyl palmitate', 'myristyl myristate', 'lanolin',
                'algae extract', 'carrageenan', 'laureth-4', 'cetearyl alcohol'
            ]
            found_comedogenic = [ing.title() for ing in comedogenic_list if ing in clean_txt]

            for ing in found_comedogenic:
                found_harmful.append(f"{ing} (Pore-Clogging Risk)")

        # Extra check: flag harsh irritants for sensitive skin
        if skin_type == 'sensitive':
            irritants_list = [
                'salicylic acid', 'glycolic acid', 'lactic acid',
                'melaleuca alternifolia', 'tea tree', 'cananga odorata',
                'fragrance', 'parfum', 'citrus', 'menthol',
                'peppermint', 'eucalyptus', 'alcohol denat', 'sd alcohol'
            ]
            found_irritants = [ing.title() for ing in irritants_list if ing in clean_txt]

            for ing in found_irritants:
                found_harmful.append(f"{ing} (Harsh Irritant)")

        # Calculate the final safety score based on all findings
        if found_harmful:
            final_score = 40.0 - (len(found_harmful) * 5)
        elif prediction == 1:
            final_score = 75 + (prob * 25) + (len(found_safe) * 2)
        else:
            final_score = prob * 70

    except Exception as e:
        logger.error(f'Prediction Error: {e}')
        final_score = 50.0

    # Clamp score between 5 and 100
    final_score = round(max(5, min(100, final_score)), 1)

    # Suggest safer alternatives if the score is poor
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


# ── Hybrid analysis endpoint ────────────────────────────────────────────────

# Maps quiz answers to skin types (each question has 5 options: a–e)
QUIZ_SCORING = {
    'q1': {'a': 'oily', 'b': 'dry', 'c': 'normal', 'd': 'sensitive', 'e': 'combination'},
    'q2': {'a': 'oily', 'b': 'dry', 'c': 'normal', 'd': 'sensitive', 'e': 'combination'},
    'q3': {'a': 'oily', 'b': 'dry', 'c': 'normal', 'd': 'sensitive', 'e': 'combination'},
}

# Descriptions shown to the user after their skin type is determined
SKIN_DESCRIPTIONS = {
    'oily': 'Your skin produces excess sebum, especially in the T-zone. Lightweight, oil-free products are ideal.',
    'dry': 'Your skin tends to feel tight and may flake. Rich, hydrating products with ceramides work best.',
    'normal': 'Your skin is well-balanced. A simple, consistent routine will keep it healthy.',
    'sensitive': 'Your skin is easily irritated. Look for gentle, fragrance-free products with soothing ingredients.',
    'combination': 'Your skin is oily in the T-zone but dry elsewhere. You may need to treat different areas differently.',
}


@require_POST
def analyze_skin_hybrid(request):
    """
    POST /analyze-hybrid/ — Combine the AI face scan result with the user's
    questionnaire answers for a more accurate skin type determination.

    Decision logic:
      - High AI confidence + quiz agrees         → AI wins ("ai_dominant")
      - High AI confidence + quiz unanimously disagrees → Quiz overrides
      - Low AI confidence + quiz disagrees        → Quiz wins ("questionnaire_preferred")
      - Both agree at any confidence              → Consensus
    """
    try:
        data = json.loads(request.body)
    except json.JSONDecodeError:
        return JsonResponse({'error': 'Invalid JSON'}, status=400)

    ai_skin_type = data.get('ai_skin_type', '').lower()
    ai_confidence = float(data.get('ai_confidence', 0))
    answers = data.get('answers', {})

    if not ai_skin_type:
        return JsonResponse({'error': 'Missing AI result.'}, status=400)

    # Tally up quiz votes — each answer maps to a skin type
    quiz_scores = {'oily': 0, 'dry': 0, 'normal': 0, 'sensitive': 0, 'combination': 0}
    for q_key, mapping in QUIZ_SCORING.items():
        answer = answers.get(q_key, '')
        skin_type_vote = mapping.get(answer, '')
        if skin_type_vote:
            quiz_scores[skin_type_vote] += 2

    quiz_winner = max(quiz_scores, key=quiz_scores.get)
    quiz_max_score = quiz_scores[quiz_winner]

    # Decide final skin type based on AI confidence vs quiz agreement
    if ai_confidence >= 85.0:
        if quiz_winner != ai_skin_type and quiz_max_score == 6:
            final_type = quiz_winner
            method = 'questionnaire_override'
        else:
            final_type = ai_skin_type
            method = 'ai_dominant'
    else:
        if quiz_winner != ai_skin_type:
            final_type = quiz_winner
            method = 'questionnaire_preferred'
        else:
            final_type = ai_skin_type
            method = 'consensus'

    # Blend AI and quiz confidence into a final percentage
    ai_weight = ai_confidence / 100.0
    quiz_confidence = (quiz_max_score / 6.0) * 100
    final_confidence = round((ai_weight * ai_confidence) + ((1 - ai_weight) * quiz_confidence), 1)
    final_confidence = min(99.0, max(50.0, final_confidence))

    logger.info(f'Hybrid final: {final_type}')

    # Recommend products matching the final skin type
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


# ── Favorites toggle endpoint ───────────────────────────────────────────────

@require_POST
def toggle_favorite(request):
    """
    POST /toggle-favorite/ — Add or remove a product from the user's favorites.
    Expects JSON body: {"product_id": 123}
    Returns: {"status": "added"} or {"status": "removed"}
    """
    if not request.user.is_authenticated:
        return JsonResponse({'error': 'Login required.'}, status=401)

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
