import logging
import os
import re
import pickle

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
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# ─── 2. Ingredients AI Logic (The Hybrid System) ─────────────────────────────
ML_MODELS_DIR = os.path.join(settings.BASE_DIR, 'analyzer', 'ml_models')

# ─── Cache for loaded ML models (load once, reuse) ───────────────────────────
_ml_model_cache = {}
_ml_vectorizer_cache = {}


def _get_ml_model(skin_type):
    """Load and cache the ML model + vectorizer for a given skin type."""
    if skin_type not in _ml_model_cache:
        model_path = os.path.join(ML_MODELS_DIR, f'model_{skin_type}.pkl')
        vec_path = os.path.join(ML_MODELS_DIR, f'vectorizer_{skin_type}.pkl')
        with open(model_path, 'rb') as f:
            _ml_model_cache[skin_type] = pickle.load(f)
        with open(vec_path, 'rb') as f:
            _ml_vectorizer_cache[skin_type] = pickle.load(f)
        logger.info(f'ML model for "{skin_type}" loaded and cached.')
    return _ml_model_cache[skin_type], _ml_vectorizer_cache[skin_type]


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


# ─── 3. OCR Helper (Pre-processing for images) ───────────────────────────────
def extract_text_from_image(pil_image):
    orig = pil_image.convert('L')
    w, h = orig.size
    # تكبير الصورة وتحسين التباين لزيادة دقة الـ OCR
    v = orig.resize((w * 2, h * 2), Image.LANCZOS)
    v = ImageEnhance.Contrast(v).enhance(2.0)

    config = r'--oem 3 --psm 6 -l eng'
    try:
        raw_text = pytesseract.image_to_string(v, config=config)
        cleaned = re.sub(r'[^a-zA-Z,\s\-&().]', ' ', raw_text)
        cleaned = re.sub(r'\s+', ' ', cleaned).strip()
        return cleaned
    except Exception as e:
        logger.warning(f'OCR extraction failed: {e}')
        return ""


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

            # ── Step 0: Face Detection Gate ──────────────────────────────
            # Reject non-face images before any AI processing
            img_cv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
            gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60))

            if len(faces) == 0:
                logger.info('Face detection gate: no face found in image.')
                return JsonResponse({
                    'error': 'No face detected. Please upload a clear photo of your face.',
                    'no_face': True
                }, status=200)

            # Crop to the largest detected face for more accurate analysis
            (x, y, w, h) = max(faces, key=lambda f: f[2] * f[3])
            # Add padding around the face (20% on each side)
            pad = int(0.2 * max(w, h))
            fx1 = max(0, x - pad)
            fy1 = max(0, y - pad)
            fx2 = min(img.width, x + w + pad)
            fy2 = min(img.height, y + h + pad)
            img = img.crop((fx1, fy1, fx2, fy2))
            logger.info(f'Face detected and cropped: ({fx1},{fy1}) to ({fx2},{fy2}).')

            # ── Step 1: YOLO detects acne locations ──────────────────────
            acne_count = 0
            acne_boxes = []
            if acne_model is not None:
                results = acne_model(img, verbose=False)
                for r in results:
                    for box, cls_id in zip(r.boxes.xyxy, r.boxes.cls):
                        if acne_model.names[int(cls_id)] == 'acne':
                            acne_count += 1
                            # Store bounding box as (x1, y1, x2, y2)
                            acne_boxes.append(box.cpu().numpy().astype(int))
                logger.info(f'YOLO detected {acne_count} acne spot(s).')

            # ── Step 2: Create a "clean" image for ResNet ────────────────
            # Blur out every acne region so ResNet classifies pure skin
            # texture without acne spots confusing it
            clean_img = img.copy()
            if acne_boxes:
                from PIL import ImageFilter
                # Create a heavily blurred version of the entire image
                blurred = img.filter(ImageFilter.GaussianBlur(radius=15))
                for (x1, y1, x2, y2) in acne_boxes:
                    # Paste the blurred patch over each acne region
                    patch = blurred.crop((x1, y1, x2, y2))
                    clean_img.paste(patch, (x1, y1, x2, y2))
                logger.info(f'Masked {len(acne_boxes)} acne region(s) for clean skin analysis.')

            # ── Step 3: ResNet classifies the CLEANED image ──────────────
            output = face_model(transform(clean_img).unsqueeze(0))
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
            if confidence_pct < 75.0:
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
    """تحليل المكونات باستخدام الموديلات الجديدة (93% دقة) ونظام الأمان اليدوي"""
    skin_type = request.POST.get('skin_type', 'normal').lower()
    ingredients_text = request.POST.get('ingredients_text', '').strip()

    # 1. OCR (لو اليوزر رفع صورة للمكونات)
    if request.FILES.get('image'):
        pil_img = Image.open(request.FILES.get('image'))
        ingredients_text = extract_text_from_image(pil_img)

    if not ingredients_text:
        return JsonResponse({'error': 'No ingredients text found.'}, status=400)

    # 2. AI Model Selection & Prediction (cached)
    final_score = 50.0
    try:
        model, vectorizer = _get_ml_model(skin_type)

        # Preprocessing (نفس طريقة Colab)
        clean_txt = ingredients_text.lower()
        for p in ['visit the', 'no info', 'boutique']:
            clean_txt = clean_txt.replace(p, '')

        vec_input = vectorizer.transform([clean_txt]).toarray()
        prediction = model.predict(vec_input)[0]
        prob = model.predict_proba(vec_input)[0][1]

        # 3. Rule-Based Safety Shield (The Manual Fix)
        found_harmful = [ing.title() for ing in HARMFUL_INGREDIENTS.get(skin_type, []) if ing in clean_txt]
        found_safe = [ing.title() for ing in SAFE_INGREDIENTS.get(skin_type, []) if ing in clean_txt]

        # الـ Logic النهائي للنتيجة
        if found_harmful:
            # لو في مواد ضارة صريحة، المنتج يسقط فوراً (أمان طبي)
            final_score = 40.0 - (len(found_harmful) * 5)
        elif prediction == 1:
            # لو الـ AI قال آمن، بنديله درجة عالية بناءً على نسبة ثقة الـ AI
            final_score = 75 + (prob * 25) + (len(found_safe) * 2)
        else:
            # لو الـ AI قال غير آمن
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
        'ocr_text': ingredients_text[:300]
    })


# ─── 5. Hybrid Skin Analysis (AI + Questionnaire) ────────────────────────────

# Questionnaire answer → skin-type point mapping
QUIZ_SCORING = {
    # Q1: How does your skin feel 30 min after washing?
    'q1': {'a': 'oily', 'b': 'dry', 'c': 'normal'},
    # Q2: How often does your face get shiny by midday?
    'q2': {'a': 'oily', 'b': 'dry', 'c': 'normal'},
    # Q3: How does your skin react to new products?
    'q3': {'a': 'oily', 'b': 'dry', 'c': 'normal'},
}

SKIN_DESCRIPTIONS = {
    'oily': 'Your skin produces excess sebum, especially in the T-zone. Lightweight, oil-free products are ideal.',
    'dry': 'Your skin tends to feel tight and may flake. Rich, hydrating products with ceramides work best.',
    'normal': 'Your skin is well-balanced. A simple, consistent routine will keep it healthy.',
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

    if not ai_skin_type or not answers:
        return JsonResponse({'error': 'Missing AI result or questionnaire answers.'}, status=400)

    # ── Questionnaire scoring ────────────────────────────────────────────
    quiz_scores = {'oily': 0, 'dry': 0, 'normal': 0}
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
