import logging
import os
import re
import pickle

import numpy as np
import pytesseract
import torch
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
CLASS_NAMES = ['dry', 'normal', 'oily']
face_model = models.resnet50(weights=None)
face_model.fc = nn.Linear(face_model.fc.in_features, 3)
face_model_path = os.path.join(settings.BASE_DIR, 'analyzer', 'model.pth')

try:
    face_model.load_state_dict(torch.load(face_model_path, map_location=torch.device('cpu')), strict=False)
    face_model.eval()
    logger.info('Face analysis model loaded successfully.')
except Exception as e:
    logger.error(f'Face Model Loading Error: {e}')

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
    """تحليل صورة الوجه لتحديد نوع البشرة"""
    if request.FILES.get('image'):
        try:
            img = Image.open(request.FILES.get('image')).convert('RGB')
            output = face_model(transform(img).unsqueeze(0))
            predicted = torch.argmax(output, dim=1).item()
            skin_type = CLASS_NAMES[predicted]

            recs = Product.objects.filter(skin_type__in=[skin_type, 'all'])[:4]
            products_list = [{'name': p.name, 'category': p.category, 'image': p.image.url if p.image else None} for p in recs]
            return JsonResponse({'skin_type': skin_type, 'products': products_list})
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
