from django.shortcuts import render, redirect
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.conf import settings
from django.contrib.auth import authenticate, login, logout
from django.contrib.auth.decorators import login_required
from .models import Product
from django.contrib.auth.models import User
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image, ImageFilter, ImageEnhance, ImageOps
import os
import re
import json
import numpy as np
import pytesseract
import tensorflow as tf
from tensorflow.keras.preprocessing.text import tokenizer_from_json
from tensorflow.keras.preprocessing.sequence import pad_sequences

pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# ─── 1. Face Analysis Model ───────────────────────────────────────────────────
CLASS_NAMES = ['dry', 'normal', 'oily']
face_model = models.resnet50(weights=None)
face_model.fc = nn.Linear(face_model.fc.in_features, 3)
model_path = os.path.join(settings.BASE_DIR, 'analyzer', 'model.pth')
try:
    face_model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')), strict=False)
    face_model.eval()
except Exception as e:
    print(f"Face Model Loading Error: {e}")

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# ─── 2. Ingredients AI Model ──────────────────────────────────────────────────
ANALYZER_DIR = os.path.join(settings.BASE_DIR, 'analyzer')
MAX_LEN = 100
try:
    ai_model = tf.keras.models.load_model(os.path.join(ANALYZER_DIR, 'balanced_classifier_best.keras'))
    with open(os.path.join(ANALYZER_DIR, 'tokenizer_config.json'), 'r') as f:
        ai_tokenizer = tokenizer_from_json(json.dumps(json.load(f)))
    with open(os.path.join(ANALYZER_DIR, 'best_thresholds.json'), 'r') as f:
        best_thresholds = json.load(f)
except Exception as e:
    ai_model, ai_tokenizer = None, None
    best_thresholds = {"Oily": 0.5, "Dry": 0.3, "Sensitive": 0.45, "Combination": 0.5, "Normal": 0.4}

# ─── 3. Knowledge Base ────────────────────────────────────────────────────────
HARMFUL_INGREDIENTS = {
    'oily': [
        'alcohol denat', 'isopropyl myristate', 'coconut oil', 'lanolin',
        'paraffin', 'mineral oil', 'cocoa butter', 'wheat germ oil',
        'sodium lauryl sulfate', 'bismuth oxychloride',
    ],
    'dry': [
        'alcohol denat', 'benzoyl peroxide', 'salicylic acid', 'fragrance',
        'sulfates', 'sodium lauryl sulfate', 'isopropyl alcohol', 'menthol', 'witch hazel',
    ],
    'sensitive': [
        # Alcohols
        'alcohol', 'alcohol denat', 'isopropyl alcohol', 'benzyl alcohol',
        # Fragrance / perfume
        'fragrance', 'parfum', 'essential oil', 'essential oils',
        # Fragrance chemicals found on labels
        'limonene', 'linalool', 'citral', 'geraniol', 'citronellol',
        'farnesol', 'eugenol', 'coumarin', 'benzyl benzoate',
        'benzyl cinnamate', 'cinnamal', 'isoeugenol',
        # Irritating essential oils
        'lavandula angustifolia', 'lavender', 'pelargonium graveolens',
        'geranium', 'citrus aurantium', 'neroli', 'santalum',
        'hippophae', 'sea buckthorn',
        # Preservatives
        'parabens', 'methylparaben', 'propylparaben', 'phenoxyethanol',
        'methylisothiazolinone', 'methylchloroisothiazolinone', 'dmdm hydantoin',
        # Sulfates
        'sodium lauryl sulfate', 'sls', 'sodium laureth sulfate', 'sles',
        # Acids & retinoids
        'lactic acid', 'salicylic acid', 'glycolic acid', 'retinol', 'retinoic acid',
        # Other irritants
        'benzoyl peroxide', 'formaldehyde', 'oxybenzone', 'cinnamate',
        'triethanolamine', 'diethanolamine', 'propylene glycol',
        'menthol', 'camphor', 'witch hazel', 'lanolin',
        'cocamidopropyl betaine', 'artificial color', 'fd&c', 'd&c',
    ],
    'normal': [
        'alcohol denat', 'fragrance', 'parabens', 'sodium lauryl sulfate',
    ],
    'combination': [
        'alcohol denat', 'coconut oil', 'mineral oil', 'cocoa butter', 'isopropyl myristate',
    ],
}

SAFE_INGREDIENTS = {
    'oily':        ['salicylic acid', 'niacinamide', 'hyaluronic acid', 'zinc', 'tea tree', 'retinol', 'benzoyl peroxide'],
    'dry':         ['hyaluronic acid', 'glycerin', 'ceramide', 'squalane', 'shea butter', 'urea', 'panthenol'],
    'sensitive':   ['aloe vera', 'chamomile', 'oat', 'ceramide', 'panthenol', 'allantoin', 'centella asiatica', 'bisabolol', 'zinc oxide', 'titanium dioxide'],
    'normal':      ['hyaluronic acid', 'glycerin', 'vitamin c', 'niacinamide', 'retinol'],
    'combination': ['hyaluronic acid', 'niacinamide', 'ceramide', 'salicylic acid', 'glycerin'],
}


# ─── 4. OCR Helper ────────────────────────────────────────────────────────────
def extract_text_from_image(pil_image):
    """
    Try multiple preprocessing variants + Tesseract configs.
    Returns the best (longest clean English) text found.
    """
    orig = pil_image.convert('L')
    w, h = orig.size

    def letter_count(txt):
        return len(re.sub(r'[^a-zA-Z]', '', txt))

    variants = []
    # 1. Plain grayscale x3
    variants.append(orig.resize((w * 3, h * 3), Image.LANCZOS))
    # 2. High contrast + sharpen x3
    v2 = ImageEnhance.Contrast(orig).enhance(2.5)
    v2 = ImageEnhance.Sharpness(v2).enhance(2.0)
    variants.append(v2.resize((w * 3, h * 3), Image.LANCZOS))
    # 3. Binarize (threshold)
    arr = np.array(orig.resize((w * 2, h * 2), Image.LANCZOS))
    threshold = int(np.mean(arr))
    variants.append(Image.fromarray(np.where(arr > threshold, 255, 0).astype(np.uint8)))
    # 4. Inverted binarize
    variants.append(Image.fromarray(np.where(arr <= threshold, 255, 0).astype(np.uint8)))
    # 5. Unsharp mask + contrast x2
    v5 = orig.filter(ImageFilter.UnsharpMask(radius=2, percent=150, threshold=3))
    v5 = ImageEnhance.Contrast(v5).enhance(2.0)
    variants.append(v5.resize((w * 2, h * 2), Image.LANCZOS))

    configs = [
        r'--oem 3 --psm 6 -l eng',
        r'--oem 3 --psm 4 -l eng',
        r'--oem 3 --psm 3 -l eng',
        r'--oem 1 --psm 6 -l eng',
        r'--oem 3 --psm 11 -l eng',
    ]

    best_text = ''
    for variant in variants:
        for cfg in configs:
            try:
                raw = pytesseract.image_to_string(variant, config=cfg)
                if letter_count(raw) > letter_count(best_text):
                    best_text = raw
            except Exception:
                continue

    cleaned = re.sub(r'[^a-zA-Z,\s\-&().]', ' ', best_text)
    cleaned = re.sub(r'\s+', ' ', cleaned).strip()
    return cleaned


# ─── 5. Rule-Based Analysis ───────────────────────────────────────────────────
def rule_based_analysis(ingredients_text, skin_type):
    """
    PRIMARY safety engine. Rule score is always trusted over AI when harmful
    ingredients are found. Sensitive skin has a hard cap of 45 if any harmful
    ingredient is detected.
    """
    text_lower = ingredients_text.lower()

    harmful_found = [
        ing.title()
        for ing in HARMFUL_INGREDIENTS.get(skin_type, [])
        if ing.lower() in text_lower
    ]
    safe_found = [
        ing.title()
        for ing in SAFE_INGREDIENTS.get(skin_type, [])
        if ing.lower() in text_lower
    ]

    if skin_type == 'sensitive':
        score = 88 - (len(harmful_found) * 22) + (len(safe_found) * 3)
        if harmful_found:
            score = min(score, 45)   # hard "not safe" cap for sensitive skin
    else:
        score = 85 - (len(harmful_found) * 15) + (len(safe_found) * 5)

    return {
        'safety_score':        max(0, min(100, score)),
        'harmful_ingredients': harmful_found,
        'safe_ingredients':    safe_found,
    }


# ─── 6. Standard Views ────────────────────────────────────────────────────────
def home(request):
    return render(request, 'index.html')

def analyzer_page(request):
    return render(request, 'analyzer.html')

def catalog(request):
    products = Product.objects.all()
    return render(request, 'catalog.html', {'products': products, 'MEDIA_URL': settings.MEDIA_URL})

def user_login(request):
    if request.method == 'POST':
        user = authenticate(request, username=request.POST.get('username'), password=request.POST.get('password'))
        if user:
            login(request, user)
            return redirect('admin_dashboard') if user.is_staff else redirect('home')
    return render(request, 'login.html')

def user_logout(request):
    logout(request)
    return redirect('home')

def register_user(request):
    if request.method == 'POST':
        User.objects.create_user(
            username=request.POST.get('username'),
            password=request.POST.get('password')
        )
    return redirect('login')

@login_required(login_url='login')
def admin_dashboard(request):
    if not request.user.is_staff:
        return redirect('home')
    return render(request, 'admin.html', {'products': Product.objects.all()})

@login_required(login_url='login')
def add_product(request):
    if request.method == 'POST' and request.user.is_staff:
        Product.objects.create(
            name=request.POST.get('name'), category=request.POST.get('category'),
            skin_type=request.POST.get('skin_type'), description=request.POST.get('description'),
            image=request.FILES.get('image')
        )
    return redirect('/admin-dashboard/')

@login_required(login_url='login')
def edit_product(request):
    if request.method == 'POST' and request.user.is_staff:
        p = Product.objects.get(id=request.POST.get('product_id'))
        p.name = request.POST.get('name')
        if request.FILES.get('image'):
            p.image = request.FILES.get('image')
        p.save()
    return redirect('/admin-dashboard/')

@login_required(login_url='login')
def delete_product(request):
    if request.method == 'POST' and request.user.is_staff:
        Product.objects.filter(id=request.POST.get('product_id')).delete()
    return redirect('/admin-dashboard/')


# ─── 7. AI Endpoints ──────────────────────────────────────────────────────────
@csrf_exempt
def analyze_skin(request):
    if request.method == 'POST' and request.FILES.get('image'):
        try:
            img = Image.open(request.FILES.get('image')).convert('RGB')
            predicted = torch.argmax(face_model(transform(img).unsqueeze(0)), dim=1).item()
            skin_type = CLASS_NAMES[predicted]
            recs = Product.objects.filter(skin_type__in=[skin_type, 'all'])[:4]
            products_list = [
                {'name': p.name, 'category': p.category, 'image': p.image.url if p.image else None}
                for p in recs
            ]
            return JsonResponse({'skin_type': skin_type, 'products': products_list})
        except Exception as e:
            return JsonResponse({'error': str(e)}, status=500)
    return JsonResponse({'error': 'Invalid request'}, status=400)


@csrf_exempt
def analyze_product(request):
    if request.method != 'POST':
        return JsonResponse({'error': 'Invalid request'}, status=400)

    skin_type = request.POST.get('skin_type', 'normal').lower()
    ingredients_text = request.POST.get('ingredients_text', '').strip()

    # ── Step 1: OCR ───────────────────────────────────────────────────────────
    if request.FILES.get('image'):
        try:
            pil_img = Image.open(request.FILES.get('image'))
            ocr_text = extract_text_from_image(pil_img)
        except Exception as e:
            return JsonResponse({'error': f'Image processing error: {str(e)}'}, status=400)

        # Need at least 20 real letters to consider OCR successful
        if len(re.sub(r'[^a-zA-Z]', '', ocr_text)) >= 20:
            ingredients_text = ocr_text
        else:
            # OCR failed — ask user to paste manually
            return JsonResponse({
                'ocr_failed': True,
                'message': (
                    'Could not read the image clearly. '
                    'Please paste the ingredients list manually in the text box for accurate results.'
                )
            }, status=200)

    if not ingredients_text:
        return JsonResponse({'error': 'Please upload an image or paste the ingredients text.'}, status=400)

    # ── Step 2: Rule-based (PRIMARY) ──────────────────────────────────────────
    rule_res    = rule_based_analysis(ingredients_text, skin_type)
    rule_score  = rule_res['safety_score']
    final_score = rule_score

    # ── Step 3: AI model (SECONDARY — never overrides a "not safe" verdict) ───
    if ai_model and ai_tokenizer and not rule_res['harmful_ingredients'] and rule_score >= 70:
        try:
            seq     = ai_tokenizer.texts_to_sequences([ingredients_text.lower()])
            padded  = pad_sequences(seq, maxlen=MAX_LEN, padding='post')
            preds   = ai_model.predict(padded, verbose=0)[0]
            idx     = {"oily": 0, "dry": 1, "sensitive": 2, "combination": 3, "normal": 4}.get(skin_type, 4)
            ai_prob = float(preds[idx])
            thresh  = best_thresholds.get(skin_type.capitalize(), 0.5)

            ai_score = (
                80 + ((ai_prob - thresh) / (1 - thresh)) * 20
                if ai_prob >= thresh
                else (ai_prob / thresh) * 79
            )
            # Rule dominates: 65% rule, 35% AI
            final_score = (rule_score * 0.65) + (ai_score * 0.35)
        except Exception:
            final_score = rule_score

    final_score = round(max(0, min(100, final_score)), 1)

    # ── Step 4: Recommendations ───────────────────────────────────────────────
    recs = Product.objects.filter(skin_type__in=[skin_type, 'all'])[:3] if final_score < 70 else []
    recommendations = [
        {'name': p.name, 'category': p.category, 'image': p.image.url if p.image else None}
        for p in recs
    ]

    return JsonResponse({
        'safety_score':        final_score,
        'harmful_ingredients': rule_res['harmful_ingredients'],
        'safe_ingredients':    rule_res.get('safe_ingredients', []),
        'skin_type':           skin_type,
        'recommendations':     recommendations,
        'ocr_text':            ingredients_text[:300],
    })