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
from PIL import Image, ImageFilter, ImageEnhance
from sklearn.feature_extraction.text import TfidfVectorizer
import sklearn
import os
import re
import pickle
import json
import numpy as np
import pytesseract

# ─── إعداد مسار Tesseract OCR ───────────────────────────────────────────────
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# ─── 1. Face Analysis Model (Vision AI) ──────────────────────────────────────
CLASS_NAMES = ['dry', 'normal', 'oily']
face_model = models.resnet50(weights=None)
face_model.fc = nn.Linear(face_model.fc.in_features, 3)
face_model_path = os.path.join(settings.BASE_DIR, 'analyzer', 'model.pth')

try:
    face_model.load_state_dict(torch.load(face_model_path, map_location=torch.device('cpu')), strict=False)
    face_model.eval()
except Exception as e:
    print(f"Face Model Loading Error: {e}")

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# ─── 2. NEW Ingredients AI Logic (The Hybrid System) ──────────────────────────
ML_MODELS_DIR = os.path.join(settings.BASE_DIR, 'analyzer', 'ml_models')

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
    except:
        return ""

# ─── 4. General Website Views ────────────────────────────────────────────────
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

# ─── 5. Admin Panel Logic ───────────────────────────────────────────────────
@login_required(login_url='login')
def admin_dashboard(request):
    if not request.user.is_staff: return redirect('home')
    return render(request, 'admin.html', {'products': Product.objects.all()})

@login_required(login_url='login')
def add_product(request):
    if request.method == 'POST' and request.user.is_staff:
        Product.objects.create(
            name=request.POST.get('name'), 
            category=request.POST.get('category'),
            skin_type=request.POST.get('skin_type'), 
            description=request.POST.get('description'),
            image=request.FILES.get('image')
        )
    return redirect('/admin-dashboard/')

@login_required(login_url='login')
def edit_product(request):
    if request.method == 'POST' and request.user.is_staff:
        p = Product.objects.get(id=request.POST.get('product_id'))
        p.name = request.POST.get('name')
        p.category = request.POST.get('category')
        p.skin_type = request.POST.get('skin_type')
        p.description = request.POST.get('description')
        if request.FILES.get('image'):
            p.image = request.FILES.get('image')
        p.save()
    return redirect('/admin-dashboard/')

@login_required(login_url='login')
def delete_product(request):
    if request.method == 'POST' and request.user.is_staff:
        Product.objects.filter(id=request.POST.get('product_id')).delete()
    return redirect('/admin-dashboard/')

# ─── 6. AI Endpoints (Face & Product Analysis) ───────────────────────────────

@csrf_exempt
def analyze_skin(request):
    """تحليل صورة الوجه لتحديد نوع البشرة"""
    if request.method == 'POST' and request.FILES.get('image'):
        try:
            img = Image.open(request.FILES.get('image')).convert('RGB')
            output = face_model(transform(img).unsqueeze(0))
            predicted = torch.argmax(output, dim=1).item()
            skin_type = CLASS_NAMES[predicted]
            
            recs = Product.objects.filter(skin_type__in=[skin_type, 'all'])[:4]
            products_list = [{'name': p.name, 'category': p.category, 'image': p.image.url if p.image else None} for p in recs]
            return JsonResponse({'skin_type': skin_type, 'products': products_list})
        except Exception as e:
            return JsonResponse({'error': str(e)}, status=500)
    return JsonResponse({'error': 'Invalid request'}, status=400)

@csrf_exempt
def analyze_product(request):
    """تحليل المكونات باستخدام الموديلات الجديدة (93% دقة) ونظام الأمان اليدوي"""
    if request.method != 'POST':
        return JsonResponse({'error': 'Invalid request'}, status=400)

    skin_type = request.POST.get('skin_type', 'normal').lower()
    ingredients_text = request.POST.get('ingredients_text', '').strip()

    # 1. OCR (لو اليوزر رفع صورة للمكونات)
    if request.FILES.get('image'):
        pil_img = Image.open(request.FILES.get('image'))
        ingredients_text = extract_text_from_image(pil_img)

    if not ingredients_text:
        return JsonResponse({'error': 'No ingredients text found.'}, status=400)

    # 2. AI Model Selection & Prediction
    final_score = 50.0
    try:
        # تحديد مسار الموديل والـ Vectorizer بناءً على نوع البشرة
        model_path = os.path.join(ML_MODELS_DIR, f'model_{skin_type}.pkl')
        vec_path = os.path.join(ML_MODELS_DIR, f'vectorizer_{skin_type}.pkl')
        
        with open(model_path, 'rb') as f: model = pickle.load(f)
        with open(vec_path, 'rb') as f: vectorizer = pickle.load(f)

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
        print(f"Prediction System Error: {e}")
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