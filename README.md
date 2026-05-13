# 🧴 AI SkinCare Safety Analyzer

> An AI-powered web application that analyzes skin type from facial photos and evaluates skincare product ingredient safety using machine learning.

**Final Year Graduation Project** — Arab Open University, Egypt  
© 2026 Engy Nabil Safwat

---

## ✨ Features

### 🤖 AI Skin Type & Acne Detection
Upload a face photo and our **ResNet-50** deep learning model instantly classifies your skin as **Oily**, **Dry**, or **Normal**. Simultaneously, a **YOLOv8** model detects acne spots, allowing for a more precise analysis and better product recommendations.


### 🔬 Ingredient Safety Analysis (Hybrid AI System)
Analyze any skincare product's ingredients list in two ways:
- **OCR Scan** — snap a photo of the product label; Tesseract extracts the text automatically
- **Text Input** — paste the ingredients manually

The system uses a **hybrid approach**:
1. **ML Classifier** (TF-IDF + trained model per skin type, ~93% accuracy) predicts overall safety
2. **Rule-Based Safety Shield** cross-references a curated database of harmful & safe ingredients for your specific skin type
3. Combined scoring produces a final **Safety Index** (0–100%)

### 📦 Curated Product Catalog
Browse **40+ skincare products** across 4 categories (Moisturizers, Cleansers, Serums, Sunscreens), each tagged with compatible skin types, benefits, and usage instructions. Filterable by category with animated transitions.

### 🛠️ Admin Dashboard
Staff users can manage the product catalog (add, edit, delete) through a dedicated admin interface with inline editing.

### 🌐 Bilingual Support (EN/AR)
Full English ↔ Arabic translation toggle on every page with proper RTL layout switching.

### 🌙 Accessibility Features
- **Dark Mode** with persistent preference (localStorage)
- **Adjustable Font Size** (3 levels)
- **Responsive Design** — fully mobile-friendly

---

## 🏗️ Tech Stack

| Layer | Technology |
|-------|-----------|
| **Backend** | Django 6.0 (Python) |
| **Database** | SQLite3 |
| **Deep Learning** | PyTorch (ResNet-50) & Ultralytics (YOLOv8) |
| **ML / NLP** | scikit-learn — TF-IDF Vectorizer + trained Random Forest classifiers |
| **OCR** | Tesseract OCR via pytesseract |
| **Image Processing** | Pillow, OpenCV (Haar Cascades for face detection), NumPy |
| **Frontend** | Vanilla HTML/CSS/JS with Google Fonts (Jost + Cormorant Garamond) |

| **Icons** | Material Icons Outlined |

---

## 📁 Project Structure

```
SkinCare_AI/
├── core/                       # Django project configuration
│   ├── settings.py
│   ├── urls.py
│   ├── wsgi.py
│   └── asgi.py
├── analyzer/                   # Main Django app
│   ├── models.py               # Product model (name, category, skin_type, etc.)
│   ├── views.py                # Page views, auth, admin CRUD, AI API endpoints
│   ├── urls.py                 # URL routing
│   ├── admin.py                # Django admin registration
│   ├── model.pth               # Pre-trained ResNet-50 weights (~90 MB)
│   └── ml_models/              # Trained sklearn classifiers & vectorizers
│       ├── skincare_model.pkl  # Unified Random Forest model
│       ├── train_model.py      # Script to train the safety model
│       └── predictor.py        # Logic to load model and predict

├── templates/                  # Django HTML templates
│   ├── index.html              # Home / landing page
│   ├── analyzer.html           # AI analyzer (face scan + formula check)
│   ├── catalog.html            # Product catalog with filters
│   ├── login.html              # Login & registration
│   └── admin.html              # Admin dashboard (product management)
├── static/
│   └── image/
│       └── banner.jpg          # Education section banner
├── media/                      # Uploaded product images
├── add_products.py             # Database seeder script (40+ products)
├── manage.py                   # Django management CLI
├── Requirements.txt            # Python dependencies
└── .gitignore
```

---

## 🚀 Getting Started

### Prerequisites

- **Python** 3.10+
- **Tesseract OCR** — [Download here](https://github.com/UB-Mannheim/tesseract/wiki)
- **Git** (optional)

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/YOUR_USERNAME/SkinCare_AI.git
   cd SkinCare_AI
   ```

2. **Create and activate a virtual environment**
   ```bash
   python -m venv venv

   # Windows
   venv\Scripts\activate

   # macOS / Linux
   source venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r Requirements.txt
   ```

4. **Configure Tesseract OCR path**
   
   In `analyzer/views.py`, update line 23 to match your Tesseract installation:
   ```python
   pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
   ```

5. **Apply database migrations**
   ```bash
   python manage.py migrate
   ```

6. **Seed the product catalog**
   ```bash
   python add_products.py
   ```

7. **Create a superuser (for admin access)**
   ```bash
   python manage.py createsuperuser
   ```

8. **Run the development server**
   ```bash
   python manage.py runserver
   ```

9. **Open in browser**: [http://127.0.0.1:8000](http://127.0.0.1:8000)

---

## 📖 Usage

### Skin Type Analysis
1. Navigate to **AI Analyzer** → **Face Scan** tab
2. Upload a clear, well-lit photo of your face (no makeup for best results)
3. Click **"Analyze My Skin"**
4. View your detected skin type and personalized product recommendations

### Product Safety Check
1. Navigate to **AI Analyzer** → **Formula Check** tab
2. Select your skin type from the pills
3. Either snap a photo of the product ingredient label **or** paste the ingredients text
4. Click **"Analyze My Formula"** to get a safety score
5. View the Safety Index, detected harmful/safe ingredients, and alternative product suggestions

### Admin Panel
1. Log in with a staff/superuser account
2. Navigate to **Admin Dashboard** (`/admin-dashboard/`)
3. Add, edit, or delete products from the catalog

---

## 🧠 AI Models

### Face Classification Model
- **Architecture**: ResNet-50 (transfer learning)
- **Classes**: Oily, Dry, Normal
- **Input**: 224×224 RGB face image
- **Normalization**: ImageNet mean/std
- **File**: `analyzer/model.pth` (~90 MB)

### Acne Detection Model
- **Architecture**: YOLOv8 (Ultralytics)
- **Features**: Detects acne spots and inflammatory marks
- **File**: `analyzer/ml_models/acne_yolov8.pt`

### Ingredient Safety Classifiers
- **Architecture**: Multi-output Random Forest Classifier
- **Feature extraction**: TF-IDF Vectorization
- **Accuracy**: ~93% on test set
- **File**: `analyzer/ml_models/ingredients/skincare_model.pkl`


---

## 🔗 API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/` | Home page |
| `GET` | `/analyzer/` | AI Analyzer page |
| `GET` | `/catalog/` | Product catalog |
| `POST` | `/analyze/` | Skin type analysis API (accepts `image` file) |
| `POST` | `/analyze-product/` | Ingredient safety API (accepts `skin_type`, `image`/`ingredients_text`) |
| `GET` | `/login/` | Login / Register page |
| `POST` | `/login/` | Authenticate user |
| `GET` | `/logout/` | Logout and redirect |
| `POST` | `/register/` | Create new user |
| `GET` | `/admin-dashboard/` | Admin product management (staff only) |
| `POST` | `/admin-dashboard/add/` | Add product (staff only) |
| `POST` | `/admin-dashboard/edit/` | Edit product (staff only) |
| `POST` | `/admin-dashboard/delete/` | Delete product (staff only) |

---

## 📦 Dependencies

```
django>=4.2.0
pillow>=10.0.0
numpy>=1.19.0
opencv-python>=4.5.0
pytesseract>=0.3.10
python-dotenv>=1.0.0
torch>=2.0.0
torchvision>=0.15.0
scikit-learn>=1.3.0
ultralytics>=8.0.0
```


---

## 📸 Pages

| Page | Route | Description |
|------|-------|-------------|
| **Home** | `/` | Hero section, feature highlights, education videos |
| **AI Analyzer** | `/analyzer/` | Dual-tab interface for face scan + formula check |
| **Catalog** | `/catalog/` | Filterable product grid with hover effects |
| **Login** | `/login/` | User/Admin authentication with role switching |
| **Admin** | `/admin-dashboard/` | Product CRUD with inline editing |

---

## 🤝 Contributing

This is a graduation project. Feature suggestions and bug reports are welcome via GitHub Issues.

---

## 📄 License

This project is developed as an academic graduation project. All rights reserved.
