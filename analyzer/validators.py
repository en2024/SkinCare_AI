# image validators
import cv2
import numpy as np
import pytesseract
import re
import logging
from PIL import Image, ImageEnhance
from django.conf import settings

logger = logging.getLogger('analyzer')

# set tesseract path
pytesseract.pytesseract.tesseract_cmd = getattr(
    settings, 'TESSERACT_CMD',
    r'C:\Program Files\Tesseract-OCR\tesseract.exe'
)

# load face cascades
_face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
)
_eye_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_eye.xml'
)

# skincare keywords
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

# generic reject response
_REJECT = {
    "valid": False,
    "overridden_confidence": "0%",
}

# hsv skin ranges
_SKIN_RANGES = [
    (np.array([0, 25, 50], dtype=np.uint8),
     np.array([25, 220, 255], dtype=np.uint8)),
    (np.array([0, 15, 40], dtype=np.uint8),
     np.array([40, 200, 255], dtype=np.uint8)),
]

_MIN_SKIN_RATIO = 0.25

# calculate skin ratio
def _compute_skin_ratio(img_bgr):
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_HSV) # Wait, it was BGR2HSV
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    combined_mask = np.zeros(hsv.shape[:2], dtype=np.uint8)

    for lower, upper in _SKIN_RANGES:
        mask = cv2.inRange(hsv, lower, upper)
        combined_mask = cv2.bitwise_or(combined_mask, mask)

    ratio = np.count_nonzero(combined_mask) / combined_mask.size
    logger.info(f'Skin ratio: {ratio:.1%}')
    return ratio

# verify organic texture
def _has_skin_texture(img_bgr):
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (256, 256))

    # laplacian variance
    laplacian = cv2.Laplacian(resized, cv2.CV_64F)
    texture_var = laplacian.var()

    # gabor filter
    kernel = cv2.getGaborKernel(
        (21, 21), sigma=3.0, theta=0, lambd=8.0, gamma=0.5
    )
    gabor = cv2.filter2D(resized, cv2.CV_64F, kernel)
    gabor_energy = np.mean(np.abs(gabor))

    logger.info(f'Texture var: {texture_var:.1f}, Gabor: {gabor_energy:.1f}')

    if texture_var < 10:
        logger.info('Texture too flat.')
        return False
    if texture_var > 3000:
        logger.info('Texture too complex.')
        return False

    return True

# check face size
def _face_is_reasonable_size(face_rect, img_shape):
    (x, y, w, h) = face_rect
    img_h, img_w = img_shape[:2]
    face_area = w * h
    img_area = img_w * img_h
    ratio = face_area / img_area

    logger.info(f'Face area ratio: {ratio:.2%}')
    return ratio >= 0.02

# validate face
def validate_face_image(pil_image):
    img_array = np.array(pil_image.convert('RGB'))
    img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    gray_eq = cv2.equalizeHist(gray)

    # detect faces
    faces = _face_cascade.detectMultiScale(
        gray_eq, scaleFactor=1.1, minNeighbors=3, minSize=(50, 50)
    )

    # retry detection
    if len(faces) == 0:
        faces = _face_cascade.detectMultiScale(
            gray_eq, scaleFactor=1.05, minNeighbors=2, minSize=(30, 30)
        )

    if len(faces) > 0:
        best_face = max(faces, key=lambda f: f[2] * f[3])

        if _face_is_reasonable_size(best_face, img_bgr.shape):
            logger.info('Face detected.')
            return True, None
        else:
            logger.info('Face too small.')

    # color check
    logger.info('Running color check.')
    skin_ratio = _compute_skin_ratio(img_bgr)

    if skin_ratio >= _MIN_SKIN_RATIO:
        # texture check
        logger.info('Running texture check.')

        if _has_skin_texture(img_bgr):
            logger.info('Texture passed.')
            return True, None
        else:
            logger.warning('Texture failed.')
            return False, {
                **_REJECT,
                "message": (
                    "The image has skin-like colors but doesn't appear to be "
                    "a close-up of human skin. Please upload a real photo."
                ),
            }

    # validation failed
    logger.warning('Validation failed.')
    return False, {
        **_REJECT,
        "message": (
            "No human face or skin detected. "
            "Please upload a clear, well-lit photo of your face "
            "or a close-up of your skin."
        ),
    }

# validate ingredients
def validate_ingredient_image(pil_image):
    gray = pil_image.convert('L')

    # enhance image
    enhanced = ImageEnhance.Contrast(gray).enhance(2.2)
    enhanced = ImageEnhance.Sharpness(enhanced).enhance(2.0)

    arr = np.array(enhanced)
    thresh = cv2.adaptiveThreshold(
        arr, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, 31, 10
    )

    config = '--oem 3 --psm 6 -l eng'
    try:
        raw_text = pytesseract.image_to_string(
            Image.fromarray(thresh), config=config
        )
    except Exception:
        raw_text = ""

    # clean text
    cleaned = re.sub(r'[^a-zA-Z,\s\-&()./]', ' ', raw_text)
    cleaned = re.sub(r'\s+', ' ', cleaned).strip()

    if len(cleaned) < 10:
        logger.warning('No text found.')
        return False, {
            **_REJECT,
            "message": (
                "No readable text detected in this image. "
                "Please upload a clear photo of the product ingredients list."
            ),
        }

    # count keywords
    lower = cleaned.lower()
    hits = [kw for kw in _INGREDIENT_KEYWORDS if kw in lower]

    if len(hits) < _MIN_KEYWORD_HITS:
        logger.warning('Not enough keywords.')
        return False, {
            **_REJECT,
            "message": (
                "This image does not contain a skincare ingredients list. "
                "Please upload a clear photo of the product label, "
                "or paste the ingredients manually."
            ),
        }

    logger.info('Keywords matched.')
    return True, None
