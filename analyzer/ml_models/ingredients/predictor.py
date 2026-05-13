# Module to predict product safety using the trained ML model
import os
import re
import ast
import pickle
import logging

logger = logging.getLogger("analyzer")

# file paths

_DIR = os.path.dirname(os.path.abspath(__file__))
_MODEL_PATH = os.path.join(_DIR, "skincare_model.pkl")

# caching the model in memory
_bundle = None
# loads the pkl bundle

def _load_model():
    """Load the model bundle once and cache it in-process."""
    global _bundle
    if _bundle is None:
        if not os.path.exists(_MODEL_PATH):
            raise FileNotFoundError(
                f"Model file not found at {_MODEL_PATH}. "
                "Run `python analyzer/ml_models/ingredients/train_model.py` first."
            )
        with open(_MODEL_PATH, "rb") as f:
            _bundle = pickle.load(f)
        logger.info("Skincare safety model loaded and cached.")
    return _bundle


# cleaning text for the model

def _clean_ingredients(raw_text: str) -> str:
    """
    Normalise raw ingredient text into a lowercase space-separated string
    suitable for the TF-IDF vectorizer.
    """
    text = raw_text.lower()
    # Strip non-alphabetic noise (preserving hyphens for compound names)
    text = re.sub(r"[^a-z\s\-]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text
# skin type mapping

_SKIN_INDEX = {
    "combination": 0,
    "dry": 1,
    "normal": 2,
    "oily": 3,
    "sensitive": 4,
}
def predict_safety(ingredients_text: str, skin_type: str) -> dict:
    """
    Predict whether a product is safe for the given skin type.

    Parameters
    ----------
    ingredients_text : str
        Raw ingredient list (comma-separated or paragraph).
    skin_type : str
        One of: combination, dry, normal, oily, sensitive.

    Returns
    -------
    dict with keys:
        prediction   – 1 (safe) or 0 (unsafe)
        probability  – float probability of being safe (0-1)
        skin_type    – the normalised skin type used
        all_results  – dict of {skin_type: {prediction, probability}} for all 5 types
    """
    bundle = _load_model()
    model = bundle["model"]
    vectorizer = bundle["vectorizer"]
    skin_types = bundle["skin_types"]  # ["Combination", "Dry", "Normal", "Oily", "Sensitive"]

    skin_type = skin_type.lower().strip()
    if skin_type not in _SKIN_INDEX:
        raise ValueError(
            f"Unknown skin type '{skin_type}'. "
            f"Must be one of: {list(_SKIN_INDEX.keys())}"
        )

    # normalize ingredients

    clean_text = _clean_ingredients(ingredients_text)
    if not clean_text:
        raise ValueError("Ingredients text is empty after cleaning.")

    # convert text to numbers

    X = vectorizer.transform([clean_text])
    predictions = model.predict(X)[0]  # shape: (5,)

    # calculate safety probabilities

    probabilities = []
    for estimator in model.estimators_:
        prob = estimator.predict_proba(X)[0]
        # prob might be shape (2,) for [unsafe_prob, safe_prob]
        # or shape (1,) if only one class was seen during training
        if len(prob) == 2:
            probabilities.append(float(prob[1]))  # P(safe)
        else:
            probabilities.append(float(predictions[len(probabilities)]))

    # build final result dict

    all_results = {}
    for i, st in enumerate(skin_types):
        all_results[st.lower()] = {
            "prediction": int(predictions[i]),
            "probability": round(probabilities[i], 4),
        }

    idx = _SKIN_INDEX[skin_type]
    return {
        "prediction": int(predictions[idx]),
        "probability": round(probabilities[idx], 4),
        "skin_type": skin_type,
        "all_results": all_results,
    }
