"""
Training Script for AI SkinCare Safety Analyzer
================================================
Trains a unified multi-output Random Forest model from cleaned_dataset.csv.

The dataset has columns:
  Ingredients, Combination, Dry, Normal, Oily, Sensitive
where each skin-type column is binary (0 = unsafe, 1 = safe).

This script:
  1. Loads and preprocesses the CSV data.
  2. Vectorises ingredient text using TF-IDF.
  3. Trains a multi-output Random Forest classifier.
  4. Saves the model + vectorizer as skincare_model.pkl.

Run once before starting the Django server:
  python analyzer/ml_models/ingredients/train_model.py
"""

import os
import sys
import ast
import re

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import pickle

# ─── Paths ────────────────────────────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(SCRIPT_DIR, "cleaned_dataset.csv")
MODEL_PATH = os.path.join(SCRIPT_DIR, "skincare_model.pkl")

SKIN_TYPES = ["Combination", "Dry", "Normal", "Oily", "Sensitive"]


def flatten_ingredients(raw: str) -> str:
    """
    Convert the raw ingredient cell (a nested list of strings) into a single
    space-separated lowercase string for TF-IDF vectorisation.
    Example:
        "[['water'], ['glycerin'], ...]"  →  "water glycerin ..."
    """
    try:
        parsed = ast.literal_eval(raw)
        tokens = []
        for group in parsed:
            tokens.append(" ".join(group))
        return " ".join(tokens).lower().strip()
    except (ValueError, SyntaxError):
        # Fallback: strip non-alphabetic chars and lowercase
        cleaned = re.sub(r"[^a-zA-Z\s]", " ", str(raw))
        return re.sub(r"\s+", " ", cleaned).lower().strip()


def main():
    print("=" * 60)
    print("  AI SkinCare Safety Analyzer — Model Training")
    print("=" * 60)

    # ── 1. Load Dataset ──────────────────────────────────────────────────
    if not os.path.exists(CSV_PATH):
        print(f"[ERROR] Dataset not found at: {CSV_PATH}")
        sys.exit(1)

    df = pd.read_csv(CSV_PATH)
    print(f"\n✅ Dataset loaded: {df.shape[0]} rows × {df.shape[1]} columns")
    print(f"   Columns: {list(df.columns)}")

    # ── 2. Preprocess Ingredients Text ────────────────────────────────────
    print("\n🔄 Preprocessing ingredient text ...")
    df["clean_text"] = df["Ingredients"].apply(flatten_ingredients)

    # Drop rows with empty text after cleaning
    df = df[df["clean_text"].str.len() > 0].reset_index(drop=True)
    print(f"   Rows after cleaning: {df.shape[0]}")

    # ── 3. TF-IDF Vectorisation ──────────────────────────────────────────
    print("🔄 Vectorising with TF-IDF ...")
    vectorizer = TfidfVectorizer(max_features=3000, ngram_range=(1, 2))
    X = vectorizer.fit_transform(df["clean_text"])
    y = df[SKIN_TYPES].values  # shape: (n_samples, 5)

    print(f"   Feature matrix: {X.shape}")
    print(f"   Label matrix:   {y.shape}")

    # ── 4. Train/Test Split ──────────────────────────────────────────────
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    print(f"\n📊 Split → Train: {X_train.shape[0]}  |  Test: {X_test.shape[0]}")

    # ── 5. Train Multi-Output Random Forest ──────────────────────────────
    print("\n🚀 Training Multi-Output Random Forest ...")
    base_rf = RandomForestClassifier(
        n_estimators=200,
        max_depth=30,
        min_samples_split=3,
        min_samples_leaf=1,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )
    model = MultiOutputClassifier(base_rf)
    model.fit(X_train, y_train)
    print("   Training complete.")

    # ── 6. Evaluation ────────────────────────────────────────────────────
    y_pred = model.predict(X_test)
    print("\n📈 Per-skin-type Classification Report:\n")
    for idx, skin in enumerate(SKIN_TYPES):
        print(f"--- {skin} ---")
        print(classification_report(y_test[:, idx], y_pred[:, idx], zero_division=0))

    # Overall accuracy per output
    accuracies = [(y_test[:, i] == y_pred[:, i]).mean() for i in range(len(SKIN_TYPES))]
    avg_acc = np.mean(accuracies)
    print(f"{'Skin Type':<15} {'Accuracy':>10}")
    print("-" * 27)
    for skin, acc in zip(SKIN_TYPES, accuracies):
        print(f"{skin:<15} {acc*100:>9.1f}%")
    print("-" * 27)
    print(f"{'Average':<15} {avg_acc*100:>9.1f}%")

    # ── 7. Save Model + Vectorizer ────────────────────────────────────────
    bundle = {
        "model": model,
        "vectorizer": vectorizer,
        "skin_types": SKIN_TYPES,
    }
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(bundle, f)
    print(f"\n💾 Model saved to: {MODEL_PATH}")
    print(f"   File size: {os.path.getsize(MODEL_PATH) / 1024:.1f} KB")
    print("\n✅ Done! You can now start your Django server.\n")


if __name__ == "__main__":
    main()
