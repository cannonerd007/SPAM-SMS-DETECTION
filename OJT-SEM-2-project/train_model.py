"""
train_model.py
==============
Trains three text classifiers on spam.csv, evaluates them, and saves
the best pipeline (TF-IDF + model) + all evaluation results to outputs/.

Fixed issues:
  - Each pipeline now gets its OWN TfidfVectorizer instance (shared instance
    caused incorrect cross_val_score results because one pipeline's fit would
    overwrite the shared transformer state used by the others).
  - ASCII-safe print statements (no emoji) for Windows cp1252 consoles.

Run once:  python train_model.py
"""

import sys
sys.path.insert(0, "D:/py_libs")   # scikit-learn installed here

import os, json
import numpy  as np
import pandas as pd
import joblib

from sklearn.model_selection         import train_test_split, cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes             import MultinomialNB
from sklearn.linear_model            import LogisticRegression
from sklearn.svm                     import LinearSVC
from sklearn.tree                    import DecisionTreeClassifier
from sklearn.pipeline                import Pipeline
from sklearn.calibration             import CalibratedClassifierCV
from sklearn.metrics                 import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, roc_auc_score, roc_curve,
)

os.makedirs("outputs", exist_ok=True)

# ── helper: fresh TF-IDF instance per pipeline ────────────────
def make_tfidf():
    return TfidfVectorizer(
        max_features  = 6000,
        ngram_range   = (1, 2),
        sublinear_tf  = True,
        strip_accents = "unicode",
        analyzer      = "word",
        min_df        = 2,
    )

# ── 1. Load & clean data ──────────────────────────────────────
print("Loading spam.csv ...")
raw = pd.read_csv("spam.csv", encoding="latin-1", usecols=[0, 1])
raw.columns = ["label", "message"]
raw = raw.dropna(subset=["message"]).copy()
raw["label_num"] = (raw["label"] == "spam").astype(int)

print(f"  Total rows : {len(raw):,}")
print(f"  Spam       : {int(raw['label_num'].sum()):,}")
print(f"  Ham        : {int((raw['label_num'] == 0).sum()):,}")

X = raw["message"]
y = raw["label_num"]

# ── 2. Train / test split (80 / 20, stratified) ───────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)
print(f"\nTrain: {len(X_train):,}  |  Test: {len(X_test):,}")

# ── 3. Define pipelines (each with its OWN tfidf instance) ────
models = {
    "Naive Bayes": Pipeline([
        ("tfidf", make_tfidf()),
        ("clf",   MultinomialNB(alpha=0.1)),
    ]),
    "Logistic Regression": Pipeline([
        ("tfidf", make_tfidf()),
        ("clf",   LogisticRegression(max_iter=1000, C=5, random_state=42)),
    ]),
    "Linear SVM": Pipeline([
        ("tfidf", make_tfidf()),
        ("clf",   CalibratedClassifierCV(
                      LinearSVC(max_iter=2000, C=1.0, random_state=42)
                  )),
    ]),
    "Decision Tree": Pipeline([
        ("tfidf", make_tfidf()),
        ("clf",   DecisionTreeClassifier(
                      max_depth=20,
                      min_samples_split=5,
                      random_state=42
                  )),
    ]),
}

# ── 4. Train & evaluate ───────────────────────────────────────
results = {}

for name, pipe in models.items():
    print(f"\nTraining  {name} ...")
    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)
    y_prob = pipe.predict_proba(X_test)[:, 1]

    acc  = accuracy_score (y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec  = recall_score   (y_test, y_pred)
    f1   = f1_score       (y_test, y_pred)
    auc  = roc_auc_score  (y_test, y_prob)
    cm   = confusion_matrix(y_test, y_pred).tolist()

    fpr, tpr, _ = roc_curve(y_test, y_prob)

    # 5-fold CV F1 on full dataset (uses its own fresh clone internally)
    cv_f1 = cross_val_score(pipe, X, y, cv=5, scoring="f1").mean()

    results[name] = {
        "accuracy"        : round(float(acc),  4),
        "precision"       : round(float(prec), 4),
        "recall"          : round(float(rec),  4),
        "f1"              : round(float(f1),   4),
        "roc_auc"         : round(float(auc),  4),
        "cv_f1"           : round(float(cv_f1),4),
        "confusion_matrix": cm,
        "roc_fpr"         : [round(v, 6) for v in fpr.tolist()],
        "roc_tpr"         : [round(v, 6) for v in tpr.tolist()],
    }

    print(f"  Acc={acc:.4f}  Prec={prec:.4f}  "
          f"Rec={rec:.4f}  F1={f1:.4f}  AUC={auc:.4f}  CV-F1={cv_f1:.4f}")

# ── 5. Pick best model (by F1) and save ───────────────────────
best_name = max(results, key=lambda k: results[k]["f1"])
best_pipe = models[best_name]
print(f"\nBest model: {best_name}  (F1={results[best_name]['f1']:.4f})")

joblib.dump(best_pipe, "outputs/spam_model.pkl")
print("  Saved -> outputs/spam_model.pkl")

# ── 6. Save all results + metadata ────────────────────────────
results["_meta"] = {
    "best_model"   : best_name,
    "train_size"   : int(len(X_train)),
    "test_size"    : int(len(X_test)),
    "total_rows"   : int(len(raw)),
    "spam_count"   : int(raw["label_num"].sum()),
    "ham_count"    : int((raw["label_num"] == 0).sum()),
    "model_names"  : list(models.keys()),
    "tfidf_params" : {
        "max_features": 6000,
        "ngram_range" : "(1, 2)",
        "sublinear_tf": True,
        "min_df"      : 2,
    },
}

with open("outputs/ml_results.json", "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2)
print("  Saved -> outputs/ml_results.json")

print("\nDone!")
