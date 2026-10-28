"""Распознавание рукописных цифр: сравнение моделей, подбор гиперпараметров, отчёт."""
import json
import sys
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score, classification_report
from sklearn.model_selection import GridSearchCV, cross_val_score, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

SEED = 42
OUT = Path(__file__).parent / "artifacts"

MODELS = {
    "logreg": make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)),
    "knn": make_pipeline(StandardScaler(), KNeighborsClassifier()),
    "random_forest": RandomForestClassifier(n_estimators=300, random_state=SEED),
    "svm": make_pipeline(StandardScaler(), SVC()),
}
SVM_GRID = {"svc__C": [1, 5, 10, 50], "svc__gamma": ["scale", 0.01, 0.001]}


def main():
    OUT.mkdir(exist_ok=True)
    X, y = load_digits(return_X_y=True)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, stratify=y, random_state=SEED)
    print(f"Train: {len(X_tr)} | Test: {len(X_te)}")

    # 1. Сравнение базовых моделей кросс-валидацией на обучающей выборке
    cv = {}
    for name, model in MODELS.items():
        scores = cross_val_score(model, X_tr, y_tr, cv=5)
        cv[name] = round(scores.mean(), 4)
        print(f"{name:14s} CV accuracy = {scores.mean():.4f} ± {scores.std():.4f}")

    # 2. Подбор гиперпараметров для SVM
    search = GridSearchCV(MODELS["svm"], SVM_GRID, cv=5).fit(X_tr, y_tr)
    print(f"Best SVM params: {search.best_params_}, CV = {search.best_score_:.4f}")

    # 3. Итоговая оценка на отложенной выборке
    best = search.best_estimator_
    pred = best.predict(X_te)
    acc = accuracy_score(y_te, pred)
    print(f"Test accuracy: {acc:.4f}")
    print(classification_report(y_te, pred))

    ConfusionMatrixDisplay.from_predictions(y_te, pred, cmap="Blues")
    plt.title(f"SVM, test accuracy = {acc:.3f}")
    plt.savefig(OUT / "confusion_matrix.png", dpi=150, bbox_inches="tight")
    joblib.dump(best, OUT / "model.joblib")
    (OUT / "metrics.json").write_text(json.dumps(
        {"cv_accuracy": cv, "best_params": search.best_params_,
         "best_cv": round(search.best_score_, 4), "test_accuracy": round(acc, 4)}, indent=2))
    return acc


if __name__ == "__main__":
    # самопроверка: если качество просело ниже 97% — что-то сломано
    sys.exit(0 if main() >= 0.97 else 1)
