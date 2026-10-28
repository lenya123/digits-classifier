"""Распознать цифру на картинке: python predict.py image.png"""
import sys
from pathlib import Path

import joblib
import numpy as np
from PIL import Image

MODEL = Path(__file__).parent / "artifacts" / "model.joblib"


def to_features(path):
    # Датасет digits — 8x8, яркость 0..16, цифра светлая на тёмном фоне
    img = Image.open(path).convert("L").resize((8, 8), Image.LANCZOS)
    arr = np.asarray(img, dtype=float)
    if arr.mean() > 127:  # тёмная цифра на светлом фоне — инвертируем
        arr = 255 - arr
    return (arr / 255 * 16).reshape(1, -1)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit("usage: python predict.py image.png")
    # модель создаётся локально через train.py, чужой файл сюда не подкладывается
    print(joblib.load(MODEL).predict(to_features(sys.argv[1]))[0])
