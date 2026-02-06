"""
Maisy's SenticNet-based Hourglass Emotion Activation Matrix

This matrix maps the 28 GoEmotions labels to the 4 Hourglass of Emotions dimensions:
- Introspection (I): Joy ↔ Sadness axis
- Temper (T): Serenity ↔ Anger axis  
- Attitude (A): Pleasantness ↔ Disgust axis
- Sensitivity (S): Eagerness ↔ Fear axis

Values derived from SenticNet knowledge base (https://sentic.net/)
Source: data/Hourglass Emotion Activation Matrix.xlsx by Maisy Song

Format: emotion_label (lowercase) -> [I, T, A, S]
"""

from pathlib import Path

import pandas as pd

# Path to Excel: project_root/data/Hourglass Emotion Activation Matrix.xlsx
_EXCEL_PATH = Path(__file__).resolve().parent.parent.parent / "data" / "Hourglass Emotion Activation Matrix.xlsx"


def _load_hg_matrix_from_excel() -> tuple[dict, dict]:
    """Load HG_MATRIX and POLARITY_INTENSITY from Excel. Returns (HG_MATRIX, POLARITY_INTENSITY)."""
    df = pd.read_excel(_EXCEL_PATH)
    matrix = {}
    polarity = {}
    for _, row in df.iterrows():
        emotion = str(row["Emotion"]).strip().lower()
        matrix[emotion] = [
            float(row["INTROSPECTION"]),
            float(row["TEMPER"]),
            float(row["ATTITUDE"]),
            float(row["SENSITIVITY"]),
        ]
        pi = row.get("POLARITY INTENSITY", row.get("POLARITY_INTENSITY"))
        if pd.notna(pi):
            polarity[emotion] = float(pi)
    return matrix, polarity


# Load from Excel; fallback to hardcoded if file missing
try:
    HG_MATRIX, POLARITY_INTENSITY = _load_hg_matrix_from_excel()
    SENTICNET_HG_MATRIX = HG_MATRIX  # alias for backward compatibility
except Exception:
    # Fallback if Excel unavailable (e.g. pandas not installed)
    HG_MATRIX = {
        "admiration": [0.000, 0.000, 0.999, 0.000],
        "amusement": [0.659, 0.000, 0.659, 0.000],
        "anger": [0.000, -0.660, 0.000, 0.000],
        "annoyance": [0.000, -0.330, 0.000, 0.000],
        "approval": [0.000, 0.000, 0.329, 0.000],
        "caring": [0.000, 0.000, 0.659, 0.659],
        "confusion": [0.000, 0.000, 0.000, -0.329],
        "curiosity": [0.000, 0.000, 0.659, 0.000],
        "desire": [0.000, 0.000, 0.659, 0.000],
        "disappointment": [0.000, 0.000, -0.659, 0.000],
        "disapproval": [0.000, 0.000, -0.329, 0.000],
        "disgust": [0.000, 0.000, -0.660, 0.000],
        "embarrassment": [0.000, 0.000, -0.329, 0.000],
        "excitement": [0.659, 0.000, 0.000, 0.659],
        "fear": [0.000, 0.000, 0.000, -0.660],
        "gratitude": [0.000, 0.000, 0.329, 0.000],
        "grief": [-1.000, 0.000, 0.000, 0.000],
        "joy": [0.660, 0.000, 0.000, 0.000],
        "love": [0.659, 0.000, 0.659, 0.000],
        "nervousness": [0.000, 0.000, 0.000, -0.329],
        "optimism": [0.659, 0.000, 0.000, 0.000],
        "pride": [0.000, 0.000, 0.659, 0.000],
        "realization": [0.000, 0.000, 0.872, 0.000],
        "relief": [0.000, 0.999, 0.000, 0.000],
        "remorse": [-0.659, 0.000, -0.659, 0.000],
        "sadness": [-0.660, 0.000, 0.000, 0.000],
        "surprise": [0.000, 0.000, 0.329, 0.000],
        "neutral": [-0.600, 0.000, 0.000, -0.443],
    }
    SENTICNET_HG_MATRIX = HG_MATRIX
    # Fallback polarity intensity
    POLARITY_INTENSITY = {
    "admiration":      0.999,
    "amusement":       0.659,
    "anger":          -0.660,
    "annoyance":      -0.330,
    "approval":        0.329,
    "caring":          0.659,
    "confusion":      -0.329,
    "curiosity":       0.659,
    "desire":          0.659,
    "disappointment": -0.659,
    "disapproval":    -0.329,
    "disgust":        -0.660,
    "embarrassment":  -0.329,
    "excitement":      0.659,
    "fear":           -0.660,
    "gratitude":       0.329,
    "grief":          -1.000,
    "joy":             0.660,
    "love":            0.659,
    "nervousness":    -0.329,
    "optimism":        0.659,
    "pride":           0.659,
    "realization":     0.872,
    "relief":          0.999,
    "remorse":        -0.659,
    "sadness":        -0.660,
    "surprise":        0.329,
    "neutral":        -0.521,
    }

# Polarity intensity values (absolute magnitude of emotional activation)
# Can be used for optional weighting of emotion contributions
# When loaded from Excel, POLARITY_INTENSITY is set above; otherwise use fallback above


def get_clinical_vector(text, pipe) -> list:
    """Extract 4D emotion vector from text using BERT-GoEmotions pipeline + HG_MATRIX."""
    import numpy as np
    if not text or len(str(text).strip()) == 0:
        return [0.0, 0.0, 0.0, 0.0]
    output = pipe(str(text)[:512])[0]
    vector = np.zeros(4)
    for item in output:
        label = item.get("label", "")
        if isinstance(label, str) and label.lower() in HG_MATRIX:
            vector += np.array(HG_MATRIX[label.lower()]) * item["score"]
    return vector.tolist()


# Metadata for documentation
MATRIX_METADATA = {
    "source": "SenticNet knowledge base",
    "author": "Maisy Song",
    "dimensions": ["Introspection", "Temper", "Attitude", "Sensitivity"],
    "num_emotions": 28,
    "value_range": [-1.0, 1.0],
}
