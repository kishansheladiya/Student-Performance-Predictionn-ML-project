import json
import os
from typing import Any, Dict, Optional

import joblib

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
MODEL_DIR = os.path.join(ROOT, "models")


def latest_model_artifact():
    # pick the latest model file in /models by modification time
    files = [f for f in os.listdir(MODEL_DIR) if f.endswith('.pkl')]
    if not files:
        return None
    paths = [os.path.join(MODEL_DIR, f) for f in files]
    latest = max(paths, key=os.path.getmtime)
    meta_json = os.path.splitext(latest)[0] + '.json'
    meta_path = os.path.join(MODEL_DIR, os.path.basename(meta_json))
    if os.path.exists(meta_path):
        with open(meta_path, 'r') as fh:
            meta = json.load(fh)
    else:
        meta = {}
    return latest, meta


class ModelWrapper:
    def __init__(self):
        self.model = None
        self.meta: Dict[str, Any] = {}

    def load(self, path: Optional[str] = None):
        if path is None:
            res = latest_model_artifact()
            if res is None:
                raise FileNotFoundError("No model artifact found in /models. Run ml/train.py first.")
            path, meta = res
            self.meta = meta
        self.model = joblib.load(path)
        return self.model


model_wrapper = ModelWrapper()
