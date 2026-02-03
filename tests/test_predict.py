import json
import os
import subprocess
import time

import pytest
from fastapi.testclient import TestClient

# Import the FastAPI app
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'backend'))
from app.main import app
from app.model_loader import model_wrapper

client = TestClient(app)


def ensure_model():
    # If no model exists, run training (may take some time)
    models_dir = os.path.join(os.path.dirname(__file__), '..', 'models')
    pkls = [f for f in os.listdir(models_dir) if f.endswith('.pkl')]
    if not pkls:
        # run training
        subprocess.check_call([sys.executable, os.path.join(os.path.dirname(__file__), '..', 'ml', 'train.py')])


def test_health():
    res = client.get('/health')
    assert res.status_code == 200
    assert res.json()['status'] == 'ok'


def test_predict_flow():
    ensure_model()
    # reload model
    model_wrapper.load()
    features = model_wrapper.meta.get('features')
    assert features is not None

    # build a minimal payload using median/typical values
    payload = {f: 0 for f in features}
    # set some realistic values
    if 'age' in payload:
        payload['age'] = 17
    if 'G1' in payload:
        payload['G1'] = 10
    if 'G2' in payload:
        payload['G2'] = 11

    res = client.post('/predict', json={'features': payload})
    assert res.status_code == 200, res.text
    body = res.json()
    assert 'prediction' in body
    assert 'top_features' in body
    assert isinstance(body['top_features'], list)
