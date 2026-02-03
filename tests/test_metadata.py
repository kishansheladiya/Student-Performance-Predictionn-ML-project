import os
import json

def test_metadata_exists():
    models_dir = os.path.join(os.path.dirname(__file__), '..', 'models')
    files = [f for f in os.listdir(models_dir) if f.endswith('.json')]
    assert len(files) > 0, "No metadata json files found in models directory. Run ml/train.py"
    meta_path = os.path.join(models_dir, files[0])
    with open(meta_path, 'r') as fh:
        meta = json.load(fh)
    assert 'features' in meta
    assert 'metrics' in meta
