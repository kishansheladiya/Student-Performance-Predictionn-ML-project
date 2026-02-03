# Student Performance Prediction

**Goal:** Predict a student's final exam score (G3) as a regression problem and provide explanations for predictions.

Project structure
```
/backend    # FastAPI service
/frontend   # React app
/ml         # data download, training scripts, notebooks
/models     # saved model artifacts and metadata
/data       # raw/processed datasets
/tests      # tests for API and data validation
```

Quickstart (local)

1. Create Python environment and install backend deps
   - cd backend
   - python -m venv .venv
   - . .venv/Scripts/activate
   - pip install -r requirements.txt

2. Train model (optional if you want a fresh model)
   - cd ml
   - python train.py
   - This will download the UCI student dataset, run EDA, train models, and save the best model to `/models`.

3. Run backend
   - cd backend
   - uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

4. Run frontend (development)
   - cd frontend
   - npm install
   - npm start

API Endpoints
- GET /health
- POST /predict
  - Example payload:
    {
      "features": {
        "sex": "F",
        "age": 17,
        "studytime": 2,
        "failures": 0,
        "G1": 10,
        "G2": 11,
        ...
      }
    }
- GET /model-info


Docker
- docker-compose up --build

Run training on GitHub Actions (no local build tools required)
- Push to GitHub and on the repository page go to Actions → Train Model → Run workflow (or trigger via workflow_dispatch)
- After the job completes, download the `models` artifact from the workflow run — place the files into your local `models/` directory.

Tests
- Run tests (ensure you have trained model or let tests trigger training):
  - pytest

Notes on training
- Run `python ml/train.py` to download the UCI dataset, run training, and save artifacts in `/models`.
 - The training script performs EDA checks and trains multiple models using a reproducible sklearn Pipeline + ColumnTransformer.
 - The best model and metadata JSON are saved to `/models`.

Notes
- The ML pipeline uses sklearn Pipeline + ColumnTransformer and trains multiple regressors (Linear, RandomForest, GradientBoosting).
- Explainability: SHAP values for tree-based models and feature importance are included in the API response.

Screenshots
- Add screenshots to `docs/screenshots` after running the app.
