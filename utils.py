import numpy as np
import os 
import joblib
def get_prediction_confidence(model, X):
    try:
        probabilities = model.predict_proba(X)
        return round(float(np.max(probabilities)),2)
    except:
        return 0
    
vectorizer = None
metrics = None
models = {}
# ---------- Model Loading ----------
def load_models():
    global vectorizer, metrics, models
    model_dir = "models"
    required_files = ["vectorizer.pkl", "metrics.pkl", "logistic.pkl", "naive_bayes.pkl", "svm.pkl"]

    for file in required_files:
        file_path = os.path.join(model_dir, file)
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Missing file: {file_path}")

    vectorizer = joblib.load(os.path.join(model_dir, "vectorizer.pkl"))
    metrics = joblib.load(os.path.join(model_dir, "metrics.pkl"))

    for name in ["logistic", "naive_bayes", "svm"]:
        models[name] = joblib.load(os.path.join(model_dir, f"{name}.pkl"))

    return models, vectorizer, metrics