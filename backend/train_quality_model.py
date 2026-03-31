import json
import os
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import joblib

from ppg_features import extract_ppg_features

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
LABELED_FILE = os.path.join(SCRIPT_DIR, "labeled_records.json")
MODEL_FILE = os.path.join(SCRIPT_DIR, "quality_model.joblib")
SCALER_FILE = os.path.join(SCRIPT_DIR, "quality_scaler.joblib")


def load_labeled():
    """Load labeled segments from JSON file"""
    if os.path.exists(LABELED_FILE):
        with open(LABELED_FILE, "r") as f:
            return json.load(f)
    return []


def main():
    # Load data
    records = load_labeled()
    if len(records) < 10:
        print(f"Need at least 10 labeled segments. Found: {len(records)}")
        print("Collect more good/bad samples using the web UI first.")
        return
    
    # Extract features and labels
    X = np.array([extract_ppg_features(r["ppgData"]) for r in records])
    y = np.array([1 if r["label"] == "good" else 0 for r in records])
    
    print(f"Loaded {len(records)} segments")
    print(f"  Good (1): {sum(y)}")
    print(f"  Bad  (0): {len(y) - sum(y)}")
    print(f"Feature dimension: {X.shape[1]}")
    
    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Standardize features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train Random Forest classifier (instead of Logistic Regression)
    model = RandomForestClassifier(
        n_estimators=100,      # Number of trees
        max_depth=10,          # Tree depth (prevent overfitting)
        min_samples_split=5,   # Minimum samples to split a node
        random_state=42
    )
    model.fit(X_train_scaled, y_train)
    
    # Evaluate
    train_acc = model.score(X_train_scaled, y_train)
    test_acc = model.score(X_test_scaled, y_test)
    y_pred = model.predict(X_test_scaled)
    
    print(f"\nTraining accuracy: {train_acc:.3f}")
    print(f"Test accuracy: {test_acc:.3f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=['bad', 'good']))
    
    # Feature importance (Random Forest advantage)
    importances = model.feature_importances_
    feature_names = ['mean', 'std', 'skewness', 'kurtosis', 
                     'range', 'zero_crossings', 'rms', 'peak_to_peak',
                     'spectral_energy', 'dominant_freq', 'peaks', 'valleys']
    print("\nFeature importances:")
    for name, imp in sorted(zip(feature_names, importances), key=lambda x: -x[1]):
        print(f"  {name}: {imp:.3f}")
    
    # Save model and scaler
    joblib.dump(model, MODEL_FILE)
    joblib.dump(scaler, SCALER_FILE)
    print(f"\nSaved model to {MODEL_FILE}")
    print(f"Saved scaler to {SCALER_FILE}")


if __name__ == "__main__":
    main()