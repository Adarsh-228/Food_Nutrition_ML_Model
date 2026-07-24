import pandas as pd
import numpy as np
import os
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import xgboost as xgb
from catboost import CatBoostClassifier

def main():
    INPUT_PATH = r"a:\ct_project\data\processed\foods_featured.csv"
    MODELS_DIR = r"a:\ct_project\models"
    
    os.makedirs(MODELS_DIR, exist_ok=True)

    print("Loading featured dataset...")
    df = pd.read_csv(INPUT_PATH)

    # Prepare features (X) and target (y)
    # Exclude non-predictive columns and the calculated score
    drop_cols = ['Food Items', 'Nutrition_Score', 'Nutrition_Label']
    X = df.drop(columns=drop_cols)
    y_raw = df['Nutrition_Label']

    # Encode labels to integers (required by XGBoost)
    le = LabelEncoder()
    y = le.fit_transform(y_raw)
    
    # Save the label encoder
    joblib.dump(le, os.path.join(MODELS_DIR, 'label_encoder.joblib'))

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    print(f"Training data shape: {X_train.shape}")
    print(f"Testing data shape: {X_test.shape}")

    # Initialize models
    models = {
        'Decision Tree': DecisionTreeClassifier(random_state=42),
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
        'XGBoost': xgb.XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', random_state=42),
        'CatBoost': CatBoostClassifier(verbose=0, random_state=42)
    }

    # Add Stacking Classifier
    estimators = [
        ('rf', models['Random Forest']),
        ('xgb', models['XGBoost']),
        ('cb', models['CatBoost'])
    ]
    models['Stacking'] = StackingClassifier(estimators=estimators, final_estimator=LogisticRegression())

    results = {}

    print("\nTraining and evaluating models...")
    for name, model in models.items():
        print(f"--- {name} ---")
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        results[name] = acc
        print(f"Accuracy: {acc:.4f}")
        
        # Save model
        model_path = os.path.join(MODELS_DIR, f"{name.replace(' ', '_').lower()}_model.joblib")
        joblib.dump(model, model_path)

    print("\n--- Summary ---")
    for name, acc in sorted(results.items(), key=lambda item: item[1], reverse=True):
        print(f"{name}: {acc:.4f}")
        
    print(f"\nAll models saved to {MODELS_DIR}")

if __name__ == "__main__":
    main()
