import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, StackingClassifier
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, fbeta_score, balanced_accuracy_score, confusion_matrix
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import xgboost as xgb
from catboost import CatBoostClassifier

def calc_specificity(y_true, y_pred, labels):
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    specificities = []
    for i in range(len(labels)):
        tn = np.sum(cm) - np.sum(cm[i,:]) - np.sum(cm[:,i]) + cm[i,i]
        fp = np.sum(cm[:,i]) - cm[i,i]
        if (tn + fp) == 0:
            specificities.append(0.0)
        else:
            specificities.append(tn / (tn + fp))
    return np.mean(specificities)

def main():
    INPUT_PATH = r"a:\ct_project\data\processed\foods_featured.csv"
    REPORTS_DIR = r"a:\ct_project\reports"
    os.makedirs(REPORTS_DIR, exist_ok=True)
    report_path = os.path.join(REPORTS_DIR, "model_evaluation_report.txt")

    df = pd.read_csv(INPUT_PATH)

    with open(report_path, "w", encoding="utf-8") as f:
        f.write("=== MODEL EVALUATION REPORT ===\n")
        
        # --- CLASSIFICATION METRICS (Nutrition_Label) ---
        f.write("\n=========================================\n")
        f.write("PART 1: CLASSIFICATION (Predicting Nutrition_Label)\n")
        f.write("=========================================\n\n")
        
        drop_cols_clf = ['Food Items', 'Nutrition_Score', 'Nutrition_Label']
        X_clf = df.drop(columns=drop_cols_clf)
        y_raw = df['Nutrition_Label']
        
        le = LabelEncoder()
        y_clf = le.fit_transform(y_raw)
        
        X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(X_clf, y_clf, test_size=0.2, random_state=42, stratify=y_clf)
        
        clf_models = {
            'Decision Tree': DecisionTreeClassifier(random_state=42),
            'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
            'XGBoost': xgb.XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', random_state=42),
            'CatBoost': CatBoostClassifier(verbose=0, random_state=42)
        }
        
        # Stacking Classifier
        estimators = [
            ('rf', clf_models['Random Forest']),
            ('xgb', clf_models['XGBoost']),
            ('cb', clf_models['CatBoost'])
        ]
        clf_models['Stacking'] = StackingClassifier(estimators=estimators, final_estimator=LogisticRegression())
        
        unique_labels = np.unique(y_clf)
        
        for name, model in clf_models.items():
            model.fit(X_train_c, y_train_c)
            preds_c = model.predict(X_test_c)
            
            acc = accuracy_score(y_test_c, preds_c)
            prec = precision_score(y_test_c, preds_c, average='weighted', zero_division=0)
            rec = recall_score(y_test_c, preds_c, average='weighted', zero_division=0)
            f1 = f1_score(y_test_c, preds_c, average='weighted', zero_division=0)
            fbeta = fbeta_score(y_test_c, preds_c, beta=0.5, average='weighted', zero_division=0) # Beta=0.5
            bal_acc = balanced_accuracy_score(y_test_c, preds_c)
            error_rate = 1.0 - acc
            specificity = calc_specificity(y_test_c, preds_c, unique_labels)
            
            f.write(f"--- {name} ---\n")
            f.write(f"Accuracy:          {acc:.4f}\n")
            f.write(f"Precision:         {prec:.4f}\n")
            f.write(f"Recall (TPR):      {rec:.4f}\n")
            f.write(f"Specificity (TNR): {specificity:.4f} (Macro Avg)\n")
            f.write(f"F1 Score:          {f1:.4f}\n")
            f.write(f"Fβ Score (β=0.5):  {fbeta:.4f}\n")
            f.write(f"Balanced Accuracy: {bal_acc:.4f}\n")
            f.write(f"Error Rate:        {error_rate:.4f}\n\n")

        # --- REGRESSION METRICS (Nutrition_Score) ---
        f.write("\n=========================================\n")
        f.write("PART 2: REGRESSION (Predicting Nutrition_Score)\n")
        f.write("=========================================\n\n")
        
        y_reg = df['Nutrition_Score']
        X_train_r, X_test_r, y_train_r, y_test_r = train_test_split(X_clf, y_reg, test_size=0.2, random_state=42)
        
        reg_models = {
            'Decision Tree Regressor': DecisionTreeRegressor(random_state=42),
            'Random Forest Regressor': RandomForestRegressor(n_estimators=100, random_state=42),
            'XGBoost Regressor': xgb.XGBRegressor(random_state=42)
        }
        
        n_samples = len(y_test_r)
        p_features = X_train_r.shape[1]

        for name, model in reg_models.items():
            model.fit(X_train_r, y_train_r)
            preds_r = model.predict(X_test_r)
            
            r2 = r2_score(y_test_r, preds_r)
            # Adjusted R2 calculation
            adj_r2 = 1 - (1 - r2) * (n_samples - 1) / (n_samples - p_features - 1)
            
            mae = mean_absolute_error(y_test_r, preds_r)
            mse = mean_squared_error(y_test_r, preds_r)
            rmse = np.sqrt(mse)
            
            f.write(f"--- {name} ---\n")
            f.write(f"R2 Square:       {r2:.4f}\n")
            f.write(f"Adj R2 Square:   {adj_r2:.4f}\n")
            f.write(f"MAE:             {mae:.4f}\n")
            f.write(f"MSE:             {mse:.4f}\n")
            f.write(f"RMSE:            {rmse:.4f}\n\n")
            
    print(f"Report generated successfully at: {report_path}")

if __name__ == '__main__':
    main()
