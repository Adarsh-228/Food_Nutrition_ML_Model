import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
import xgboost as xgb
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

INPUT_PATH = r"a:\ct_project\data\processed\foods_featured.csv"
df = pd.read_csv(INPUT_PATH)
drop_cols = ['Food Items', 'Nutrition_Score', 'Nutrition_Label']
X = df.drop(columns=drop_cols)
y = df['Nutrition_Score']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

models = {
    'Decision Tree': DecisionTreeRegressor(random_state=42),
    'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
    'XGBoost': xgb.XGBRegressor(random_state=42)
}

print("Metrics for predicting Nutrition_Score:\n")
for name, model in models.items():
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    r2 = r2_score(y_test, preds)
    mae = mean_absolute_error(y_test, preds)
    mse = mean_squared_error(y_test, preds)
    rmse = np.sqrt(mse)
    
    print(f"--- {name} ---")
    print(f"R2 Square: {r2:.4f}")
    print(f"MAE:       {mae:.4f}")
    print(f"MSE:       {mse:.4f}")
    print(f"RMSE:      {rmse:.4f}\n")
