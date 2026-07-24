import pandas as pd
import numpy as np
import os

def calc_energy_points(kcal):
    kj = kcal * 4.184
    bins = [-np.inf, 335, 670, 1005, 1340, 1675, 2010, 2345, 2680, 3015, 3350, np.inf]
    labels = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    return pd.cut(kj, bins=bins, labels=labels, right=True).astype(int)

def calc_sugar_points(sugar):
    bins = [-np.inf, 4.5, 9.0, 13.5, 18.0, 22.5, 27.0, 31.5, 36.0, 40.5, 45.0, np.inf]
    labels = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    return pd.cut(sugar, bins=bins, labels=labels, right=True).astype(int)

def calc_fat_points(fat):
    # Proxy for Sat Fat: 1 point per 3g total fat
    bins = [-np.inf, 3, 6, 9, 12, 15, 18, 21, 24, 27, 30, np.inf]
    labels = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    return pd.cut(fat, bins=bins, labels=labels, right=True).astype(int)

def calc_fiber_points(fiber):
    bins = [-np.inf, 0.9, 1.9, 2.8, 3.7, 4.7, np.inf]
    labels = [0, 1, 2, 3, 4, 5]
    return pd.cut(fiber, bins=bins, labels=labels, right=True).astype(int)

def calc_protein_points(protein):
    bins = [-np.inf, 1.6, 3.2, 4.8, 6.4, 8.0, np.inf]
    labels = [0, 1, 2, 3, 4, 5]
    return pd.cut(protein, bins=bins, labels=labels, right=True).astype(int)

def main():
    RAW_DATA_PATH = r"a:\ct_project\data\raw\foods.csv"
    PROCESSED_DATA_PATH = r"a:\ct_project\data\processed\foods_scored.csv"

    os.makedirs(os.path.dirname(PROCESSED_DATA_PATH), exist_ok=True)

    print("Loading raw dataset...")
    df = pd.read_csv(RAW_DATA_PATH)
    
    # Fill NA values just in case
    df = df.fillna(0)

    # 1. Calculate A Points (Bad)
    df['Points_Energy'] = calc_energy_points(df['Energy kcal'])
    df['Points_Sugar'] = calc_sugar_points(df['Freesugar(g)'])
    df['Points_Fat'] = calc_fat_points(df['Fat(g)'])
    
    df['A_Points'] = df['Points_Energy'] + df['Points_Sugar'] + df['Points_Fat']

    # 2. Calculate C Points (Good)
    df['Points_Fiber'] = calc_fiber_points(df['Fibre(g)'])
    df['Points_Protein'] = calc_protein_points(df['Protein(g)'])
    
    df['C_Points'] = df['Points_Fiber'] + df['Points_Protein']

    # 3. Calculate Final Score
    # Simplified logic: since we don't have fruit/veg/nuts to override protein limits, 
    # we just use A - C if A < 11, else A - Fiber
    # However, for simplicity and to match NutriScore more closely, we often use A - C. 
    # Let's apply: If A >= 11, Score = A - Fiber. Else A - C.
    
    def calculate_final_score(row):
        if row['A_Points'] < 11:
            return row['A_Points'] - row['C_Points']
        else:
            return row['A_Points'] - row['Points_Fiber']

    df['Nutrition_Score'] = df.apply(calculate_final_score, axis=1)

    # 4. Assign Labels based on standard Nutri-Score mapping
    # A (-15 to -1), B (0 to 2) -> Healthy
    # C (3 to 10) -> Moderate
    # D (11 to 18), E (19 to 40) -> Unhealthy
    
    def assign_label(score):
        if score <= 2:
            return 'Healthy'
        elif score <= 10:
            return 'Moderate'
        else:
            return 'Unhealthy'
            
    df['Nutrition_Label'] = df['Nutrition_Score'].apply(assign_label)

    # Clean up intermediate point columns
    drop_cols = ['Points_Energy', 'Points_Sugar', 'Points_Fat', 'A_Points', 
                 'Points_Fiber', 'Points_Protein', 'C_Points']
    df = df.drop(columns=drop_cols)

    # 5. Save the processed dataset
    df.to_csv(PROCESSED_DATA_PATH, index=False)
    print(f"Scored dataset saved to: {PROCESSED_DATA_PATH}")

    # 6. Validate class distribution
    print("\n--- CLASS DISTRIBUTION ---")
    print(df['Nutrition_Label'].value_counts(normalize=True) * 100)
    print(df['Nutrition_Label'].value_counts())

if __name__ == "__main__":
    main()
