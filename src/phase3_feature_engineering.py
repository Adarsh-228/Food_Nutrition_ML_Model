import pandas as pd
import numpy as np
import os

def main():
    INPUT_PATH = r"a:\ct_project\data\processed\foods_scored.csv"
    OUTPUT_PATH = r"a:\ct_project\data\processed\foods_featured.csv"

    print("Loading scored dataset...")
    df = pd.read_csv(INPUT_PATH)

    print("Engineering features...")
    
    # Avoid division by zero by replacing 0 with a small epsilon or handling conditionally
    # For Energy and Carbs, they might be 0 for some items, so we use np.where or add epsilon
    epsilon = 1e-9

    # 1. Protein-to-Energy Ratio
    df['Protein_to_Energy'] = df['Protein(g)'] / (df['Energy kcal'] + epsilon)

    # 2. Sugar-to-Carbs Ratio
    df['Sugar_to_Carbs'] = df['Freesugar(g)'] / (df['Carbs'] + epsilon)
    
    # 3. Fat-to-Energy Ratio
    df['Fat_to_Energy'] = df['Fat(g)'] / (df['Energy kcal'] + epsilon)

    # Clean up potentially infinite values if there were any weird cases
    df.replace([np.inf, -np.inf], 0, inplace=True)
    
    # Check new columns
    print("New features summary:")
    print(df[['Protein_to_Energy', 'Sugar_to_Carbs', 'Fat_to_Energy']].describe())

    # Save
    df.to_csv(OUTPUT_PATH, index=False)
    print(f"\nFeatured dataset saved to: {OUTPUT_PATH}")

if __name__ == "__main__":
    main()
