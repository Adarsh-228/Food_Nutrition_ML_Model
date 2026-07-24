import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys

def main():
    # Set paths
    DATA_PATH = r"a:\ct_project\data\raw\foods.csv"
    REPORT_DIR = r"a:\ct_project\reports"

    os.makedirs(REPORT_DIR, exist_ok=True)

    print("Loading data...")
    try:
        df = pd.read_csv(DATA_PATH)
    except Exception as e:
        print(f"Error loading CSV: {e}")
        sys.exit(1)

    print("\n--- DATASET SHAPE ---")
    print(f"Rows: {df.shape[0]}, Columns: {df.shape[1]}")

    print("\n--- MISSING VALUES ---")
    print(df.isnull().sum())

    print("\n--- DUPLICATE ROWS ---")
    print(df.duplicated().sum())

    print("\n--- SUMMARY STATISTICS ---")
    print(df.describe().to_string())

    # Save plots for distributions and outliers
    features = df.select_dtypes(include=['float64', 'int64']).columns
    
    # Distributions
    plt.figure(figsize=(15, 12))
    for i, feature in enumerate(features, 1):
        plt.subplot(3, 3, i)
        sns.histplot(df[feature], kde=True)
        plt.title(f'Distribution of {feature}')
    plt.tight_layout()
    plt.savefig(os.path.join(REPORT_DIR, 'distributions.png'))
    plt.close()

    # Outliers (Boxplots)
    plt.figure(figsize=(15, 12))
    for i, feature in enumerate(features, 1):
        plt.subplot(3, 3, i)
        sns.boxplot(y=df[feature])
        plt.title(f'Boxplot of {feature}')
    plt.tight_layout()
    plt.savefig(os.path.join(REPORT_DIR, 'outliers.png'))
    plt.close()

    print("\nEDA complete. Visualizations saved to 'reports/distributions.png' and 'reports/outliers.png'")

if __name__ == "__main__":
    main()
