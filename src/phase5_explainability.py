import pandas as pd
import shap
import joblib
import os
import matplotlib.pyplot as plt

def main():
    INPUT_PATH = r"a:\ct_project\data\processed\foods_featured.csv"
    MODEL_PATH = r"a:\ct_project\models\random_forest_model.joblib"
    REPORTS_DIR = r"a:\ct_project\reports"
    
    os.makedirs(REPORTS_DIR, exist_ok=True)
    
    print("Loading model and dataset...")
    df = pd.read_csv(INPUT_PATH)
    model = joblib.load(MODEL_PATH)
    
    # Prepare features
    drop_cols = ['Food Items', 'Nutrition_Score', 'Nutrition_Label']
    X = df.drop(columns=drop_cols)
    
    print("Generating SHAP values...")
    # Initialize explainer for tree models
    explainer = shap.TreeExplainer(model)
    
    # Calculate SHAP values for a sample to avoid very long runtimes
    shap_values = explainer.shap_values(X)
    
    print("Generating Summary Plot...")
    plt.figure()
    
    # Check if shap_values is an Explanation object or an array with 3 dimensions
    if hasattr(shap_values, 'shape') and len(shap_values.shape) == 3:
        # shap_values.shape is (n_samples, n_features, n_classes)
        # We can pass the whole array for a multi-class summary plot
        shap.summary_plot(shap_values, X, show=False)
        shap_values_class0 = shap_values[:, :, 0]
    elif isinstance(shap_values, list):
        shap.summary_plot(shap_values, X, show=False)
        shap_values_class0 = shap_values[0]
    else:
        shap.summary_plot(shap_values, X, show=False)
        shap_values_class0 = shap_values
        
    summary_plot_path = os.path.join(REPORTS_DIR, 'shap_summary_plot.png')
    plt.savefig(summary_plot_path, bbox_inches='tight')
    plt.close()
    
    print(f"SHAP Summary plot saved to {summary_plot_path}")
    
    # Generate a single dependence plot for the most important feature (e.g., Energy kcal or Protein)
    print("Generating Dependence Plot for Energy kcal...")
    plt.figure()
    
    shap.dependence_plot("Energy kcal", shap_values_class0, X, show=False)
        
    dependence_plot_path = os.path.join(REPORTS_DIR, 'shap_dependence_plot.png')
    plt.savefig(dependence_plot_path, bbox_inches='tight')
    plt.close()
    
    print(f"SHAP Dependence plot saved to {dependence_plot_path}")
    print("\nPhase 5 Complete.")

if __name__ == "__main__":
    main()
