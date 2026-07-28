import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import plotly.express as px
import plotly.graph_objects as go
from sklearn.metrics.pairwise import euclidean_distances

# Configuration
st.set_page_config(page_title="Nutrition Model Comparison", layout="wide", page_icon="🍎")

# Define Paths
MODELS_DIR = r"a:\ct_project\models"
FEATURED_DATA_PATH = r"a:\ct_project\data\processed\foods_featured.csv"

# Pre-defined Accuracies (from phase4_modeling.py output for display purposes)
# In a real app, this might be loaded from a JSON file saved during training.
MODEL_ACCURACIES = {
    'CatBoost': 0.9990,
    'Decision Tree': 0.9981,
    'Stacking': 0.9981,
    'XGBoost': 0.9981,
    'Random Forest': 0.9971
}

@st.cache_resource
def load_models():
    """Load all saved models and the label encoder."""
    models = {}
    try:
        models['Decision Tree'] = joblib.load(os.path.join(MODELS_DIR, 'decision_tree_model.joblib'))
        models['Random Forest'] = joblib.load(os.path.join(MODELS_DIR, 'random_forest_model.joblib'))
        models['XGBoost'] = joblib.load(os.path.join(MODELS_DIR, 'xgboost_model.joblib'))
        models['CatBoost'] = joblib.load(os.path.join(MODELS_DIR, 'catboost_model.joblib'))
        models['Stacking'] = joblib.load(os.path.join(MODELS_DIR, 'stacking_model.joblib'))
        le = joblib.load(os.path.join(MODELS_DIR, 'label_encoder.joblib'))
        return models, le
    except Exception as e:
        st.error(f"Error loading models: {e}. Please ensure models are trained and saved in {MODELS_DIR}.")
        return None, None

@st.cache_data
def load_data():
    """Load the featured dataset for recommendations."""
    if os.path.exists(FEATURED_DATA_PATH):
        return pd.read_csv(FEATURED_DATA_PATH)
    return None

def process_inputs(inputs_dict):
    """Convert raw inputs into the feature format expected by the models."""
    df = pd.DataFrame([inputs_dict])
    
    epsilon = 1e-9
    df['Protein_to_Energy'] = df['Protein(g)'] / (df['Energy kcal'] + epsilon)
    df['Sugar_to_Carbs'] = df['Freesugar(g)'] / (df['Carbs'] + epsilon)
    df['Fat_to_Energy'] = df['Fat(g)'] / (df['Energy kcal'] + epsilon)
    
    df.replace([np.inf, -np.inf], 0, inplace=True)
    
    # Ensure column order matches the training data
    expected_cols = ['Energy kcal', 'Carbs', 'Protein(g)', 'Fat(g)', 'Freesugar(g)', 
                     'Fibre(g)', 'Cholestrol(mg)', 'Calcium(mg)', 
                     'Protein_to_Energy', 'Sugar_to_Carbs', 'Fat_to_Energy']
    
    return df[expected_cols]

def main():
    st.title("🍎 Nutrition Label Predictor & Model Comparison")
    st.markdown("Predict the health label of a food item using multiple ML models, compare their performance, and discover healthier alternatives!")

    # --- Sidebar for Input ---
    st.sidebar.header("Input Food Nutritional Values (per 100g)")
    
    inputs = {
        'Energy kcal': st.sidebar.number_input("Energy (kcal)", min_value=0.0, value=250.0),
        'Carbs': st.sidebar.number_input("Carbs (g)", min_value=0.0, value=30.0),
        'Protein(g)': st.sidebar.number_input("Protein (g)", min_value=0.0, value=5.0),
        'Fat(g)': st.sidebar.number_input("Fat (g)", min_value=0.0, value=10.0),
        'Freesugar(g)': st.sidebar.number_input("Free Sugar (g)", min_value=0.0, value=15.0),
        'Fibre(g)': st.sidebar.number_input("Fibre (g)", min_value=0.0, value=2.0),
        'Cholestrol(mg)': st.sidebar.number_input("Cholesterol (mg)", min_value=0.0, value=0.0),
        'Calcium(mg)': st.sidebar.number_input("Calcium (mg)", min_value=0.0, value=20.0)
    }

    # Load Resources
    models, label_encoder = load_models()
    dataset = load_data()

    if not models or label_encoder is None:
        return

    # Process Input
    X_input = process_inputs(inputs)

    # --- Predictions ---
    st.header("1. Model Predictions")
    
    predictions = {}
    probabilities = {}
    
    for name, model in models.items():
        pred_idx = model.predict(X_input)[0]
        # XGBoost output might just be integer if not wrapped, let's ensure it's mapped correctly
        if hasattr(label_encoder, 'inverse_transform'):
            pred_label = label_encoder.inverse_transform([pred_idx])[0]
        else:
            pred_label = str(pred_idx)
            
        predictions[name] = pred_label
        
        # Get probabilities if available
        if hasattr(model, 'predict_proba'):
            prob = model.predict_proba(X_input)[0]
            probabilities[name] = prob

    # Display Predictions nicely
    cols = st.columns(len(models))
    for i, (name, label) in enumerate(predictions.items()):
        color = "green" if label == "Healthy" else "orange" if label == "Moderate" else "red"
        cols[i].metric(label=f"{name}", value=label, delta=f"Acc: {MODEL_ACCURACIES.get(name, 0)*100:.1f}%", delta_color="off")
        
    # --- Visualizations ---
    st.header("2. Model Visualizations")
    
    tab1, tab2 = st.tabs(["Prediction Probabilities", "Feature Importances"])
    
    with tab1:
        st.subheader("Confidence Scores across Models")
        if probabilities:
            # Prepare DataFrame for Plotly
            classes = label_encoder.classes_
            prob_data = []
            for name, probs in probabilities.items():
                for cls_name, p in zip(classes, probs):
                    prob_data.append({'Model': name, 'Class': cls_name, 'Probability': p})
            
            prob_df = pd.DataFrame(prob_data)
            fig = px.bar(prob_df, x="Model", y="Probability", color="Class", 
                         barmode="group", title="Prediction Probabilities by Model",
                         color_discrete_map={"Healthy": "green", "Moderate": "orange", "Unhealthy": "red"})
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Probability data is not available for these models.")

    with tab2:
        st.subheader("What drives the model decisions?")
        # Feature importance for tree-based models
        importances = {}
        if hasattr(models['Random Forest'], 'feature_importances_'):
            importances['Random Forest'] = models['Random Forest'].feature_importances_
        if hasattr(models['XGBoost'], 'feature_importances_'):
            importances['XGBoost'] = models['XGBoost'].feature_importances_
            
        if importances:
            imp_df = pd.DataFrame(importances, index=X_input.columns).reset_index()
            imp_df = imp_df.melt(id_vars='index', var_name='Model', value_name='Importance')
            imp_df.rename(columns={'index': 'Feature'}, inplace=True)
            
            fig2 = px.bar(imp_df, x="Importance", y="Feature", color="Model", barmode="group",
                          orientation='h', title="Feature Importance Comparison")
            st.plotly_chart(fig2, use_container_width=True)
        else:
            st.info("Feature importance not supported for the selected models.")

    # --- Recommendations ---
    st.header("3. Top 3 Healthy Recommendations")
    
    if dataset is not None:
        healthy_foods = dataset[dataset['Nutrition_Label'] == 'Healthy'].copy()
        
        if not healthy_foods.empty:
            # We want to find healthy foods that are most similar in terms of Macros (Energy, Protein, Carbs, Fat)
            # This helps recommend a food that serves a similar dietary purpose but is healthier.
            macro_cols = ['Energy kcal', 'Protein(g)', 'Carbs', 'Fat(g)']
            
            target_vector = np.array([[inputs['Energy kcal'], inputs['Protein(g)'], inputs['Carbs'], inputs['Fat(g)']]])
            dataset_vectors = healthy_foods[macro_cols].values
            
            # Calculate Euclidean distance
            distances = euclidean_distances(target_vector, dataset_vectors)[0]
            healthy_foods['Distance'] = distances
            
            # Get top 3 closest
            top_3 = healthy_foods.sort_values(by='Distance').head(3)
            
            st.write("Based on the macronutrient profile you entered, here are 3 healthy alternatives:")
            
            for idx, row in top_3.iterrows():
                with st.expander(f"🌟 {row['Food Items']} (Similarity Score: {1000/(1+row['Distance']):.1f})"):
                    col1, col2, col3, col4 = st.columns(4)
                    col1.metric("Energy", f"{row['Energy kcal']} kcal")
                    col2.metric("Protein", f"{row['Protein(g)']} g")
                    col3.metric("Carbs", f"{row['Carbs']} g")
                    col4.metric("Fat", f"{row['Fat(g)']} g")
        else:
            st.warning("No healthy foods found in the dataset.")
    else:
        st.warning("Dataset not found. Cannot provide recommendations.")

if __name__ == "__main__":
    main()
