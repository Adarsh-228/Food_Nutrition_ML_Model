import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import os

def load_data(filepath):
    """Loads the dataset."""
    return pd.read_csv(filepath)

def recommend_alternatives(food_name, df, top_n=3):
    """
    Recommends healthy alternatives for a given food item.
    """
    # Features to use for similarity
    features = [
        'Protein(g)', 'Fibre(g)', 'Calcium(mg)', 
        'Energy kcal', 'Fat(g)', 'Freesugar(g)', 'Cholestrol(mg)'
    ]
    
    # Ensure food exists
    target = df[df['Food Items'].str.lower() == food_name.lower()]
    if target.empty:
        return f"Food item '{food_name}' not found in the dataset."
        
    target_idx = target.index[0]
    target_features = target[features].values
    target_label = target['Nutrition_Label'].values[0]
    
    print(f"\nTarget: {target['Food Items'].values[0]} | Current Label: {target_label}")
    
    if target_label == 'Healthy':
        return "This food is already classified as Healthy! No alternatives needed."
        
    # Filter for 'Healthy' items only
    healthy_df = df[df['Nutrition_Label'] == 'Healthy']
    
    if healthy_df.empty:
        return "No healthy alternatives available in the dataset."
        
    healthy_features = healthy_df[features].values
    
    # Calculate Cosine Similarity
    similarities = cosine_similarity(target_features, healthy_features).flatten()
    
    # Get top N indices
    top_indices = similarities.argsort()[-top_n:][::-1]
    
    recommendations = healthy_df.iloc[top_indices].copy()
    recommendations['Similarity_Score'] = similarities[top_indices]
    
    print("\n--- Top Recommended Alternatives ---")
    for idx, row in recommendations.iterrows():
        print(f"-> {row['Food Items']} (Similarity: {row['Similarity_Score']:.2f})")
        
    return recommendations[['Food Items', 'Similarity_Score'] + features]

def main():
    INPUT_PATH = r"a:\ct_project\data\processed\foods_featured.csv"
    
    print("Loading dataset for recommendation engine...")
    df = load_data(INPUT_PATH)
    
    # Test cases
    test_foods = [
        "Chocolate cake", # Likely unhealthy
        "French fries"    # Likely unhealthy
    ]
    
    # If the exact names don't match, we will just pick an Unhealthy item from the dataset dynamically
    unhealthy_sample = df[df['Nutrition_Label'] == 'Unhealthy'].iloc[0]['Food Items']
    test_foods.append(unhealthy_sample)
    
    for food in test_foods:
        try:
            print(f"\n>>> Requesting recommendation for: {food}")
            recommend_alternatives(food, df)
        except Exception as e:
            print(f"Error processing {food}: {e}")

if __name__ == "__main__":
    main()
