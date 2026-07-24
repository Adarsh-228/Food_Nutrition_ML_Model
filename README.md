# Nutrition Scoring Project

This project implements a complete end-to-end machine learning pipeline for nutrition scoring and recommendation, as outlined in the following phases:

- **Phase 1**: Data Understanding
- **Phase 2**: Nutrition Scoring
- **Phase 3**: Feature Engineering
- **Phase 4**: Model Development
- **Phase 5**: Explainability
- **Phase 6**: Recommendation Engine
- **Phase 7**: Evaluation

## Nutrition Scoring Logic
For **Phase 2**, we engineer a **Nutrition Score** to assign labels (`Healthy`, `Moderate`, `Unhealthy`) since the dataset lacks them.

1. **Beneficial Nutrients**: `Protein(g)`, `Fibre(g)`, `Calcium(mg)`
2. **Detrimental Nutrients**: `Fat(g)`, `Freesugar(g)`, `Cholestrol(mg)`, `Energy kcal`

**Calculation**:
- All nutrients are normalized to a `0 - 1` scale using Min-Max Scaling.
- `Score = (Normalized Protein + Normalized Fibre + Normalized Calcium) - (Normalized Fat + Normalized Freesugar + Normalized Cholestrol + Normalized Energy)`
- The dataset is divided into three equal classes (quantiles) based on this score:
  - Top 33% -> `Healthy`
  - Middle 33% -> `Moderate`
  - Bottom 33% -> `Unhealthy`

## Directory Structure
- `data/raw/`: Raw datasets
- `data/processed/`: Cleaned and preprocessed data
- `notebooks/`: Jupyter notebooks for exploration
- `src/`: Source code for the project
- `models/`: Trained ML models
- `reports/`: Generated analysis and figures
