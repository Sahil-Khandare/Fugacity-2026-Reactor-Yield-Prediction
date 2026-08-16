# Fugacity 2026 — Reactor Yield Prediction

Machine learning solution for predicting **overall reactor yield** from chemical process operating conditions using ensemble tree-based regression and physics-inspired feature engineering.

## Overview

This project was developed for **Fugacity 2026**, a machine learning competition focused on predicting the overall yield of a chemical reactor.

The task is to predict `overall_yield` from five provided process variables:

- Flow rate
- Reactant concentration
- Inlet temperature
- Reactor length
- Jacket temperature

The dataset contains a limited number of labeled samples, making **robust validation, feature engineering, reproducibility, and model generalization** important considerations.

The final solution uses an **Extra Trees Regressor** combined with physically motivated feature engineering to capture nonlinear relationships and interactions between reactor operating conditions.

---

## Features

- Exploratory Data Analysis (EDA)
- Missing-value and dataset inspection
- Correlation analysis
- Feature distribution analysis
- Feature-vs-target visualization
- Multiple regression model comparison
- Physics-inspired feature engineering
- Extra Trees ensemble regression
- 5-fold cross-validation
- Hyperparameter experimentation
- Holdout validation
- Reproducible random seeds
- Final model retraining on all available labeled data
- Automated prediction generation
- Competition-ready CSV submission

---

## Architecture / Workflow

```text
                 Competition Dataset
                         │
                         ▼
                Data Exploration
                         │
                         ▼
              Data Preprocessing
                         │
                         ▼
              Train / Validation Split
                   120 / 30 samples
                         │
                         ▼
              Feature Engineering
                         │
              ┌──────────┴──────────┐
              │                     │
              ▼                     ▼
    Temperature Difference    Flow / Length Ratio
              │                     │
              └──────────┬──────────┘
                         ▼
                  Model Training
                         │
              ┌──────────┼──────────┐
              ▼          ▼          ▼
        Linear Reg.  Random Forest  Extra Trees
              │          │          │
              └──────────┴──────────┘
                         ▼
                 Model Evaluation
                         │
                         ▼
                Extra Trees Selected
                         │
                         ▼
              Cross-Validation & Tuning
                         │
                         ▼
              Final Model on 150 Samples
                         │
                         ▼
                Test Set Prediction
                         │
                         ▼
                  submission.csv
```

---

## Tech Stack

### Programming Language

- Python

### Machine Learning

- Scikit-learn

### Data Processing

- Pandas
- NumPy

### Visualization

- Matplotlib
- Seaborn

### Development Environment

- Google Colab

---

## Project Structure

```text
Fugacity-2026/
│
├── data/
│   ├── train_dataset.csv
│   └── test_dataset.csv
│
├── notebooks/
│   └── Fugacity_2026.ipynb
│
├── submission/
│   └── Pekka.csv
│
├── README.md
└── requirements.txt
```
---

## Installation

Clone the repository:

```bash
git clone https://github.com/Sahil-Khandare/Fugacity-2026-Reactor-Yield-Prediction.git
cd Fugacity-2026-Reactor-Yield-Prediction
```

Install the required dependencies:

```bash
pip install -r requirements.txt
```

### Required Libraries

```text
numpy
pandas
matplotlib
seaborn
scikit-learn
```

---

## Usage

This project was developed and executed using **Google Colab**.

Open the Jupyter Notebook in Google Colab and run the cells sequentially from:

```text
Data Loading
      ↓
EDA
      ↓
Train/Validation Split
      ↓
Feature Engineering
      ↓
Model Training
      ↓
Cross-Validation
      ↓
Model Selection
      ↓
Final Training
      ↓
Prediction
      ↓
Submission Generation
```

The final prediction file is generated as a CSV containing:

```text
overall_yield
```

with exactly 50 predictions in the same order as the competition test dataset.

---

## Dataset & Methodology

The competition dataset provides five process variables for predicting `overall_yield`.

### Input Features

| Feature | Description |
|---|---|
| `flow_rate_L_min` | Reactor flow rate |
| `concentration_mol_L` | Reactant concentration |
| `inlet_temperature_K` | Reactor inlet temperature |
| `length_m` | Reactor length |
| `jacket_temperature_K` | Jacket temperature |

### Target

```text
overall_yield
```

The target represents the overall reactor yield to be predicted.

### Dataset Split

The labeled dataset contained:

```text
150 labeled samples
50 test samples
```

During model development:

```text
120 samples → Training
30 samples  → Holdout Validation
```

The validation set was kept separate from model training and hyperparameter selection.

After model selection, the final model was retrained using all 150 labeled samples before generating predictions for the 50-row test dataset.

---

## Feature Engineering

Two physically motivated features were introduced using only the variables supplied in the competition dataset.

### 1. Temperature Difference

```python
temperature_difference_K = (
    inlet_temperature_K - jacket_temperature_K
)
```

This represents the temperature difference between the reactor inlet and jacket conditions.

### 2. Flow-to-Length Ratio

```python
flow_length_ratio = (
    flow_rate_L_min / length_m
)
```

This combines flow rate and reactor length into a derived process variable.

No external datasets, proprietary data, or pretrained weights were used.

### Final Feature Set

```text
flow_rate_L_min
concentration_mol_L
inlet_temperature_K
length_m
jacket_temperature_K
temperature_difference_K
flow_length_ratio
```

---

## Model & Evaluation

Several regression approaches were evaluated:

- Linear Regression
- Random Forest Regressor
- Extra Trees Regressor
- Gradient Boosting Regressor
- HistGradientBoosting Regressor

### Model Selection

Extra Trees consistently performed better than the other evaluated models on the validation data.

The final model used:

```python
ExtraTreesRegressor(
    n_estimators=300,
    max_features=1.0,
    random_state=42
)
```

### Validation Results

| Model | Holdout RMSE |
|---|---:|
| Linear Regression | 31.42 |
| Random Forest | 21.49 |
| Extra Trees | 19.59 |
| Gradient Boosting | 24.86 |
| HistGradientBoosting | 22.82 |
| Extra Trees + Engineered Features | **17.98** |

### Cross-Validation

The final feature-engineered Extra Trees approach achieved:

```text
5-Fold CV RMSE: 19.14
CV Standard Deviation: 3.21
```

The holdout validation RMSE was:

```text
17.98
```

Lower RMSE indicates better predictive performance.

> These are local validation results and should not be interpreted as the official competition leaderboard score.

---

## Screenshots

### Exploratory Data Analysis

Add screenshots of:

- Feature distributions
- Correlation heatmap
- Feature vs. yield scatter plots

Example:

```markdown
![Correlation Heatmap](screenshots/correlation_heatmap.png)
```

### Model Evaluation

Add the actual-vs-predicted visualization:

```markdown
![Actual vs Predicted](screenshots/actual_vs_predicted.png)
```

### Feature Importance

```markdown
![Feature Importance](screenshots/feature_importance.png)
```

---

## Technical Implementation

### Reproducibility

A fixed random seed was used throughout the modeling workflow:

```python
RANDOM_STATE = 42
```

This ensures reproducible:

- Train/validation splits
- Cross-validation folds
- Randomized model behavior
- Ensemble model construction

### Cross-Validation

5-fold cross-validation was used during model evaluation and hyperparameter experimentation.

```python
KFold(
    n_splits=5,
    shuffle=True,
    random_state=RANDOM_STATE
)
```

### Evaluation Metric

The competition uses **Root Mean Squared Error (RMSE)**:

\[
RMSE =
\sqrt{
rac{1}{n}
\sum_{i=1}^{n}
(y_i-\hat{y_i})^2
}
\]

Lower RMSE indicates better predictive performance.

---

## Challenges & Solutions

### 1. Nonlinear Process Behavior

The relationship between reactor operating conditions and yield was not well represented by a linear model.

**Solution:**  
Tree-based ensemble models were evaluated to capture nonlinear relationships and feature interactions.

---

### 2. Limited Dataset Size

Only a relatively small number of labeled samples were available.

**Solution:**  
5-fold cross-validation was used during model development to obtain a more robust estimate of model performance.

---

### 3. Feature Interactions

Individual process variables did not fully capture the relationships present in the reactor system.

**Solution:**  
Physically motivated derived features were introduced, including:

```text
temperature_difference_K
flow_length_ratio
```

---

### 4. Model Complexity and Overfitting

Increasing model complexity does not necessarily improve generalization on a small dataset.

**Solution:**  
Models were compared using both cross-validation and a held-out validation set rather than selecting a model purely based on training performance.

---

### 5. Data Leakage Prevention

During development, the validation set was kept separate from model tuning.

The final model was only retrained on all labeled samples **after model selection was completed**.

---

## Results

The final selected approach was:

```text
Extra Trees Regressor
+
Physically Motivated Feature Engineering
```

### Final Local Performance

```text
Holdout RMSE: 17.98
5-Fold CV RMSE: 19.14
```

Feature engineering improved the holdout result from approximately:

```text
19.59 → 17.98 RMSE
```

and improved the mean cross-validation result from approximately:

```text
20.59 → 19.14 RMSE
```

The final model was trained on all 150 labeled samples before generating predictions for the 50-row competition test dataset.

---

## Future Improvements

Potential improvements include:

- More extensive physics-informed feature engineering
- More robust feature selection
- Out-of-fold ensemble blending
- Model stacking
- Bayesian hyperparameter optimization
- Uncertainty estimation
- Error analysis by operating regime
- Additional validation strategies for small datasets
- Model interpretability using permutation importance or SHAP
- Improved documentation of the underlying chemical-process intuition

---

## Testing

The project includes validation through:

- Holdout validation
- 5-fold cross-validation
- Prediction sanity checks
- Submission-format validation

Final prediction checks included:

```text
50 predictions
0 missing values
0 predictions below 0
0 predictions above 100
1 output column
```

---

## Author

**Sahil Khandare**

Software Engineering Student  
India
