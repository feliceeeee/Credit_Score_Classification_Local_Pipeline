# Credit Score Classification — Local Pipeline

An end-to-end machine learning project that classifies a customer's credit score into Poor, Standard, or Good categories based on demographic, financial, and credit behavior data. Beyond the initial notebook-based analysis, the project is refactored into an object-oriented, MLflow-tracked local training pipeline and deployed as an interactive Streamlit web app.

## Highlights

- Exploratory data analysis of customer demographic, financial, and credit behavior data
- Data cleaning: handling invalid categorical values, extreme placeholder values, and inconsistent numeric formats stored as text
- Feature engineering: conversion of credit history age into total months, validation of loan count using loan type information, and log transformation of skewed income
- Statistical association analysis using the Kruskal–Wallis test (numerical features) and Cramér's V (categorical features) against the target variable
- Handling of class imbalance using `class_weight` on tree-based models
- Baseline modeling and hyperparameter tuning (GridSearchCV) across Decision Tree, Random Forest, Gradient Boosting, and XGBoost
- Refactored, OOP-based training pipeline with separate classes for data ingestion, preprocessing, training, and evaluation
- Experiment tracking, metric logging, and model versioning using MLflow
- Automated best-model selection and deployment-readiness check based on an accuracy threshold
- Model serving via an interactive Streamlit web app

## Data

The dataset contains 25,000 records and 29 columns, including:
- Demographic information (age, occupation)
- Financial information (annual income, monthly in-hand salary, outstanding debt, monthly balance)
- Credit account details (number of bank accounts, credit cards, and loans)
- Credit behavior (payment delays, delayed payments, credit inquiries, credit mix, payment behavior)
- Credit history length
- `Credit_Score` as the target variable (Good, Standard, or Poor)

## Model Experiments

**Notebook stage (exploration):** Decision Tree, Random Forest, Gradient Boosting, and XGBoost were compared as baselines, with `class_weight='balanced'` used to address class imbalance. Random Forest, XGBoost, and Gradient Boosting were then tuned via `GridSearchCV`.

**Pipeline stage (production refactor):** The winning configurations from the notebook were carried into an OOP pipeline, with each stage encapsulated in its own class:

- `DataIngestion` — validates and saves the raw dataset
- `CreditScorePreprocessor` — cleans data, engineers features, splits train/test, and builds the `ColumnTransformer` (median imputation for numeric features; ordinal encoding for `Credit_Mix` and `Payment_of_Min_Amount`; one-hot encoding for `Payment_Behaviour`)
- `CreditScoreTrainer` — trains a tuned Random Forest (`n_estimators=500`, `min_samples_split=5`) and a tuned XGBoost (`learning_rate=0.1`, `max_depth=6`, `n_estimators=300`) inside sklearn pipelines, logging parameters and models to MLflow
- `ModelEvaluator` — loads each trained model from its MLflow run, computes accuracy, weighted F1, and macro test AUC (OvR), and logs metrics back to MLflow
- `CreditScorePipeline` — orchestrates ingestion → preprocessing → training → evaluation, selects the best model by weighted F1, saves it as `best_model.pkl`, and flags deployment as APPROVED or REJECTED against an accuracy threshold (0.72)

## Results

**Random Forest** was selected as the final model, achieving the highest accuracy, weighted F1, and test AUC after tuning. Classification errors remained concentrated between the Poor/Good classes and the majority Standard class, consistent with the moderate class overlap seen in the EDA. Since its accuracy (73.1%) exceeds the 0.72 deployment threshold, the pipeline marks it as **APPROVED for deployment**, and it is saved as `artifacts/best_model.pkl` for the Streamlit app. XGBoost was retained as a competitive secondary model for comparison.

## Project Structure

- `credit_score.ipynb` — EDA, preprocessing, and initial model experimentation
- `data_ingestion.py` — `DataIngestion` class: validates and saves the raw CSV to `ingested/`
- `preprocessing.py` — `CreditScorePreprocessor` class: cleaning, feature engineering, splitting, and transformer construction
- `train.py` — `CreditScoreTrainer` class: trains Random Forest and XGBoost pipelines with MLflow logging
- `evaluation.py` — `ModelEvaluator` class: loads MLflow-logged models and computes evaluation metrics
- `pipeline.py` — `CreditScorePipeline` class: orchestrates the full ingestion → preprocessing → training → evaluation → best-model-selection workflow
- `app.py` — Streamlit app for interactive credit score prediction using `artifacts/best_model.pkl`
- `requirements.txt` — dependencies for the Streamlit deployment
- `artifacts/` — saved model files (`random_forest_pipeline.pkl`, `xgboost_pipeline.pkl`, `best_model.pkl`)

## How to Run

### Notebook (EDA and experimentation)

1. Clone this repository:

```
git clone https://github.com/your-username/credit-score-classification-local-pipeline.git
```

2. Install the required libraries:

```
pip install pandas numpy matplotlib seaborn scipy scikit-learn xgboost jupyter
```

3. Ensure the dataset is located at: `credit_score.csv`
4. Open and run `credit_score.ipynb`

### Full Pipeline (ingestion, training, evaluation, MLflow)

1. Install additional dependencies:

```
pip install mlflow joblib
```

2. Run the full pipeline:

```
python pipeline.py
```

This ingests the raw data, trains both the Random Forest and XGBoost pipelines, logs parameters/metrics/models to MLflow, selects the best model by weighted F1, and saves it to `artifacts/best_model.pkl`.

3. (Optional) View experiment tracking:

```
mlflow ui
```

### Deployment (Streamlit)

```
pip install -r requirements.txt
streamlit run app.py
```

The app loads `artifacts/best_model.pkl` and lets you input a customer's profile to get a predicted credit score (Poor / Standard / Good) along with class probabilities.
