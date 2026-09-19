from pathlib import Path
import joblib
import mlflow
import mlflow.sklearn
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
    
class CreditScoreTrainer:

    def __init__(self, experiment_name="Credit Score Prediction", artifact_path="artifacts", random_state=42):
        self.experiment_name = experiment_name
        self.artifact_dir = Path(artifact_path)
        self.random_state = random_state

        self.artifact_dir.mkdir(parents=True, exist_ok=True)
        mlflow.set_experiment(self.experiment_name)

    def train_random_forest(self, x_train, y_train, transformer):
        rf_pipeline = Pipeline([('preprocessing', transformer),
                                ('model', RandomForestClassifier(n_estimators=500, max_depth=None, min_samples_split=5, class_weight='balanced', random_state=self.random_state))])

        with mlflow.start_run(run_name="Random Forest") as run:
            mlflow.log_param("model", "Random Forest")
            mlflow.log_param("n_estimators", 500)
            mlflow.log_param("max_depth", None)
            mlflow.log_param("min_samples_split", 5)

            rf_pipeline.fit(x_train, y_train)

            model_path = (self.artifact_dir / "random_forest_pipeline.pkl")
            joblib.dump(rf_pipeline, model_path)
            mlflow.sklearn.log_model(rf_pipeline, "model")

            print(f"Random Forest saved to {model_path}")
            return run.info.run_id
        
    def train_xgboost(self, x_train, y_train, transformer):
        xgb_pipeline = Pipeline([('preprocessing', transformer),
                                ('model', XGBClassifier(objective='multi:softprob', eval_metric='mlogloss', learning_rate=0.1, max_depth=6, n_estimators=300, random_state=self.random_state))])

        with mlflow.start_run(run_name="XGBoost") as run:
            mlflow.log_param("model", "XGBoost")
            mlflow.log_param("learning_rate", 0.1)
            mlflow.log_param("max_depth", 6)
            mlflow.log_param("n_estimators", 300)

            xgb_pipeline.fit(x_train, y_train)

            model_path = (self.artifact_dir / "xgboost_pipeline.pkl")
            joblib.dump(xgb_pipeline, model_path)
            mlflow.sklearn.log_model(xgb_pipeline, "model")

            print(f"XGBoost saved to {model_path}")
            return run.info.run_id