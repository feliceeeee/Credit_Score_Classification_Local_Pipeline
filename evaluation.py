import mlflow
import mlflow.sklearn
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

class ModelEvaluator:

    def run(self, run_ids, x_test, y_test):
        results = []

        for model_name, run_id in run_ids.items():
            model_uri = f"runs:/{run_id}/model"
            model = mlflow.sklearn.load_model(model_uri)

            y_pred = model.predict(x_test)
            y_proba = model.predict_proba(x_test)
            
            accuracy = accuracy_score(y_test, y_pred)
            weighted_f1 = f1_score(y_test, y_pred, average="weighted")
            test_auc = roc_auc_score(y_test, y_proba, multi_class="ovr", average="macro")

            with mlflow.start_run(run_id=run_id):
                mlflow.log_metric("accuracy", accuracy)
                mlflow.log_metric("weighted_f1", weighted_f1)
                mlflow.log_metric("test_auc", test_auc)

            results.append({"Model": model_name, "Accuracy": accuracy, "Weighted F1": weighted_f1, "Test AUC": test_auc})

        return results