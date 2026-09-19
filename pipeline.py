from pathlib import Path
import shutil
from data_ingestion import DataIngestion
from preprocessing import CreditScorePreprocessor
from train import CreditScoreTrainer
from evaluation import ModelEvaluator

class CreditScorePipeline:
    def __init__(self, raw_data_path, accuracy_threshold=0.72):
        self.base_dir = Path(__file__).parent
        self.raw_data_path = Path(raw_data_path)
        self.ingested_dir = (self.base_dir / "ingested")
        self.accuracy_threshold = accuracy_threshold

        self.ingestor = DataIngestion(self.raw_data_path, self.ingested_dir)
        self.preprocessor = CreditScorePreprocessor()
        self.trainer = CreditScoreTrainer()
        self.evaluator = ModelEvaluator()

    def execute(self):
        print("Credit Score Training Pipeline")

        ingested_file = self.ingestor.run()

        df = self.preprocessor.clean_data(ingested_file)

        x_train, x_test, y_train, y_test = self.preprocessor.split_data(df)

        transformer = self.preprocessor.get_transformer(x_train)

        rf_run_id = self.trainer.train_random_forest(x_train, y_train, transformer)

        xgb_run_id = self.trainer.train_xgboost(x_train, y_train, transformer)

        run_ids = {"Random Forest": rf_run_id, "XGBoost": xgb_run_id}

        results = self.evaluator.run(run_ids, x_test, y_test)

        print("\nFINAL RESULTS")
        for result in results:
            print(f"\nModel : {result['Model']}")
            print(f"Accuracy     : {result['Accuracy']:.4f}")
            print(f"Weighted F1  : {result['Weighted F1']:.4f}")
            print(f"Test AUC     : {result['Test AUC']:.4f}")

        best_model = max(results, key=lambda x: x["Weighted F1"])
        print("\nBEST MODEL")
        print(f"Model : {best_model['Model']}")
        print(f"Weighted F1 : {best_model['Weighted F1']:.4f}")
        
        if best_model["Model"] == "Random Forest":
            best_model_path = (self.trainer.artifact_dir / "random_forest_pipeline.pkl")
        else:
            best_model_path = (self.trainer.artifact_dir / "xgboost_pipeline.pkl")

        deployment_model_path = (self.trainer.artifact_dir / "best_model.pkl")

        shutil.copy(best_model_path, deployment_model_path)

        print(f"\nBest model artifact saved to {deployment_model_path}")

        if best_model["Accuracy"] >= self.accuracy_threshold:
            print("\nDeployment Status : APPROVED")
        else:
            print("\nDeployment Status : REJECTED")

if __name__ == "__main__":
    DATA_INPUT = Path(__file__).parent / "credit_score.csv"

    pipeline = CreditScorePipeline(raw_data_path=DATA_INPUT, accuracy_threshold=0.72)
    pipeline.execute()