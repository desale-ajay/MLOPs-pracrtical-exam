"""Student 5: track candidate runs in MLflow and export the best model."""
from pathlib import Path
import joblib
import mlflow
import mlflow.sklearn
from model_training import train_candidates, ROOT

def main():
    mlflow.set_tracking_uri((ROOT / "mlruns").as_uri())
    mlflow.set_experiment("Wine_Classification")

    results = train_candidates()
    best = max(results, key=lambda item: item["accuracy"])

    for item in results:
        with mlflow.start_run(run_name=item["name"]):
            mlflow.log_param("model_name", item["name"])
            mlflow.log_param("dataset", "Scikit-learn Wine")
            mlflow.log_param("random_state", 42)
            mlflow.log_metric("test_accuracy", float(item["accuracy"]))
            mlflow.sklearn.log_model(item["model"], artifact_path="model")

    # Log the selected model in a separate, clearly named run.
    with mlflow.start_run(run_name=f"BEST_{best['name']}"):
        mlflow.log_param("selected_model", best["name"])
        mlflow.log_metric("best_test_accuracy", float(best["accuracy"]))
        mlflow.sklearn.log_model(best["model"], artifact_path="best_model")

    model_dir = ROOT / "models"
    model_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(best["model"], model_dir / "best_model.joblib")
    (model_dir / "best_model.txt").write_text(
        f"Best model: {best['name']}\nTest accuracy: {best['accuracy']:.4f}\n",
        encoding="utf-8"
    )
    print(f"Selected {best['name']} with accuracy {best['accuracy']:.4f}")
    print(f"Saved model to {model_dir / 'best_model.joblib'}")

if __name__ == "__main__":
    main()
