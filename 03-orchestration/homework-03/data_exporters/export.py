import pickle
import mlflow


mlflow.set_tracking_uri('http://mlflow:5000')
# mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("nyc-taxi-experiment")

if 'data_exporter' not in globals():
    from mage_ai.data_preparation.decorators import data_exporter


@data_exporter
def export_data(data, *args, **kwargs):
    art_path = 'mlops/homework-03/models/lin_reg.bin'
    with open(art_path, 'wb') as f_out:
        pickle.dump((data), f_out)

    mlflow.end_run()
    with mlflow.start_run():

        mlflow.set_tag("developer", "kate")
        alpha = 0.1
        mlflow.log_param("alpha", alpha)
    
        mlflow.log_artifact(local_path=art_path, artifact_path="models_pickle")

        mlflow.sklearn.log_model(data[1], "logreg")