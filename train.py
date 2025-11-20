from modules.preprocess import preprocessing, split
from modules.print_draw import  draw_loss
from models.models import create_nn_model, train_model, model_predict
from modules.evaluate import evaluate_performance
import pandas as pd
import joblib
from os.path import join as join
import mlflow
import mlflow.data
from datetime import datetime

mlflow.set_experiment("Entraînement du " + datetime.now().strftime("%d/%m/%Y, %H:%M:%S"))

# mlflow.sklearn.autolog()
old_model = joblib.load(join('models','model_2024_08.pkl'))

# Chargement des datasets
df_old = pd.read_csv(join('data','df_old.csv'))
df_new = pd.read_csv(join('data','df_new.csv'))

new_dataset = mlflow.data.from_pandas(
    df_new, source='data/df_new.csv', name="Nouveau dataset", targets="montant_pret"
)


 # Charger le préprocesseur
preprocessor_loaded = joblib.load(join('models','preprocessor.pkl'))

# preprocesser les data
old_X, old_y, _ = preprocessing(df_old)
new_X, new_y, _ = preprocessing(df_new)

# split data in train and test dataset
_, old_X_test, _, old_y_test = split(old_X, old_y)
new_X_train, new_X_test, new_y_train, new_y_test = split(new_X, new_y)

with mlflow.start_run(run_name = "Évaluation de l'ancien modèle"):
    epochs = 100
    mlflow.log_param("epochs", epochs)
    mlflow.log_input(new_dataset)

    old_y_pred = model_predict(old_model, old_X_test)
    old_metrics = evaluate_performance(old_y_test, old_y_pred)   
    mlflow.log_metric("MSE_old", old_metrics["MSE"])
    mlflow.log_metric("MAE_old", old_metrics["MAE"])
    mlflow.log_metric("R²_old", old_metrics["R²"])

    new_y_pred = model_predict(old_model, new_X_test)
    new_metrics = evaluate_performance(new_y_test, new_y_pred)   
    mlflow.log_metric("MSE_new", new_metrics["MSE"])
    mlflow.log_metric("MAE_new", new_metrics["MAE"])
    mlflow.log_metric("R²_new", new_metrics["R²"])
    mlflow.sklearn.log_model(old_model, "Ancien  modèle")

with mlflow.start_run(run_name = "Ré-entraînement de l'ancien modèle"):
    epochs = 100
    mlflow.log_param("epochs", epochs)
    mlflow.log_input(new_dataset)
    retrained_old_model, hist, metrics = train_model(old_model, new_X_train, new_y_train, X_val=new_X_test, y_val=new_y_test, epochs=epochs)
    plot = draw_loss(hist)

    old_y_pred = model_predict(retrained_old_model, old_X_test)
    old_metrics = evaluate_performance(old_y_test, old_y_pred)   
    mlflow.log_metric("MSE_old", old_metrics["MSE"])
    mlflow.log_metric("MAE_old", old_metrics["MAE"])
    mlflow.log_metric("R²_old", old_metrics["R²"])

    mlflow.log_metric("MSE_new", metrics["MSE"])
    mlflow.log_metric("MAE_new", metrics["MAE"])
    mlflow.log_metric("R²_new", metrics["R²"])
    mlflow.log_figure(plot, "loss.png")
    mlflow.sklearn.log_model(old_model, "Ancien modèle ré-entraîné")

with mlflow.start_run(run_name = "Entraînement du nouveau modèle"):
    epochs = 100
    mlflow.log_param("epochs", epochs)
    mlflow.log_input(new_dataset)
    new_model = create_nn_model(new_X_train.shape[1])
    new_model, hist, metrics = train_model(new_model, new_X_train, new_y_train, X_val=new_X_test, y_val=new_y_test, epochs=epochs)
    plot = draw_loss(hist)
    
    old_y_pred = model_predict(new_model, old_X_test)
    old_metrics = evaluate_performance(old_y_test, old_y_pred)   
    mlflow.log_metric("MSE_old", old_metrics["MSE"])
    mlflow.log_metric("MAE_old", old_metrics["MAE"])
    mlflow.log_metric("R²_old", old_metrics["R²"])
    
    mlflow.log_metric("MSE_new", metrics["MSE"])
    mlflow.log_metric("MAE_new", metrics["MAE"])
    mlflow.log_metric("R²_new", metrics["R²"])
    mlflow.log_figure(plot, "loss.png")
    mlflow.sklearn.log_model(new_model, "Nouveau modèle")
    # sauvegarder le nouveau modèle
    joblib.dump(new_model, join('models','model_2025_11.pkl'))
