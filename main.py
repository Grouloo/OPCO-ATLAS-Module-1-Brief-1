from modules.preprocess import preprocessing, split
from modules.evaluate import evaluate_performance
from modules.print_draw import print_data, draw_loss
from models.models import train_model, model_predict
import pandas as pd
import joblib
from os.path import join as join
import mlflow

mlflow.set_experiment("MLflow Quickstart")

mlflow.sklearn.autolog()


# Chargement des datasets
df_old = pd.read_csv(join('data','df_old.csv'))

# Charger le préprocesseur
preprocessor_loaded = joblib.load(join('models','preprocessor.pkl'))

# preprocesser les data
X, y, _ = preprocessing(df_old)

# split data in train and test dataset
X_train, X_test, y_train, y_test = split(X, y)

# charger le modèle
model_2024_08 = joblib.load(join('models','model_2024_08.pkl'))

#%% predire sur les valeurs de train
y_pred = model_predict(model_2024_08, X_train)

# mesurer les performances MSE, MAE et R²
perf = evaluate_performance(y_train, y_pred)  

print_data(perf, exp_name="Évaluation 1, données d'entraînement")

#%% predire sur les valeurs de tests
y_pred = model_predict(model_2024_08, X_test)

# mesurer les performances MSE, MAE et R²
perf = evaluate_performance(y_test, y_pred)   

print_data(perf, exp_name="Évaluation 2, données de test")

#%% WARNING ZONE on test d'entrainer le modèle plus longtemps mais sur les mêmes données
model2, hist2 = train_model(model_2024_08, X_train, y_train, X_val=X_test, y_val=y_test)
y_pred = model_predict(model_2024_08, X_test)
perf = evaluate_performance(y_test, y_pred)  
print_data(perf, exp_name="Évaluation 3, modèle ré-entraîné, données de test")
fig = draw_loss(hist2)
fig.show()




