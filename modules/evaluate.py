from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

def evaluate_performance(y_true: list, y_pred: list):
    """
    Fonction pour mesurer les performances du modèle avec MSE, MAE et R².
    Entrées :
        - y_true : Sorties attendues (list)
        - y_pred : Sorties prédites par le modèle (list)
    Sortie :
        - MSE : Erreur quadratique moyenne (int)
        - MAE : Erreur absolue moyenne, écart entre les prédictions et les résultats attendus (int)
        - R² : Coefficient de détermination, mesure de la qualité des prédictions du modèle entre 0 et 1 (float)
    """
    mse = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    return {'MSE': mse, 'MAE': mae, 'R²': r2} 