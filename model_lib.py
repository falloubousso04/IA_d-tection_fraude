"""
Module partagé — doit être importable à la fois par le script d'entraînement
et par l'API, car joblib a besoin de retrouver la définition de la classe
au chargement (pickle).
"""
import numpy as np


class FraudModel:
    """Wrapper unique : preprocessing (ColumnTransformer) + XGBoost + décodage des classes."""
    def __init__(self, pipeline_pre, model, classes, feature_num, feature_cat, scenario_name):
        self.pre = pipeline_pre
        self.model = model
        self.classes = list(classes)
        self.feature_num = feature_num
        self.feature_cat = feature_cat
        self.scenario_name = scenario_name

    def predict_proba(self, X_df):
        Xt = self.pre.transform(X_df[self.feature_num + self.feature_cat])
        return self.model.predict_proba(Xt)

    def predict(self, X_df):
        proba = self.predict_proba(X_df)
        idx = np.argmax(proba, axis=1)
        return np.array(self.classes)[idx], proba
