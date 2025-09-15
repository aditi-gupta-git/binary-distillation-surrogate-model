import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

class ModelTrainer:
    def __init__(self, models: dict):
        self.models = models
        self.results = {}

    def train_and_evaluate(self, X_train, y_train, X_test, y_test):
        for name, model in self.models.items():
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            mae = mean_absolute_error(y_test, y_pred)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            r2 = r2_score(y_test, y_pred)

            self.results[name] = {"MAE": mae, "RMSE": rmse, "R2": r2}

        return self.results
