import pandas as pd

class PredictPipeline:
    def __init__(self, model, preprocessor):
        self.model = model
        self.preprocessor = preprocessor

    def predict(self, input_dict):
        df = pd.DataFrame([input_dict])
        X = self.preprocessor.transform(df)
        return self.model.predict(X)