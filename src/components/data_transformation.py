from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

class DataTransformation:
    def __init__(self):
        self.preprocessor = None

    def create_preprocessor(self, num_features, cat_features):
        numeric_transformer = Pipeline([
            ("scaler", StandardScaler())
        ])
        categorical_transformer = Pipeline([
            ("encoder", OneHotEncoder(handle_unknown="ignore"))
        ])

        self.preprocessor = ColumnTransformer(
            transformers=[
                ("num", numeric_transformer, num_features),
                ("cat", categorical_transformer, cat_features)
            ]
        )
        return self.preprocessor
