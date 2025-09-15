from sklearn.model_selection import train_test_split
from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor

def run_training(file_path: str):
    ingestion = DataIngestion(file_path)
    df = ingestion.load_data()

    X = df[["RefluxRatio", "BoilupRatio", "Feed_MoleFraction", "FeedFlowrate"]]
    y = df[["Distillate_MoleFraction", "Reboiler_Duty"]]  

    num_features = ["RefluxRatio", "BoilupRatio", "Feed_MoleFraction", "FeedFlowrate","Stages"]
    cat_features = ["Feed_ThermalCondition"]

    transformer = DataTransformation()
    preprocessor = transformer.create_preprocessor(num_features, cat_features)

    X_processed = preprocessor.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X_processed, y, test_size=0.2, random_state=42)

    models = {
        "Linear Regression": LinearRegression(),
        "Polynomial Regression (deg 2)": Pipeline([
            ("poly", PolynomialFeatures(degree=2)),
            ("lr", LinearRegression())
        ]),
        "Random Forest": RandomForestRegressor(n_estimators=200, random_state=42),
        "XGBoost": XGBRegressor(n_estimators=200, learning_rate=0.1, random_state=42)
    }

    trainer = ModelTrainer(models)
    results = trainer.train_and_evaluate(X_train, y_train, X_test, y_test)

    print("\nModel Results")
    for model, metrics in results.items():
        print(f"{model}: {metrics}")

    return results, preprocessor
