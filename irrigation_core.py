"""Data loading, model training, and field recommendation logic for the dashboard.

Kept free of Streamlit so it can be unit tested and reused.
"""
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, log_loss
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier

DATA_PATH = Path(__file__).with_name("irrigation_prediction.csv")
TARGET = "Irrigation_Need"
CLASS_ORDER = ["Low", "Medium", "High"]
# Location metadata is excluded from training so it cannot leak into predictions.
MAP_COLUMNS = ["Field_ID", "Latitude", "Longitude", "Coordinate_Source"]
TEST_SIZE = 0.2
RANDOM_STATE = 42
FOREST_TREES = 100

PRIORITY_SCORES = {"High": 100, "Medium": 60, "Low": 20}
BASE_WATER_MM = {"High": 35, "Medium": 20, "Low": 0}
RECOMMENDED_ACTIONS = {
    "High": "Irrigate immediately",
    "Medium": "Schedule irrigation",
    "Low": "Monitor only",
}
TARGET_MOISTURE = 45.0
MAX_WATER_MM = 45.0


@dataclass
class TrainedModel:
    # Single interpretable tree, used for the diagrams and depth analysis.
    tree: DecisionTreeClassifier
    # Ensemble of entropy trees, used for predictions. A lone unpruned tree ends in
    # pure leaves and only ever reports 0% / 100%; averaging bootstrapped trees gives
    # graded, better-calibrated probabilities near decision boundaries.
    forest: RandomForestClassifier
    encoders: dict
    feature_columns: list
    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_test: pd.Series

    def category_options(self, column):
        return [str(value) for value in self.encoders[column].classes_]

    def encode_input(self, values):
        """Turn a dict of raw field values into a one-row feature frame."""
        missing = set(self.feature_columns) - set(values)
        if missing:
            raise ValueError(f"Missing input values: {sorted(missing)}")
        row = {}
        for column in self.feature_columns:
            value = values[column]
            if column in self.encoders:
                value = self.encoders[column].transform([str(value)])[0]
            row[column] = value
        return pd.DataFrame([row], columns=self.feature_columns)

    def predict(self, values):
        """Return (predicted class, {class: probability})."""
        features = self.encode_input(values)
        prediction = self.forest.predict(features)[0]
        probabilities = dict(zip(self.forest.classes_, self.forest.predict_proba(features)[0]))
        return prediction, {label: float(probabilities.get(label, 0.0)) for label in CLASS_ORDER}

    def evaluation(self, estimator=None):
        """Test-set metrics for the prediction forest, or for another fitted estimator."""
        estimator = self.forest if estimator is None else estimator
        y_pred = estimator.predict(self.X_test)
        return {
            "accuracy": accuracy_score(self.y_test, y_pred),
            "log_loss": log_loss(self.y_test, estimator.predict_proba(self.X_test), labels=estimator.classes_),
            "confusion_matrix": confusion_matrix(self.y_test, y_pred, labels=CLASS_ORDER),
            "report": pd.DataFrame(
                classification_report(
                    self.y_test, y_pred, labels=CLASS_ORDER, output_dict=True, zero_division=0
                )
            ).T,
        }

    def feature_importance(self):
        return (
            pd.DataFrame({"Feature": self.feature_columns, "Importance": self.tree.feature_importances_})
            .sort_values("Importance", ascending=False)
            .reset_index(drop=True)
        )

    def depth_curve(self, depths=range(1, 13)):
        """Test accuracy for entropy trees capped at each depth."""
        rows = []
        for depth in depths:
            tree = DecisionTreeClassifier(criterion="entropy", max_depth=depth, random_state=RANDOM_STATE)
            tree.fit(self.X_train, self.y_train)
            rows.append({"Max Depth": depth, "Accuracy": accuracy_score(self.y_test, tree.predict(self.X_test))})
        return pd.DataFrame(rows)


def load_data(path=DATA_PATH):
    return pd.read_csv(path)


def train_model(df):
    """Train an ID3-style (entropy) decision tree and an entropy random forest."""
    features = df.drop(columns=[TARGET] + [c for c in MAP_COLUMNS if c in df.columns])
    encoders = {}
    for column in features.select_dtypes(include=["object", "category"]).columns:
        encoder = LabelEncoder()
        features[column] = encoder.fit_transform(features[column].astype(str))
        encoders[column] = encoder

    X_train, X_test, y_train, y_test = train_test_split(
        features, df[TARGET], test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=df[TARGET]
    )
    tree = DecisionTreeClassifier(criterion="entropy", random_state=RANDOM_STATE)
    tree.fit(X_train, y_train)
    # max_features=None keeps every feature available at each split; restricting it
    # (e.g. "sqrt") hides the few decisive features and cuts High-need recall sharply.
    forest = RandomForestClassifier(
        n_estimators=FOREST_TREES,
        criterion="entropy",
        max_features=None,
        min_samples_leaf=5,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    forest.fit(X_train, y_train)
    return TrainedModel(tree, forest, encoders, features.columns.tolist(), X_train, X_test, y_train, y_test)


def irrigation_stress(probabilities):
    """Weighted 0-100 stress score from class probabilities."""
    return 100 * (
        probabilities.get("Low", 0) * 0.1
        + probabilities.get("Medium", 0) * 0.5
        + probabilities.get("High", 0) * 1.0
    )


def add_field_recommendations(df):
    """Add priority score, action, water depth (mm) and volume (L) per field."""
    out = df.copy()
    need = out[TARGET]
    out["Priority_Score"] = need.map(PRIORITY_SCORES)
    out["Recommended_Action"] = need.map(RECOMMENDED_ACTIONS)
    moisture_deficit = (TARGET_MOISTURE - out["Soil_Moisture"]).clip(lower=0)
    water_mm = (need.map(BASE_WATER_MM) + moisture_deficit * 0.25).clip(upper=MAX_WATER_MM)
    out["Recommended_Water_mm"] = water_mm.where(need != "Low", 0).round(1)
    # 1 mm of water over 1 hectare = 10,000 litres.
    out["Estimated_Water_L"] = out["Field_Area_hectare"] * out["Recommended_Water_mm"] * 10_000
    return out
