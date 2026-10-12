import pandas as pd
import pytest

from irrigation_core import (
    CLASS_ORDER,
    MAP_COLUMNS,
    add_field_recommendations,
    irrigation_stress,
    load_data,
    train_model,
)


@pytest.fixture(scope="module")
def data():
    return load_data()


@pytest.fixture(scope="module")
def trained(data):
    return train_model(data)


def sample_values(data, index=0):
    return data.iloc[index].to_dict()


def test_location_columns_are_not_features(trained):
    assert not set(MAP_COLUMNS) & set(trained.feature_columns)
    assert "Irrigation_Need" not in trained.feature_columns


def test_model_beats_majority_baseline(data, trained):
    baseline = data["Irrigation_Need"].value_counts(normalize=True).max()
    assert trained.evaluation()["accuracy"] > baseline + 0.1


def test_predict_returns_known_class_and_normalised_probabilities(data, trained):
    prediction, probabilities = trained.predict(sample_values(data))
    assert prediction in CLASS_ORDER
    assert set(probabilities) == set(CLASS_ORDER)
    assert sum(probabilities.values()) == pytest.approx(1.0)


def test_predict_rejects_missing_inputs(data, trained):
    values = sample_values(data)
    del values["Soil_Moisture"]
    with pytest.raises(ValueError, match="Soil_Moisture"):
        trained.predict(values)


def test_predict_rejects_unknown_category(data, trained):
    values = sample_values(data)
    values["Crop_Type"] = "Banana"
    with pytest.raises(ValueError):
        trained.predict(values)


def test_category_options_match_dataset(data, trained):
    assert set(trained.category_options("Crop_Type")) == set(data["Crop_Type"].unique())


def test_irrigation_stress_bounds():
    assert irrigation_stress({"High": 1.0}) == pytest.approx(100)
    assert irrigation_stress({"Low": 1.0}) == pytest.approx(10)
    assert irrigation_stress({"Medium": 0.5, "Low": 0.5}) == pytest.approx(30)


def test_field_recommendations():
    fields = pd.DataFrame({
        "Irrigation_Need": ["High", "Medium", "Low", "High"],
        "Soil_Moisture": [15.0, 40.0, 10.0, 60.0],
        "Field_Area_hectare": [2.0, 1.0, 5.0, 1.0],
    })
    result = add_field_recommendations(fields)
    # High: 35 + (45 - 15) * 0.25 = 42.5 mm; Medium: 20 + 5 * 0.25 = 21.25 -> 21.2 mm.
    assert result["Recommended_Water_mm"].tolist() == [42.5, 21.2, 0.0, 35.0]
    assert result.loc[0, "Estimated_Water_L"] == pytest.approx(2.0 * 42.5 * 10_000)
    assert result.loc[2, "Estimated_Water_L"] == 0
    assert result["Recommended_Action"].tolist()[2] == "Monitor only"
    assert "Priority_Score" not in fields.columns  # input left untouched


def test_water_depth_is_capped():
    fields = pd.DataFrame({
        "Irrigation_Need": ["High"], "Soil_Moisture": [0.0], "Field_Area_hectare": [1.0],
    })
    assert add_field_recommendations(fields).loc[0, "Recommended_Water_mm"] == 45.0


def test_forest_gives_graded_probabilities(trained):
    tree_confidence = trained.tree.predict_proba(trained.X_test).max(axis=1)
    forest_confidence = trained.forest.predict_proba(trained.X_test).max(axis=1)
    assert (tree_confidence == 1.0).all()  # the problem being solved: pure leaves only
    assert (forest_confidence < 0.9).sum() > 10


def test_forest_is_better_calibrated_without_losing_accuracy(trained):
    forest, tree = trained.evaluation(), trained.evaluation(trained.tree)
    assert forest["log_loss"] < tree["log_loss"]
    assert forest["accuracy"] >= tree["accuracy"] - 0.005
    high_recall = forest["report"].loc["High", "recall"]
    assert high_recall >= 0.95
