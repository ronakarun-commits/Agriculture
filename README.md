# Smart Irrigation Predictor

A Streamlit dashboard for agricultural irrigation decision support. The application combines a machine-learning classifier with field-level operational guidance and an interactive map.

## What the project does

The dashboard has three main areas:

1. **Irrigation Predictor**
   - Collects crop, soil, weather, water-source, and field observations.
   - Predicts with a random forest of ID3-style (entropy) decision trees.
   - Predicts irrigation need as `High`, `Medium`, or `Low`.
   - Shows class probabilities, pump status, irrigation stress, and an input summary.
   - Dropdown options come from the categories the model was trained on.

2. **Visualizations**
   - Displays a single interpretable entropy tree (adjustable depth) and its full text rules.
   - Shows a confusion matrix, per-class precision, recall, and F1, and log loss for the forest vs. the single tree.
   - Shows feature importance.
   - Compares test accuracy across tree depths.
   - Explanatory notes are computed from the trained model, so they stay accurate when the data changes.

3. **GIS Satellite Map**
   - Plots field records by latitude and longitude.
   - Filters by region and irrigation urgency.
   - Shows soil moisture, rainfall, field size, and crop information on hover.
   - Calculates a priority score, recommended action, estimated irrigation depth, and estimated water volume.
   - Displays a ranked table of fields needing attention, downloadable as CSV.

## Repository contents

- `smart_irrigation_predictor.py` - Streamlit application entry point (UI only).
- `irrigation_core.py` - Data loading, model training, evaluation, and field water recommendations, with no Streamlit dependency.
- `tests/` - Pytest suite for `irrigation_core.py`.
- `irrigation_prediction.csv` - Training and map dataset with 10,000 field records.
- `requirements.txt` - Python dependencies used by the application.
- `requirements-dev.txt` - Adds test dependencies.
- `.gitignore` - Ignores Python caches, virtual environments, and local environment files.

## Dataset fields

The model uses soil, weather, crop, irrigation, and field-management fields, including:

- Soil type, pH, moisture, organic carbon, and electrical conductivity
- Temperature, humidity, rainfall, sunlight, and wind speed
- Crop type and growth stage
- Season, irrigation type, water source, and mulching
- Field area, previous irrigation, and region
- `Irrigation_Need`, the prediction target

The map also uses `Field_ID`, `Latitude`, `Longitude`, and `Coordinate_Source`. These map fields are deliberately excluded from model training so location metadata does not become an accidental prediction feature.

## Setup

Use Python 3.13 or a compatible recent Python version.

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

## Run the dashboard

From the project directory:

```powershell
streamlit run smart_irrigation_predictor.py
```

Then open `http://localhost:8501`.

## Run the tests

```powershell
python -m pip install -r requirements-dev.txt
python -m pytest
```

## Model behavior

The application encodes categorical values with `LabelEncoder` and makes a stratified split with `test_size=0.2` and `random_state=42`. It then trains two models on the same split:

- **Prediction model:** a `RandomForestClassifier` of 100 trees with `criterion="entropy"`, `max_features=None`, and `min_samples_leaf=5`. Averaging the trees' votes gives graded probabilities for borderline fields; a lone unpruned tree ends in pure leaves and only ever reports 0% or 100%.
- **Explanation model:** a single unpruned `DecisionTreeClassifier` with `criterion="entropy"` (depth 7, 53 leaves on the current dataset), used for the tree diagram, feature importance, and depth analysis.

Both reach about 99.8% test accuracy, including 100% recall on the rare `High` class (about 3.4% of records). The forest's log loss is roughly 0.009 against 0.07 for the single tree, so its probabilities are much better calibrated. Restricting features per split (`max_features="sqrt"`) or pruning the single tree was tested and rejected, because both cut `High` recall sharply.

The forest takes around 5 seconds to train on first load; Streamlit caches it after that. This result should not be treated as production performance until the model is evaluated on independently collected field data.

## Map coordinate note

The current latitude and longitude values are deterministic demo coordinates clustered around inland regional centers. They are not surveyed field locations. For production use, replace them with real GPS measurements and, if available, field-boundary polygons or GeoJSON.

## Recommended next steps

- Replace demo coordinates with real GPS or sensor data.
- Add weather forecasts and evapotranspiration.
- Record historical soil-moisture and irrigation readings.
- Compare the forest with Gradient Boosting, and use cross-validation for hyperparameter selection.
- Add authentication, alerts, data upload validation, and irrigation scheduling.
