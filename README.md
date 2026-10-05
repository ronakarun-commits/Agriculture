# Smart Irrigation Predictor

A Streamlit dashboard for agricultural irrigation decision support. The application combines a machine-learning classifier with field-level operational guidance and an interactive map.

## What the project does

The dashboard has three main areas:

1. **Irrigation Predictor**
   - Collects crop, soil, weather, water-source, and field observations.
   - Trains an ID3-style decision tree using entropy.
   - Predicts irrigation need as `High`, `Medium`, or `Low`.
   - Shows class probabilities, pump status, irrigation stress, and an input summary.

2. **Visualizations**
   - Displays the trained decision tree.
   - Shows a confusion matrix.
   - Shows feature importance.
   - Compares accuracy across tree depths.

3. **GIS Satellite Map**
   - Plots field records by latitude and longitude.
   - Filters by region and irrigation urgency.
   - Shows soil moisture, rainfall, field size, and crop information on hover.
   - Calculates a priority score, recommended action, estimated irrigation depth, and estimated water volume.
   - Displays a ranked table of fields needing attention.

## Repository contents

- `smart_irrigation_predictor.py` - Streamlit application entry point.
- `irrigation_prediction.csv` - Training and map dataset with 10,000 field records.
- `requirements.txt` - Python dependencies used by the application.
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

## Model behavior

The application encodes categorical values with `LabelEncoder`, splits the data with `test_size=0.2` and `random_state=42`, and trains a `DecisionTreeClassifier` with `criterion="entropy"`.

The current dataset produces approximately 99.5% test accuracy in the local smoke test. This result should not be treated as production performance until the model is evaluated on independently collected field data.

## Map coordinate note

The current latitude and longitude values are deterministic demo coordinates clustered around inland regional centers. They are not surveyed field locations. For production use, replace them with real GPS measurements and, if available, field-boundary polygons or GeoJSON.

## Recommended next steps

- Replace demo coordinates with real GPS or sensor data.
- Add weather forecasts and evapotranspiration.
- Record historical soil-moisture and irrigation readings.
- Compare the decision tree with Random Forest or Gradient Boosting.
- Add authentication, alerts, data upload validation, and irrigation scheduling.
