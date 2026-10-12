import matplotlib.pyplot as plt
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.tree import export_text, plot_tree

from irrigation_core import (
    CLASS_ORDER,
    TARGET,
    add_field_recommendations,
    irrigation_stress,
    load_data,
    train_model,
)

NEED_COLORS = {"High": "#d84343", "Medium": "#f0a020", "Low": "#3f9d5a"}

st.set_page_config(
    page_title="Smart Irrigation Predictor",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown(
    """
    <style>
    :root {
        --ink: #17324d;
        --muted: #607286;
        --leaf: #1f7a5a;
        --line: #d9e5e8;
    }
    .block-container { padding-top: 2rem; padding-bottom: 3rem; }
    .app-hero {
        background: linear-gradient(110deg, #17324d 0%, #1f7a5a 100%);
        border-radius: 16px;
        padding: 1.6rem 2rem;
        margin-bottom: 1.5rem;
        color: white;
        box-shadow: 0 10px 28px rgba(23, 50, 77, 0.14);
    }
    .app-hero h1 { margin: 0; font-size: 2.15rem; letter-spacing: 0; color: white; }
    .app-hero p { margin: 0.45rem 0 0; color: #d7eee5; font-size: 1rem; }
    .section-kicker {
        color: #b9ead5;
        font-size: 0.75rem;
        font-weight: 800;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        margin-bottom: 0.25rem;
    }
    div[data-testid="stMetric"] {
        border: 1px solid var(--line);
        border-left: 4px solid var(--leaf);
        padding: 0.8rem 1rem;
        border-radius: 10px;
    }
    .pump-card {
        border-radius: 12px;
        padding: 22px;
        text-align: center;
        color: white;
    }
    .pump-card h4 { margin: 0; opacity: 0.85; color: white; }
    .pump-card h1 { margin: 5px 0; font-size: 40px; color: white; }
    .pump-card p { margin: 0; opacity: 0.9; }
    </style>
    """,
    unsafe_allow_html=True
)


@st.cache_data
def get_data():
    return load_data()


@st.cache_resource
def get_model():
    return train_model(get_data())


@st.cache_data
def get_depth_curve():
    return get_model().depth_curve()


df_raw = get_data()
trained = get_model()
tree = trained.tree

st.markdown(
    "<div class='app-hero'><div class='section-kicker'>FIELD INTELLIGENCE</div>"
    "<h1>Smart Irrigation Predictor</h1>"
    "<p>Decision support for healthier crops, clearer priorities, and smarter water use.</p></div>",
    unsafe_allow_html=True
)

PAGES = ["Irrigation Predictor", "Visualizations", "GIS Satellite Map"]
with st.sidebar:
    st.title("Navigation")
    page = st.radio("Page", PAGES, label_visibility="collapsed")
    st.markdown("---")
    st.caption(
        f"Model: random forest of {trained.forest.n_estimators} entropy trees · {len(df_raw):,} records · "
        f"test accuracy {trained.evaluation()['accuracy']:.1%}"
    )


def styled_slider(label, min_val, max_val, default, step=1.0):
    """Slider with a bold value readout aligned to the right of its label."""
    if label not in st.session_state:
        st.session_state[label] = float(default)

    st.markdown(
        f"<div style='display: flex; justify-content: space-between; align-items: center; margin: 6px 0 8px;'>"
        f"<span style='font-size: 15px; font-weight: 600;'>{label}</span>"
        f"<span style='font-size: 20px; font-weight: bold;'>{st.session_state[label]:g}</span>"
        f"</div>",
        unsafe_allow_html=True
    )
    return st.slider(
        label=label,
        min_value=float(min_val),
        max_value=float(max_val),
        step=float(step),
        key=label,
        label_visibility="collapsed"
    )


def option_select(label, column):
    return st.selectbox(label, trained.category_options(column))


def stress_gauge(score):
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=score,
        number={"suffix": "%", "font": {"size": 36}},
        title={"text": "Irrigation Stress Level", "font": {"size": 18}},
        gauge={
            "axis": {"range": [0, 100], "dtick": 20},
            "bar": {"color": "#17324d", "thickness": 0.3},
            "steps": [
                {"range": [0, 35], "color": "#9fd8b0"},
                {"range": [35, 70], "color": "#ffd58a"},
                {"range": [70, 100], "color": "#f4a3a3"}
            ]
        }
    ))
    fig.update_layout(height=270, margin=dict(l=30, r=30, t=60, b=10))
    return fig


# ==========================================
# PAGE 1: IRRIGATION PREDICTOR
# ==========================================
if page == "Irrigation Predictor":
    st.subheader("Enter Field & Environmental Details")
    st.markdown("Fill in key field observations below to evaluate irrigation requirements.")

    st.markdown("#### Core Field & Crop Parameters")
    col1, col2 = st.columns(2)

    with col1:
        crop_type = option_select("Crop Type", "Crop_Type")
        crop_stage = option_select("Crop Growth Stage", "Crop_Growth_Stage")
        soil_type = option_select("Soil Type", "Soil_Type")

    with col2:
        soil_moisture = styled_slider("Soil Moisture (%)", 8.0, 65.0, 35.0, step=0.5)
        season = option_select("Season", "Season")
        mulching = option_select("Mulching Used?", "Mulching_Used")

    with st.expander("Advanced Soil, Weather & Secondary Parameters", expanded=False):
        col_adv1, col_adv2 = st.columns(2)
        with col_adv1:
            temp = styled_slider("Temperature (°C)", 12.0, 42.0, 27.0, step=0.5)
            humidity = styled_slider("Humidity (%)", 25.0, 95.0, 60.0, step=1.0)
            rainfall = styled_slider("Rainfall (mm)", 0.0, 2500.0, 1200.0, step=10.0)
            sunlight = styled_slider("Sunlight Hours/Day", 4.0, 11.0, 7.5, step=0.1)
            wind_speed = styled_slider("Wind Speed (km/h)", 0.5, 20.0, 10.0, step=0.5)
            field_area = styled_slider("Field Area (Hectares)", 0.3, 15.0, 5.0, step=0.1)

        with col_adv2:
            soil_ph = styled_slider("Soil pH", 4.8, 8.2, 6.5, step=0.1)
            organic_carbon = styled_slider("Organic Carbon (%)", 0.3, 1.6, 0.9, step=0.05)
            electrical_conductivity = styled_slider("Electrical Conductivity (dS/m)", 0.1, 3.5, 1.8, step=0.1)
            irrigation_type = option_select("Current Irrigation Type", "Irrigation_Type")
            water_source = option_select("Water Source", "Water_Source")
            previous_irrigation = styled_slider("Previous Irrigation (mm)", 0.0, 120.0, 50.0, step=1.0)
            region = option_select("Region", "Region")

    if soil_moisture < 20.0:
        st.warning("Soil moisture is critically low. High irrigation may be required.")
    if rainfall < 500.0:
        st.info("Low rainfall region detected. Monitor water availability closely.")
    if temp > 35.0:
        st.warning("High temperatures will increase evapotranspiration rates.")

    field_values = {
        "Soil_Type": soil_type,
        "Soil_pH": soil_ph,
        "Soil_Moisture": soil_moisture,
        "Organic_Carbon": organic_carbon,
        "Electrical_Conductivity": electrical_conductivity,
        "Temperature_C": temp,
        "Humidity": humidity,
        "Rainfall_mm": rainfall,
        "Sunlight_Hours": sunlight,
        "Wind_Speed_kmh": wind_speed,
        "Crop_Type": crop_type,
        "Crop_Growth_Stage": crop_stage,
        "Season": season,
        "Irrigation_Type": irrigation_type,
        "Water_Source": water_source,
        "Field_Area_hectare": field_area,
        "Mulching_Used": mulching,
        "Previous_Irrigation_mm": previous_irrigation,
        "Region": region,
    }

    if st.button("Predict Irrigation Need & Pump Status", type="primary", width="stretch"):
        prediction, probabilities = trained.predict(field_values)

        st.markdown("---")
        st.subheader("Prediction Result & Automated Actuation")

        pump_col1, pump_col2 = st.columns([2, 3])
        pump_states = {
            "High": ("#1b5e20", "ON", "Irrigation valve opened · actuator active"),
            "Medium": ("#c75b00", "STANDBY", "Scheduled for evening run"),
            "Low": ("#455a64", "OFF", "Moisture sufficient · pump deactivated"),
        }
        background, status, detail = pump_states[prediction]
        with pump_col1:
            st.markdown(
                f"<div class='pump-card' style='background-color: {background};'>"
                f"<h4>PUMP STATUS</h4><h1>{status}</h1><p>{detail}</p></div>",
                unsafe_allow_html=True
            )

        with pump_col2:
            st.plotly_chart(stress_gauge(irrigation_stress(probabilities)))

        st.markdown("### Field Condition Analysis")
        if prediction == "High":
            st.error("**HIGH IRRIGATION NEED** — Critical moisture deficit detected. Automated pump switch activated.")
        elif prediction == "Medium":
            st.warning("**MEDIUM IRRIGATION NEED** — Moderate moisture levels. Irrigation queued for the optimal timeframe.")
        else:
            st.success("**LOW IRRIGATION NEED** — Soil moisture levels are adequate. Water conservation mode active.")

        col3, col4, col5 = st.columns(3)
        col3.metric("High Need Probability", f"{probabilities['High']:.1%}")
        col4.metric("Medium Need Probability", f"{probabilities['Medium']:.1%}")
        col5.metric("Low Need Probability", f"{probabilities['Low']:.1%}")
        st.caption(
            f"Probabilities are the averaged votes of {trained.forest.n_estimators} entropy decision trees. "
            "Clear-cut fields score near 100%; inputs close to a decision boundary show split probabilities."
        )

        st.markdown("---")
        st.subheader("Input Summary")
        summary = pd.DataFrame({
            "Parameter": ["Soil Type", "Soil pH", "Soil Moisture", "Crop", "Growth Stage", "Season",
                          "Temperature", "Rainfall", "Mulching"],
            "Value": [soil_type, f"{soil_ph:g}", f"{soil_moisture:g}%", crop_type, crop_stage, season,
                      f"{temp:g} °C", f"{rainfall:g} mm", mulching]
        })
        st.dataframe(summary, width="stretch", hide_index=True)

# ==========================================
# PAGE 2: VISUALIZATIONS
# ==========================================
if page == "Visualizations":
    st.subheader("Model Visualizations & Performance Metrics")
    st.markdown(
        f"Predictions come from a random forest of {trained.forest.n_estimators} ID3 (entropy) trees. "
        "The tree diagram, feature importance and depth analysis use a single entropy tree trained "
        "on the same data, because one tree is readable where a forest is not."
    )

    evaluation = trained.evaluation()
    tree_evaluation = trained.evaluation(tree)
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Forest Test Accuracy", f"{evaluation['accuracy']:.1%}")
    m2.metric("Single Tree Accuracy", f"{tree_evaluation['accuracy']:.1%}")
    m3.metric("Tree Depth / Leaves", f"{tree.get_depth()} / {tree.get_n_leaves()}")
    m4.metric(
        "Log Loss (forest vs tree)", f"{evaluation['log_loss']:.3f}",
        delta=f"{evaluation['log_loss'] - tree_evaluation['log_loss']:.3f} vs tree", delta_color="inverse",
        help="Lower is better. Measures how well predicted probabilities match outcomes."
    )

    tab1, tab2, tab3, tab4 = st.tabs([
        "Decision Tree Architecture",
        "Model Evaluation",
        "Feature Importance",
        "Accuracy vs Tree Depth"
    ])

    with tab1:
        st.markdown("#### Trained Decision Tree Diagram")
        max_drawn = min(5, tree.get_depth())
        show_depth = st.slider("Levels to draw", 1, max_drawn, min(3, max_drawn))
        st.caption(
            "Each node shows its entropy split, sample count, and majority class. "
            "Deeper levels are collapsed to keep the diagram readable; see the text rules for the full tree."
        )
        fig_tree, ax = plt.subplots(figsize=(min(3 * 2 ** show_depth, 40), 2.2 * (show_depth + 1)))
        plot_tree(
            tree,
            max_depth=show_depth,
            feature_names=trained.feature_columns,
            class_names=[str(c) for c in tree.classes_],
            filled=True,
            rounded=True,
            fontsize=8,
            ax=ax
        )
        st.pyplot(fig_tree)
        plt.close(fig_tree)

        with st.expander("Full tree as text rules"):
            st.code(export_text(tree, feature_names=trained.feature_columns), language="text")

    with tab2:
        fig = px.imshow(
            evaluation["confusion_matrix"],
            text_auto=True,
            x=CLASS_ORDER,
            y=CLASS_ORDER,
            title="Forest Confusion Matrix (Irrigation Need Prediction)",
            color_continuous_scale="Blues"
        )
        fig.update_xaxes(title="Predicted Class")
        fig.update_yaxes(title="Actual Class")
        st.plotly_chart(fig)

        report = evaluation["report"].loc[CLASS_ORDER, ["precision", "recall", "f1-score", "support"]]
        st.markdown("#### Per-class metrics")
        st.dataframe(
            report.style.format({"precision": "{:.1%}", "recall": "{:.1%}", "f1-score": "{:.1%}",
                                 "support": "{:.0f}"}),
            width="stretch"
        )
        class_share = df_raw[TARGET].value_counts(normalize=True)
        st.info(
            f"High need makes up only {class_share.get('High', 0):.1%} of the dataset, so check its recall "
            "here rather than relying on overall accuracy alone."
        )

    with tab3:
        importance_df = trained.feature_importance()
        fig = px.bar(
            importance_df[importance_df["Importance"] > 0].sort_values("Importance"),
            x="Importance",
            y="Feature",
            orientation="h",
            title="ID3 Information Gain / Feature Importances",
            color="Importance",
            color_continuous_scale="Blues"
        )
        st.plotly_chart(fig)
        top_features = ", ".join(importance_df["Feature"].head(4).str.replace("_", " "))
        unused = (importance_df["Importance"] == 0).sum()
        st.info(
            f"The strongest drivers of the tree's splits are {top_features}. "
            f"{unused} of {len(importance_df)} features are never used by the trained tree."
        )

    with tab4:
        st.markdown("### Accuracy vs Tree Depth")
        depth_curve = get_depth_curve()
        fig = px.line(
            depth_curve, x="Max Depth", y="Accuracy", markers=True,
            title="ID3 Decision Tree Test Accuracy across Max Depths"
        )
        fig.update_yaxes(tickformat=".0%")
        st.plotly_chart(fig)
        best_accuracy = depth_curve["Accuracy"].max()
        plateau_depth = int(depth_curve.loc[depth_curve["Accuracy"] >= best_accuracy - 0.005, "Max Depth"].min())
        st.info(
            f"Accuracy reaches within 0.5 points of its best ({best_accuracy:.1%}) at depth {plateau_depth}. "
            f"The unpruned tree grows to depth {tree.get_depth()} on its own."
        )

# ==========================================
# PAGE 3: GIS SATELLITE MAP
# ==========================================
if page == "GIS Satellite Map":
    st.subheader("GIS Satellite & Spatial Moisture Analytics")
    st.markdown(
        "Identifies irrigation urgency, ranks fields for action, and estimates the water volume "
        "required for each zone."
    )
    st.markdown("---")

    sample_size = min(300, len(df_raw))
    map_df = add_field_recommendations(df_raw.sample(sample_size, random_state=42))
    coordinate_source = map_df["Coordinate_Source"].iloc[0] if "Coordinate_Source" in map_df else "Unknown"
    st.caption(
        f"Showing {sample_size} sampled fields. Coordinate source: {coordinate_source}. "
        "Replace demo coordinates with surveyed GPS data for production use."
    )

    col_map1, col_map2 = st.columns([3, 1])

    with col_map2:
        st.markdown("#### GIS Map Filters")
        region_options = ["All Regions"] + sorted(map_df["Region"].unique())
        selected_region = st.selectbox("Select Region", options=region_options, index=0)
        need_options = ["All Urgency Levels", "High", "Medium", "Low"]
        selected_need = st.selectbox("Filter Irrigation Need", options=need_options, index=0)
        color_by = st.selectbox(
            "Color Code By",
            options=["Irrigation_Need", "Priority_Score", "Soil_Moisture", "Crop_Type"]
        )

    filtered_map_df = map_df
    if selected_region != "All Regions":
        filtered_map_df = filtered_map_df[filtered_map_df["Region"] == selected_region]
    if selected_need != "All Urgency Levels":
        filtered_map_df = filtered_map_df[filtered_map_df[TARGET] == selected_need]

    with col_map1:
        if filtered_map_df.empty:
            st.warning("No fields match the selected filters.")
        else:
            fig_map = px.scatter_map(
                filtered_map_df,
                lat="Latitude",
                lon="Longitude",
                color=color_by,
                size="Field_Area_hectare",
                hover_name="Crop_Type",
                hover_data={
                    "Field_ID": True,
                    "Soil_Moisture": ":.1f",
                    "Rainfall_mm": ":.0f",
                    "Temperature_C": ":.1f",
                    TARGET: True,
                    "Recommended_Action": True,
                    "Recommended_Water_mm": True,
                    "Estimated_Water_L": ":,.0f",
                    "Latitude": False,
                    "Longitude": False
                },
                color_discrete_map=NEED_COLORS if color_by == TARGET else None,
                category_orders={TARGET: ["High", "Medium", "Low"]},
                zoom=4.2,
                center={"lat": 20.5937, "lon": 78.9629},
                map_style="open-street-map",
                title="Field Irrigation Priority Map"
            )
            fig_map.update_layout(height=550, margin={"r": 0, "t": 40, "l": 0, "b": 0})
            st.plotly_chart(fig_map)

    st.markdown("---")
    st.markdown("### Irrigation Action Summary")
    has_rows = not filtered_map_df.empty
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Active Fields Monitored", len(filtered_map_df))
    c2.metric("Avg Soil Moisture", f"{filtered_map_df['Soil_Moisture'].mean():.1f}%" if has_rows else "N/A")
    c3.metric("High Need Fields", int((filtered_map_df[TARGET] == "High").sum()))
    c4.metric("Total Monitored Area", f"{filtered_map_df['Field_Area_hectare'].sum():.1f} Ha")
    c5.metric("Estimated Water", f"{filtered_map_df['Estimated_Water_L'].sum():,.0f} L")

    st.markdown("### Highest Priority Fields")
    priority_columns = [
        "Field_ID", "Region", "Crop_Type", "Soil_Moisture", TARGET,
        "Priority_Score", "Recommended_Action", "Recommended_Water_mm", "Estimated_Water_L"
    ]
    priority_table = filtered_map_df.sort_values(
        ["Priority_Score", "Estimated_Water_L"], ascending=False
    )[priority_columns]
    st.dataframe(priority_table.head(15), width="stretch", hide_index=True)
    st.download_button(
        "Download full priority list (CSV)",
        priority_table.to_csv(index=False).encode("utf-8"),
        file_name="irrigation_priority_fields.csv",
        mime="text/csv",
        disabled=not has_rows
    )
