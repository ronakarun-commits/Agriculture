import streamlit as st
import pandas as pd
import numpy as np
import math
import time
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
from sklearn.tree import DecisionTreeClassifier, export_text, plot_tree
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score

st.set_page_config(
    page_title="Smart Irrigation Predictor",
    page_icon="🌾",
    layout="wide",  
    initial_sidebar_state="expanded"
)

@st.cache_resource
def train_id3_model():
    # Read CSV dataset
    df = pd.read_csv('irrigation_prediction.csv')
    data = df.copy()
    
    # Store fitted encoders for categorical columns
    encoders = {}
    categorical_cols = data.select_dtypes(include=['object', 'category']).columns
    
    for col in categorical_cols:
        if col != 'Irrigation_Need': # Keep target separate
            le = LabelEncoder()
            data[col] = le.fit_transform(data[col].astype(str))
            encoders[col] = le

    # Define target and features using exact column name 'Irrigation_Need'
    X = data.drop(columns=['Irrigation_Need'])
    y = df['Irrigation_Need'] # Keep target as string ('High', 'Medium', 'Low')

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train ID3 Decision Tree (Entropy criterion)
    model = DecisionTreeClassifier(criterion='entropy', random_state=42)
    model.fit(X_train, y_train)
    
    return model, encoders, X_train.columns.tolist(), X_test, y_test, df

# Load data and train model
model, label_encoders, feature_columns, X_test, y_test, df_raw = train_id3_model()

# Header
st.markdown(
    "<h1 style='text-align: center;'>🌾 Smart Irrigation Predictor</h1>", 
    unsafe_allow_html=True
)
st.markdown(
    "<h3 style='text-align: center;'>ML Based Irrigation and Spatial Analytics</h3>", 
    unsafe_allow_html=True
)
st.markdown("---")

# Initialize Session State
if "page" not in st.session_state:
    st.session_state.page = "Irrigation Predictor"

# Sidebar Navigation
st.sidebar.title("Navigation")
with st.sidebar:
    if st.button("Irrigation Predictor", use_container_width=True):
        st.session_state.page = "Irrigation Predictor"
    if st.button("Visualizations", use_container_width=True):
        st.session_state.page = "Visualizations"
    if st.button("GIS Satellite Map", use_container_width=True):
        st.session_state.page = "GIS Satellite Map"

# Helper Function for Custom Styled Sliders (Fixed Flexbox Layout to Prevent Shrinkage)
def styled_slider(label, min_val, max_val, default, step=1.0, icon=""):
    if label not in st.session_state:
        st.session_state[label] = default

    st.markdown(
        f"<div style='display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px; margin-top: 6px;'>"
        f"<span style='font-size: 15px; font-weight: 600;'>{icon} {label}</span>"
        f"<span style='font-size: 20px; font-weight: bold;'>{st.session_state[label]}</span>"
        f"</div>",
        unsafe_allow_html=True
    )

    value = st.slider(
        label=label,
        min_value=float(min_val),
        max_value=float(max_val),
        step=float(step),
        key=label,
        label_visibility="collapsed"
    )
    return value

# ==========================================
# PAGE 1: IRRIGATION PREDICTOR
# ==========================================
if st.session_state.page == "Irrigation Predictor":
    st.subheader("Enter Field & Environmental Details")
    st.markdown("Fill in key field observations below to evaluate irrigation requirements.")

    # Core Operational Factors
    st.markdown("#### Core Field & Crop Parameters")
    col1, col2 = st.columns(2)

    with col1:
        crop_type = st.selectbox("Crop Type", ["Wheat", "Maize", "Cotton", "Rice", "Sugarcane", "Potato"])
        crop_stage = st.selectbox("Crop Growth Stage", ["Sowing", "Vegetative", "Flowering", "Harvest"])
        soil_type = st.selectbox("Soil Type", ["Clay", "Silt", "Sandy", "Loamy"])

    with col2:
        soil_moisture = styled_slider("Soil Moisture (%)", 8.0, 65.0, 35.0, step=0.5)
        season = st.selectbox("Season", ["Rabi", "Kharif", "Zaid"])
        mulching = st.selectbox("Mulching Used?", ["Yes", "No"])

    # Secondary / Environmental Factors in clean expander
    with st.expander("Advanced Soil, Weather & Secondary Parameters", expanded=False):
        col_adv1, col_adv2 = st.columns(2)
        with col_adv1:
            temp = styled_slider("Temperature (°C)", 12.0, 42.0, 27.0, step=0.5)
            humidity = styled_slider("Humidity (%)", 25.0, 95.0, 60.0, step=1.0)
            rainfall = styled_slider("Rainfall (mm)", 0.0, 2500.0, 1200.0, step=10.0)
            sunlight = styled_slider("Sunlight Hours/Day", 4.0, 11.0, 7.5, step=0.1)
            wind_speed = styled_slider("Wind Speed (km/h)", 0.5, 20.0, 10.0, step=0.5)
            field_area = styled_slider("Field Area (Hectares)", 0.3, 15.0, 5.0, step=0.5)

        with col_adv2:
            soil_ph = styled_slider("Soil pH", 4.8, 8.2, 6.5, step=0.1)
            organic_carbon = styled_slider("Organic Carbon (%)", 0.3, 1.6, 0.9, step=0.05)
            electrical_conductivity = styled_slider("Electrical Conductivity (dS/m)", 0.1, 3.5, 1.8, step=0.1)
            irrigation_type = st.selectbox("Current Irrigation Type", ["Rainfed", "Canal", "Drip", "Sprinkler"])
            water_source = st.selectbox("Water Source", ["Reservoir", "Groundwater", "River", "Rainwater"])
            previous_irrigation = styled_slider("Previous Irrigation (mm)", 0.0, 120.0, 50.0, step=1.0)
            region = st.selectbox("Region", ["North", "South", "East", "West", "Central"])

    # Real-time Threshold Alerts
    if soil_moisture < 20.0:
        st.warning("Soil Moisture is critically low! High irrigation may be required.")
    if rainfall < 500.0:
        st.info("Low rainfall region detected. Monitor water availability closely.")
    if temp > 35.0:
        st.warning("High temperatures will increase evapotranspiration rates.")

    # Format input row for model prediction
    input_dict = {
        'Soil_Type': label_encoders['Soil_Type'].transform([soil_type])[0],
        'Soil_pH': soil_ph,
        'Soil_Moisture': soil_moisture,
        'Organic_Carbon': organic_carbon,
        'Electrical_Conductivity': electrical_conductivity,
        'Temperature_C': temp,
        'Humidity': humidity,
        'Rainfall_mm': rainfall,
        'Sunlight_Hours': sunlight,
        'Wind_Speed_kmh': wind_speed,
        'Crop_Type': label_encoders['Crop_Type'].transform([crop_type])[0],
        'Crop_Growth_Stage': label_encoders['Crop_Growth_Stage'].transform([crop_stage])[0],
        'Season': label_encoders['Season'].transform([season])[0],
        'Irrigation_Type': label_encoders['Irrigation_Type'].transform([irrigation_type])[0],
        'Water_Source': label_encoders['Water_Source'].transform([water_source])[0],
        'Field_Area_hectare': field_area,
        'Mulching_Used': label_encoders['Mulching_Used'].transform([mulching])[0],
        'Previous_Irrigation_mm': previous_irrigation,
        'Region': label_encoders['Region'].transform([region])[0]
    }

    input_df = pd.DataFrame([input_dict])

    # Prediction Action Button
    if st.button("Predict Irrigation Need & Pump Status", use_container_width=True):
        prediction = model.predict(input_df)[0]
        probabilities = model.predict_proba(input_df)[0]
        class_names = model.classes_
        
        prob_dict = dict(zip(class_names, probabilities))
        high_prob = prob_dict.get('High', 0.0) * 100
        med_prob = prob_dict.get('Medium', 0.0) * 100
        low_prob = prob_dict.get('Low', 0.0) * 100

        st.markdown("---")
        st.subheader("Prediction Result & Automated Actuation")

        # Top Banner: Automated Smart Pump Control Status
        pump_col1, pump_col2 = st.columns([2, 3])
        
        with pump_col1:
            if prediction == "High":
                st.markdown("""
                <div style="background-color: #1b5e20; border-radius: 10px; padding: 22px; text-align: center; color: white;">
                    <h4 style="margin:0; padding:0; color: #a5d6a7;">PUMP STATUS</h4>
                    <h1 style="margin:5px 0; font-size: 42px; color: #ffffff;">ON</h1>
                    <p style="margin:0; color: #c8e6c9;">Irrigation Valve Opened | Actuator Active</p>
                </div>
                """, unsafe_allow_html=True)
            elif prediction == "Medium":
                st.markdown("""
                <div style="background-color: #e65100; border-radius: 10px; padding: 22px; text-align: center; color: white;">
                    <h4 style="margin:0; padding:0; color: #ffe0b2;">PUMP STATUS</h4>
                    <h1 style="margin:5px 0; font-size: 38px; color: #ffffff;">STANDBY</h1>
                    <p style="margin:0; color: #fff3e0;">Scheduled for Evening Run</p>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown("""
                <div style="background-color: #b71c1c; border-radius: 10px; padding: 22px; text-align: center; color: white;">
                    <h4 style="margin:0; padding:0; color: #ef9a9a;">PUMP STATUS</h4>
                    <h1 style="margin:5px 0; font-size: 42px; color: #ffffff;">OFF</h1>
                    <p style="margin:0; color: #ffcdd2;">Moisture Sufficient | Pump Deactivated</p>
                </div>
                """, unsafe_allow_html=True)

        with pump_col2:
            target_stress = (low_prob * 0.1) + (med_prob * 0.5) + (high_prob * 1.0)
            
            def create_speedometer_gauge(score):
                angle_deg = 180 - (score / 100.0 * 180)
                angle_rad = math.radians(angle_deg)
                
                length = 0.35
                center_x, center_y = 0.5, 0.20
                
                tip_x = center_x + length * math.cos(angle_rad)
                tip_y = center_y + length * math.sin(angle_rad)
                
                base_width = 0.025
                perp_rad = angle_rad + math.pi / 2
                left_x = center_x + base_width * math.cos(perp_rad)
                left_y = center_y + base_width * math.sin(perp_rad)
                right_x = center_x - base_width * math.cos(perp_rad)
                right_y = center_y - base_width * math.sin(perp_rad)

                arrow_path = f"M {left_x} {left_y} L {tip_x} {tip_y} L {right_x} {right_y} Z"

                fig = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=score,
                    number={'suffix': "%", 'font': {'size': 32, 'color': 'white', 'weight': 'bold'}},
                    title={'text': "Irrigation Stress Level (%)", 'font': {'size': 20, 'color': '#ffffff'}},
                    domain={'x': [0, 1], 'y': [0, 1]},
                    gauge={
                        'axis': {'range': [0, 100], 'tickwidth': 2, 'tickcolor': "white", 'dtick': 20},
                        'bar': {'color': "rgba(0,0,0,0)"},
                        'steps': [
                            {'range': [0, 35], 'color': "#4caf50"},
                            {'range': [35, 70], 'color': "#ff9800"},
                            {'range': [70, 100], 'color': "#f44336"}
                        ]
                    }
                ))

                fig.add_shape(
                    type="path",
                    path=arrow_path,
                    fillcolor="#ffffff",
                    line=dict(color="#111111", width=1.5)
                )
                
                fig.add_shape(
                    type="circle",
                    x0=center_x - 0.04, y0=center_y - 0.04,
                    x1=center_x + 0.04, y1=center_y + 0.04,
                    fillcolor="#222222",
                    line=dict(color="#ffffff", width=2)
                )
                fig.add_shape(
                    type="circle",
                    x0=center_x - 0.02, y0=center_y - 0.02,
                    x1=center_x + 0.02, y1=center_y + 0.02,
                    fillcolor="#e0e0e0",
                    line=dict(color="#222222", width=1)
                )

                fig.update_layout(
                    height=290,
                    margin=dict(l=35, r=35, t=80, b=10),
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)"
                )
                return fig

            gauge_placeholder = st.empty()
            steps = 15
            for i in range(1, steps + 1):
                current_val = (target_stress / steps) * i
                fig_anim = create_speedometer_gauge(current_val)
                gauge_placeholder.plotly_chart(fig_anim, use_container_width=True, key=f"speedo_step_{i}")
                time.sleep(0.02)

        st.markdown("### Field Condition Analysis")
        if prediction == "High":
            st.error("**HIGH IRRIGATION NEED** — Critical moisture deficit detected! Automated pump switch activated.")
        elif prediction == "Medium":
            st.warning("**MEDIUM IRRIGATION NEED** — Moderate moisture levels. Drip irrigation queued for optimal timeframe.")
        else:
            st.success("**LOW IRRIGATION NEED** — Soil moisture levels are optimal. Water conservation mode active.")

        col3, col4, col5 = st.columns(3)
        with col3:
            st.metric("High Need Probability", f"{high_prob:.1f}%")
        with col4:
            st.metric("Medium Need Probability", f"{med_prob:.1f}%")
        with col5:
            st.metric("Low Need Probability", f"{low_prob:.1f}%")

        st.markdown("---")
        st.subheader("Input Summary")
        summary = pd.DataFrame({
            "Parameter": ["Soil Type", "Soil pH", "Soil Moisture", "Crop", "Growth Stage", "Season", "Temperature", "Rainfall", "Mulching"],
            "Value": [soil_type, f"{soil_ph}", f"{soil_moisture}%", crop_type, crop_stage, season, f"{temp} °C", f"{rainfall} mm", mulching]
        })
        st.dataframe(summary, use_container_width=True)

# ==========================================
# PAGE 2: VISUALIZATIONS
# ==========================================
if st.session_state.page == "Visualizations":
    st.subheader("Model Visualizations & Performance Metrics")
    st.markdown("Interactive insights generated from ID3 Decision Tree training.")

    tab1, tab2, tab3, tab4 = st.tabs([
        "Decision Tree Architecture",
        "Confusion Matrix",
        "Feature Importance",
        "Accuracy vs Tree Depth"
    ])

    with tab1:
        st.markdown("#### Trained Decision Tree Diagram")
        st.info("This plot illustrates the trained Scikit-Learn Decision Tree structure, showing entropy decision splits, evaluated features, sample numbers, and class predictions at each node.")
        
        fig_tree, ax = plt.subplots(figsize=(20, 10))
        plot_tree(
            model,
            feature_names=feature_columns,
            class_names=[str(c) for c in model.classes_],
            filled=True,
            rounded=True,
            fontsize=8,
            ax=ax
        )
        st.pyplot(fig_tree)

    with tab2:
        y_pred = model.predict(X_test)
        cm = confusion_matrix(y_test, y_pred, labels=['Low', 'Medium', 'High'])
        fig = px.imshow(
            cm,
            text_auto=True,
            x=["Low", "Medium", "High"],
            y=["Low", "Medium", "High"],
            title="Confusion Matrix (Irrigation Need Prediction)",
            color_continuous_scale="Blues"
        )
        fig.update_xaxes(title="Predicted Class")
        fig.update_yaxes(title="Actual Class")
        st.plotly_chart(fig, use_container_width=True)
        st.info("The confusion matrix shows high precision across Low and Medium irrigation classes, with strong diagonal alignment indicating accurate field predictions.")

    with tab3:
        importance_df = pd.DataFrame({
            "Feature": feature_columns,
            "Importance": model.feature_importances_
        }).sort_values(by="Importance", ascending=True)

        fig = px.bar(
            importance_df[importance_df["Importance"] > 0],
            x="Importance",
            y="Feature",
            orientation="h",
            title="ID3 Information Gain / Feature Importances",
            color="Importance",
            color_continuous_scale="Blues"
        )
        st.plotly_chart(fig, use_container_width=True)
        st.info("Growth Stage, Soil Moisture, Mulching, and Rainfall are the dominant features driving the model's split points.")

    with tab4:
        st.markdown("### Accuracy vs Tree Depth")
        depths = list(range(1, 11))
        accuracies = []
        X_tr = df_raw.drop(columns=['Irrigation_Need'])
        for col in X_tr.select_dtypes(include=['object', 'category']).columns:
            X_tr[col] = label_encoders[col].transform(X_tr[col].astype(str))
        
        X_t_train, X_t_test, y_t_train, y_t_test = train_test_split(
            X_tr, df_raw['Irrigation_Need'], test_size=0.2, random_state=42
        )

        for d in depths:
            dt = DecisionTreeClassifier(criterion="entropy", max_depth=d, random_state=42)
            dt.fit(X_t_train, y_t_train)
            accuracies.append(accuracy_score(y_t_test, dt.predict(X_t_test)))

        fig = px.line(
            x=depths, y=accuracies, markers=True,
            labels={"x": "Tree Depth", "y": "Accuracy"},
            title="ID3 Decision Tree Accuracy across Max Depths"
        )
        st.plotly_chart(fig, use_container_width=True)
        st.info("Model accuracy plateaus around depth 5. Pruning at max depth = 5 prevents overfitting on high-frequency noise.")

# ==========================================
# PAGE 3: GIS SATELLITE MAP
# ==========================================
if st.session_state.page == "GIS Satellite Map":
    st.subheader("GIS Satellite & Spatial Moisture Analytics")
    
    st.markdown("""
    This dashboard currently uses simulated regional coordinates because the dataset does not contain GPS fields.
    It identifies irrigation urgency, ranks fields for action, and estimates the water volume required for each zone.
    """)
    st.markdown("---")

    region_coords = {
        'North': {'lat': 28.6139, 'lon': 77.2090},
        'South': {'lat': 12.9716, 'lon': 77.5946},
        'East': {'lat': 22.5726, 'lon': 88.3639},
        'West': {'lat': 19.0760, 'lon': 72.8777},
        'Central': {'lat': 21.1458, 'lon': 79.0882}
    }

    map_df = df_raw.sample(min(300, len(df_raw)), random_state=42).copy()

    lats, lons = [], []
    random_generator = np.random.default_rng(42)
    for reg in map_df['Region']:
        base = region_coords.get(reg, {'lat': 20.5937, 'lon': 78.9629})
        lats.append(base['lat'] + random_generator.uniform(-1.5, 1.5))
        lons.append(base['lon'] + random_generator.uniform(-1.5, 1.5))

    map_df['latitude'] = lats
    map_df['longitude'] = lons
    priority_scores = {'High': 100, 'Medium': 60, 'Low': 20}
    water_targets = {'High': 35, 'Medium': 20, 'Low': 0}
    map_df['Priority_Score'] = map_df['Irrigation_Need'].map(priority_scores)
    map_df['Recommended_Action'] = map_df['Irrigation_Need'].map({
        'High': 'Irrigate immediately',
        'Medium': 'Schedule irrigation',
        'Low': 'Monitor only'
    })
    map_df['Recommended_Water_mm'] = map_df['Irrigation_Need'].map(water_targets)
    map_df['Estimated_Water_L'] = (
        map_df['Field_Area_hectare'] * map_df['Recommended_Water_mm'] * 10000
    )

    col_map1, col_map2 = st.columns([3, 1])

    with col_map2:
        st.markdown("#### GIS Map Filters")
        
        region_options = ["All Regions"] + list(map_df['Region'].unique())
        selected_region = st.selectbox("Select Region", options=region_options, index=0)
        
        need_options = ["All Urgency Levels", "High", "Medium", "Low"]
        selected_need = st.selectbox("Filter Irrigation Need", options=need_options, index=0)
        
        color_by = st.selectbox(
            "Color Code By",
            options=['Irrigation_Need', 'Priority_Score', 'Soil_Moisture', 'Crop_Type']
        )

    filtered_map_df = map_df.copy()
    if selected_region != "All Regions":
        filtered_map_df = filtered_map_df[filtered_map_df['Region'] == selected_region]
    if selected_need != "All Urgency Levels":
        filtered_map_df = filtered_map_df[filtered_map_df['Irrigation_Need'] == selected_need]

    with col_map1:
        color_map = {'High': '#ef5350', 'Medium': '#ffca28', 'Low': '#66bb6a'}

        if filtered_map_df.empty:
            st.warning("No fields match the selected filters.")
        else:
            fig_map = px.scatter_mapbox(
                filtered_map_df,
                lat="latitude",
                lon="longitude",
                color=color_by,
                size="Field_Area_hectare",
                hover_name="Crop_Type",
                hover_data={
                    "Soil_Moisture": ":.1f%",
                    "Rainfall_mm": ":.0f mm",
                    "Temperature_C": ":.1f °C",
                    "Irrigation_Need": True,
                    "Recommended_Action": True,
                    "Recommended_Water_mm": True,
                    "Estimated_Water_L": ":,.0f L",
                    "latitude": False,
                    "longitude": False
                },
                color_discrete_map=color_map if color_by == 'Irrigation_Need' else None,
                zoom=4.2,
                center={"lat": 20.5937, "lon": 78.9629},
                mapbox_style="open-street-map",
                title="Field Irrigation Priority Map"
            )

            fig_map.update_layout(height=550, margin={"r": 0, "t": 40, "l": 0, "b": 0})
            st.plotly_chart(fig_map, use_container_width=True)

    st.markdown("---")
    st.markdown("### Irrigation Action Summary")
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Active Fields Monitored", len(filtered_map_df))
    c2.metric("Avg Soil Moisture", f"{filtered_map_df['Soil_Moisture'].mean():.1f}%" if len(filtered_map_df) > 0 else "N/A")
    c3.metric("High Need Fields", len(filtered_map_df[filtered_map_df['Irrigation_Need'] == 'High']))
    c4.metric("Total Monitored Area", f"{filtered_map_df['Field_Area_hectare'].sum():.1f} Ha" if len(filtered_map_df) > 0 else "0 Ha")
    c5.metric("Estimated Water", f"{filtered_map_df['Estimated_Water_L'].sum():,.0f} L" if len(filtered_map_df) > 0 else "0 L")

    st.markdown("### Highest Priority Fields")
    priority_columns = [
        'Region', 'Crop_Type', 'Soil_Moisture', 'Irrigation_Need',
        'Priority_Score', 'Recommended_Action', 'Recommended_Water_mm',
        'Estimated_Water_L'
    ]
    priority_table = filtered_map_df.sort_values('Priority_Score', ascending=False)
    st.dataframe(
        priority_table[priority_columns].head(15),
        use_container_width=True,
        hide_index=True
    )
