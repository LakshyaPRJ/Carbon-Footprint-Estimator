import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import joblib
import os

# Set wide layout and page name
st.set_page_config(layout="wide", page_title="Carbon Footprint Estimator")

# Title with better emoji
st.title("🌱 Personal Carbon Footprint Estimator")

# Custom CSS to make graphs smaller
st.markdown("""
<style>
.small-graph {
    width: 100% !important;
    max-width: 600px;
    margin: 0 auto;
}
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_or_train_model():
    try:
        model = joblib.load('carbon_model.pkl')
        X_test = joblib.load('X_test.pkl')
        y_test = joblib.load('y_test.pkl')
        return model, X_test, y_test
    except FileNotFoundError:
        pass

    # Generate synthetic data
    np.random.seed(42)
    n_samples = 1000
    data = {
        'diet': np.random.choice(['Omnivore', 'Vegetarian', 'Vegan', 'Pescatarian'], n_samples),
        'transport_km': np.random.uniform(0, 100, n_samples),
        'electricity_kwh': np.random.uniform(0, 1000, n_samples),
        'waste_kg': np.random.uniform(0, 20, n_samples),
        'meals_per_day': np.random.randint(1, 5, n_samples)
    }
    df = pd.DataFrame(data)

    df['co2_kg_week'] = (
            (df['transport_km'] * 0.14 * 7) +
            (df['electricity_kwh'] * 0.82 / 4) +
            (df['meals_per_day'] * 1.25 * 7) +
            (df['waste_kg'] * 0.1)
    )

    X = pd.get_dummies(df.drop('co2_kg_week', axis=1))
    y = df['co2_kg_week']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    os.makedirs('data', exist_ok=True)
    joblib.dump(model, 'carbon_model.pkl')
    joblib.dump(X_test, 'X_test.pkl')
    joblib.dump(y_test, 'y_test.pkl')

    return model, X_test, y_test


try:
    model, X_test, y_test = load_or_train_model()
    predictions = model.predict(X_test)
    mse = mean_squared_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)
except Exception as e:
    st.error(f"Error initializing model: {str(e)}")
    st.stop()

# User inputs
st.subheader("📊 Enter Your Weekly Data")
col1, col2 = st.columns(2)

with col1:
    diet = st.selectbox("Diet Type", ['Omnivore', 'Vegetarian', 'Vegan', 'Pescatarian'])
    transport_km = st.slider("Daily commute distance (km)", 0.0, 100.0, 10.0)
    electricity_kwh = st.slider("Weekly electricity consumption (kWh)", 0.0, 250.0, 50.0)

with col2:
    waste_kg = st.slider("Waste generated per week (kg)", 0.0, 20.0, 5.0)
    meals_per_day = st.slider("Meals per day", 1, 5, 3)

# Prepare input for prediction
input_data = pd.DataFrame({
    'diet': [diet],
    'transport_km': [transport_km],
    'electricity_kwh': [electricity_kwh],
    'waste_kg': [waste_kg],
    'meals_per_day': [meals_per_day]
})

input_processed = pd.get_dummies(input_data)
train_columns = model.feature_names_in_
input_processed = input_processed.reindex(columns=train_columns, fill_value=0)

if st.button("Calculate Carbon Footprint"):
    prediction = model.predict(input_processed)[0]

    st.subheader("📊 Your Carbon Footprint")
    col3, col4 = st.columns(2)

    with col3:
        st.metric(label="Estimated CO₂ Emissions", value=f"{prediction:.2f} kg CO₂/week")
        st.info("""
        **Context:**
        - Average person emits about 100-150 kg CO₂/week
        - Sustainable target is under 50 kg CO₂/week
        """)

    with col4:
        st.subheader("Model Performance")
        st.metric(label="Mean Squared Error (MSE)", value=f"{mse:.2f}")
        st.metric(label="R-squared (R²)", value=f"{r2:.2f}")
        st.caption("Lower MSE and higher R² indicate better model performance")

    # Visualization section with smaller graphs
    st.subheader("📈 Emissions Analysis")

    # Create a container for graphs with constrained width
    graph_container = st.container()

    with graph_container:
        # Feature importance plot (smaller)
        fig1, ax1 = plt.subplots(figsize=(8, 4))
        features = model.feature_names_in_
        importances = model.feature_importances_
        sns.barplot(x=importances, y=features, ax=ax1)
        ax1.set_title('Feature Importance in Emission Calculation')
        ax1.set_xlabel('Importance Score')
        ax1.set_ylabel('Lifestyle Factors')
        st.pyplot(fig1, bbox_inches='tight')

        # Actual vs Predicted plot (smaller)
        fig2, ax2 = plt.subplots(figsize=(8, 4))
        ax2.scatter(y_test, predictions, alpha=0.5, s=30)  # Smaller point size
        ax2.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--')
        ax2.set_xlabel('Actual Emissions (kg/week)')
        ax2.set_ylabel('Predicted Emissions (kg/week)')
        ax2.set_title('Actual vs Predicted Emissions')
        st.pyplot(fig2, bbox_inches='tight')

# Add GitHub link
st.markdown("""
---
🔗 [View on GitHub](https://github.com/LakshyaPRJ/Carbon-Footprint-Estimator)
""")
