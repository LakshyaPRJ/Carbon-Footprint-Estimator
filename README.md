To Run,
in terminal type: streamlit run main.py

🌱 Personal Carbon Footprint Estimator
A machine learning-powered web application that estimates an individual’s weekly carbon footprint based on lifestyle habits such as diet, transportation, electricity consumption, waste generation, and meals per day.
The project aims to increase awareness of personal environmental impact and demonstrate the application of ML models in sustainability-focused analytics.

Live Demo
🔗 App URL: [https://co2footprintestimator.streamlit.app(https://co2footprintestimator.streamlit.app/)

How It Works
User enters weekly lifestyle data via an interactive UI
Input data is preprocessed using one-hot encoding
A Random Forest Regressor predicts weekly CO₂ emissions

The app displays:
Estimated carbon footprint
Model performance metrics (MSE, R²)
Feature importance
Actual vs predicted emission comparison

Features:
Interactive Streamlit UI
Machine Learning-based prediction (Random Forest)
Synthetic dataset generation for training
Model evaluation (MSE & R²)
Feature importance visualization
Actual vs Predicted emissions plot
Cached model loading for faster performance

Tech Stack:
Frontend / UI: Streamlit
Backend / ML: Python, Scikit-learn
Data Handling: Pandas, NumPy
Visualization: Matplotlib, Seaborn
Model Persistence: Joblib

Machine Learning Details:
Model: Random Forest Regressor
Target Variable: Weekly CO₂ emissions (kg)

Evaluation Metrics:
Mean Squared Error (MSE)
R-squared (R²)

Training Data: Synthetic dataset simulating real-world lifestyle patterns

Model Performance
The model is evaluated on unseen test data
Feature importance highlights the most impactful lifestyle factors
High R² and low MSE indicate strong predictive performance

Installation & Local Setup
1️⃣ Clone the repository
git clone https://github.com/yourusername/carbon-footprint-estimator.git
cd carbon-footprint-estimator
2️⃣ Install dependencies
pip install -r requirements.txt
3️⃣ Run the app
streamlit run main.py

Use Cases:
Environmental awareness & education
Sustainability analytics projects
Machine learning portfolio demonstration
Streamlit deployment showcase

Future Improvements:
Replace synthetic data with real-world datasets
Add country-specific emission factors
User authentication & history tracking
Monthly / yearly footprint analysis
Recommendation system for emission reduction
