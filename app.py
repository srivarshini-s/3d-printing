import pandas as pd
import numpy as np
import streamlit as st
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# Load data
df = pd.read_csv(r"C:\Users\S SRIVARSHINI\Downloads\Measured_data.csv", encoding='latin1')

# Define features and target variables
X = df[['Printing speed(mm/s)', 'Layer thickness(mm)', 'Nozzle temperature(°C)']]
targets = {
    'SRTB': df['Average surface roughness SRTB(Ra µm)'],
    'SRS': df['Average surface roughness SRS(Ra µm)'],
    'PrintingTime': df['Printing time(min)']
}
units = {
    'SRTB': 'µm',
    'SRS': 'µm',
    'PrintingTime': 'min'
}

# Function to train and evaluate linear regression model
def train_linear_regression(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    return model, mse, r2

# Train models
models = {}
for target_name, y in targets.items():
    model, mse, r2 = train_linear_regression(X, y)
    models[target_name] = model

# Streamlit app
st.set_page_config(page_title="3D Printing Parameter Predictor", page_icon="🖨️", layout="centered")

st.markdown(
    """
    <style>
    .main {
        background-color: #f5f5f5;
        color: #000000;
    }
    .sidebar .sidebar-content {
        background: #dcdcdc;
    }
    .css-1d391kg {
        color: #4CAF50;
    }
    .css-1avcm0n {
        color: #FFFFFF;
        background-color: #4CAF50;
        border-color: #4CAF50;
    }
    .css-1aumxhk {
        background-color: #FFFFFF;
        color: #4CAF50;
        border-color: #4CAF50;
    }
    .css-1aumxhk:hover {
        background-color: #4CAF50;
        color: #FFFFFF;
    }
    .css-1aumxhk:focus {
        background-color: #4CAF50;
        color: #FFFFFF;
    }
    .css-1aumxhk:active {
        background-color: #4CAF50;
        color: #FFFFFF;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.title("🖨️ 3D Printing Parameter Predictor")

st.sidebar.header("Input Parameters")
printing_speed = st.sidebar.slider('Printing speed (mm/s)', float(X['Printing speed(mm/s)'].min()), float(X['Printing speed(mm/s)'].max()), float(X['Printing speed(mm/s)'].mean()))
layer_thickness = st.sidebar.slider('Layer thickness (mm)', float(X['Layer thickness(mm)'].min()), float(X['Layer thickness(mm)'].max()), float(X['Layer thickness(mm)'].mean()))
nozzle_temperature = st.sidebar.slider('Nozzle temperature (°C)', float(X['Nozzle temperature(°C)'].min()), float(X['Nozzle temperature(°C)'].max()), float(X['Nozzle temperature(°C)'].mean()))

input_data = np.array([[printing_speed, layer_thickness, nozzle_temperature]])

# Predictions
if st.sidebar.button('Predict'):
    predictions = {}
    for target_name, model in models.items():
        prediction = model.predict(input_data)
        predictions[target_name] = prediction[0]

    st.subheader("Predicted Values")
    for target_name, prediction in predictions.items():
        unit = units[target_name]
        st.write(f"<div style='color: #4CAF50; font-size: 24px;'><strong>{target_name}:</strong> {prediction:.2f} {unit}</div>", unsafe_allow_html=True)

# Run the app with: streamlit run your_script_name.py

