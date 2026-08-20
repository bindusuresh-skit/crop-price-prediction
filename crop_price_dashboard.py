import streamlit as st
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error

from prophet import Prophet

# -----------------------------
# Load Dataset
# -----------------------------
@st.cache_data
def load_data():
    return pd.read_csv("crop_price_dataset.csv")

df = load_data()

# -----------------------------
# Title
# -----------------------------
st.title("🌾 Smart Crop Price Prediction Dashboard")

st.write(
    "Predict crop prices and future trends using Machine Learning."
)

# -----------------------------
# View Dataset
# -----------------------------
with st.expander("📊 View Dataset"):
    st.dataframe(df)

# -----------------------------
# Remove Missing Values
# -----------------------------
df = df.dropna()

# -----------------------------
# Label Encoding
# -----------------------------
encoders = {}

categorical_columns = [
    'month',
    'commodity_name',
    'state_name',
    'district_name'
]

for col in categorical_columns:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    encoders[col] = le

# -----------------------------
# Features & Target
# -----------------------------
X = df[
    [
        'month',
        'commodity_name',
        'state_name',
        'district_name'
    ]
]

y = df['avg_modal_price']

# -----------------------------
# Split Data
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# -----------------------------
# Train Model
# -----------------------------
model = RandomForestRegressor(
    n_estimators=100,
    random_state=42
)

model.fit(X_train, y_train)

# -----------------------------
# Prediction Section
# -----------------------------
st.header("🔍 Predict Crop Price")

month = st.selectbox(
    "Select Month",
    encoders['month'].classes_
)

commodity = st.selectbox(
    "Select Commodity",
    encoders['commodity_name'].classes_
)

state = st.selectbox(
    "Select State",
    encoders['state_name'].classes_
)

district = st.selectbox(
    "Select District",
    encoders['district_name'].classes_
)

# -----------------------------
# Prediction Button
# -----------------------------
if st.button("💰 Predict Price"):

    input_data = pd.DataFrame({
        'month': [
            encoders['month'].transform([month])[0]
        ],
        'commodity_name': [
            encoders['commodity_name'].transform([commodity])[0]
        ],
        'state_name': [
            encoders['state_name'].transform([state])[0]
        ],
        'district_name': [
            encoders['district_name'].transform([district])[0]
        ]
    })

    prediction = model.predict(input_data)[0]

    st.success(
        f"Predicted Crop Price: ₹{prediction:.2f} per quintal"
    )

# -----------------------------
# Accuracy Section
# -----------------------------
st.header("📈 Model Accuracy")

predictions = model.predict(X_test)

r2 = r2_score(y_test, predictions)
mae = mean_absolute_error(y_test, predictions)

st.write(f"R² Score: {r2:.2f}")
st.write(f"Mean Absolute Error: ₹{mae:.2f}")

# -----------------------------
# Forecasting
# -----------------------------
st.header("📅 Future Crop Price Forecast")

forecast_df = load_data()

# Create Date Column
forecast_df['ds'] = pd.to_datetime(
    forecast_df['year'].astype(str)
    + '-'
    + forecast_df['month']
)

forecast_df['y'] = forecast_df['avg_modal_price']

# Prophet Model
prophet_model = Prophet()

prophet_model.fit(
    forecast_df[['ds', 'y']]
)

# Future Dates
future = prophet_model.make_future_dataframe(
    periods=12,
    freq='ME'
)

forecast = prophet_model.predict(future)

# -----------------------------
# Forecast Graph
# -----------------------------
st.subheader("📊 Future Price Trend")

st.line_chart(
    forecast.set_index('ds')['yhat']
)

# -----------------------------
# Forecast Table
# -----------------------------
with st.expander("📅 Forecast Data"):
    st.dataframe(
        forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(12)
    )