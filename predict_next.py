
import pandas as pd
import numpy as np
import joblib
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler

# بارگذاری داده
df = pd.read_csv("data/eurusd_real.csv")
df['date'] = pd.to_datetime(df['date'])
df = df.sort_values("date")
data = df['close'].values.reshape(-1, 1)

# بارگذاری نرمال‌ساز و مدل
scaler = joblib.load("models/scaler.save")
model = load_model("models/lstm_eurusd.h5")

# نرمال‌سازی داده
scaled_data = scaler.transform(data)

# انتخاب آخرین n مقدار برای پیش‌بینی
window_size = 10
if len(scaled_data) < window_size:
    raise ValueError("Not enough data for prediction.")

X_pred = np.array([scaled_data[-window_size:]])
predicted_scaled = model.predict(X_pred)
predicted_price = scaler.inverse_transform(predicted_scaled)[0][0]

print(f"📈 آخرین قیمت واقعی: {data[-1][0]:.5f}")
print(f"🤖 پیش‌بینی قیمت بعدی توسط LSTM: {predicted_price:.5f}")
