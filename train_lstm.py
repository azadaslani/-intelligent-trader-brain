
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import EarlyStopping
import os

# بارگذاری داده
df = pd.read_csv("data/eurusd_real.csv")
df['date'] = pd.to_datetime(df['date'])
df = df.sort_values("date")
data = df['close'].values.reshape(-1, 1)

# نرمال‌سازی
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(data)

# ساخت X و y
def create_dataset(data, window_size=10):
    X, y = [], []
    for i in range(len(data) - window_size):
        X.append(data[i:i+window_size])
        y.append(data[i+window_size])
    return np.array(X), np.array(y)

X, y = create_dataset(scaled_data)
input_shape = (X.shape[1], X.shape[2])

# ساخت مدل LSTM
model = Sequential()
model.add(LSTM(64, input_shape=input_shape))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mse')

# آموزش مدل
model.fit(X, y, epochs=50, batch_size=8, verbose=1, callbacks=[EarlyStopping(patience=5)])

# ذخیره مدل
os.makedirs("models", exist_ok=True)
model.save("models/lstm_eurusd.h5")

# ذخیره نرمال‌ساز
import joblib
joblib.dump(scaler, "models/scaler.save")

print("✅ مدل با موفقیت آموزش داده شد و ذخیره شد.")
