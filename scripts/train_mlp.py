import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# Generate the chaotic Mackey-Glass time series (a non-trivial prediction task)
def mackey_glass(tmax, tau=17, beta=0.2, gamma=0.1, n=10, dt=1):
    t = np.arange(0, tmax, dt)
    x = np.zeros_like(t, dtype=np.float32)
    x[0] = 1.2
    for i in range(1, len(t)):
        if i - tau >= 0:
            x[i] = x[i-1] + dt * (beta * x[i - tau] / (1 + x[i - tau]**n) - gamma * x[i-1])
        else:
            x[i] = x[i-1]
    return x

# Generate time series data
data = mackey_glass(3000)
window_size = 50
forecast_horizon = 1

# Create supervised dataset with sliding windows
def create_dataset(series, window_size, horizon):
    X, y = [], []
    for i in range(len(series) - window_size - horizon):
        X.append(series[i:i+window_size])
        y.append(series[i+window_size+horizon-1])
    return np.array(X), np.array(y)

X, y = create_dataset(data, window_size, forecast_horizon)

# Normalize features and target
scaler_X = StandardScaler()
scaler_y = StandardScaler()
X = scaler_X.fit_transform(X)
y = scaler_y.fit_transform(y.reshape(-1, 1)).flatten()

# Split the data into training and testing sets (no shuffling to preserve temporal order)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# Define a deep MLP model
model = models.Sequential([
    layers.Input(shape=(window_size,)),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.3),
    layers.Dense(256, activation='relu'),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.2),
    layers.Dense(1)
])

# Compile the model
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss='mse')

# Train the model
history = model.fit(X_train, y_train, epochs=100, batch_size=64, validation_split=0.1, verbose=1)

# Predict on test data
y_pred = model.predict(X_test)

# Inverse transform predictions and targets
y_test_inv = scaler_y.inverse_transform(y_test.reshape(-1, 1)).flatten()
y_pred_inv = scaler_y.inverse_transform(y_pred).flatten()

# Plot the results
plt.figure(figsize=(12, 5))
plt.plot(y_test_inv, label="True")
plt.plot(y_pred_inv, label="Predicted")
plt.legend()
plt.title("MLP Forecasting on Mackey-Glass Time Series")
plt.xlabel("Time")
plt.ylabel("Value")
plt.show()