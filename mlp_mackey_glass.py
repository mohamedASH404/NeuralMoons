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