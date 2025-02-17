import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Load dataset
base_file_name = 'C:/Users/jawad/Downloads/Python-EPSC-NSFA-Pipeline/'
df = pd.read_pickle(f'{base_file_name}/simulation_optimization/training_df_v6_100normal.pkl')

# Extract features (X) and targets (y)
X = df[['peak_amplitudes', 'rise_times', 'decay_taus']]
y = df[['mean', 'std_dev', 'glutamate_scale']]

X_stack = np.hstack([
    np.sort(np.array(X["peak_amplitudes"].tolist()), axis=1),
    np.sort(np.array(X["rise_times"].tolist()), axis=1),
    np.sort(np.array(X["decay_taus"].tolist()), axis=1)
])


y_array = np.array(y.values, dtype=np.float32)

# Normalize features (helps training stability)
X_stack = (X_stack - np.mean(X_stack, axis=0)) / np.std(X_stack, axis=0)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_stack, y_array, test_size=0.2, random_state=1)

# Build the TensorFlow/Keras Model
model = keras.Sequential([
    layers.Dense(128, activation='relu', input_shape=(X_train.shape[1],)),
    layers.Dense(64, activation='relu'),
    layers.Dense(32, activation='relu'),
    layers.Dense(y_train.shape[1])  # Output layer (3 values: mean, std_dev, glutamate_scale)
])

# Compile the model
model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001), loss='mse')

# Train the model
epochs = 5000
history = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=epochs, batch_size=32, verbose=1)

# Predict on test data
y_pred = model.predict(X_test)

# Compute MSE
mse = mean_squared_error(y_test, y_pred)
print("Mean Squared Error on Test Data:", mse)
print(f"Example Simulated: {y_test[0]} and Example Predicted: {y_pred[0]}")

model.save("little_model.keras")