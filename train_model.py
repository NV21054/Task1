import tensorflow as tf
from ucimlrepo import fetch_ucirepo

# Fetch dataset
wine_quality = fetch_ucirepo(id=186)
X = wine_quality.data.features
y = wine_quality.data.targets

# Normalize features
X = (X - X.mean()) / X.std()

# Define the model
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(X.shape[1],)),
    tf.keras.layers.Dense(1)
])

# Compile the model
model.compile(optimizer='sgd', loss='mse')

# Train the model
model.fit(X, y, epochs=1000, verbose=1)

# Save the trained model
model.save('model.h5')
