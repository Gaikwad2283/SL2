import tensorflow as tf
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import mean_squared_error
# Load the breast cancer dataset
df = load_breast_cancer()
X_train, X_test, y_train, y_test = train_test_split(df.data, df.target, test_size=0.20, random_state=42)

# Scale the data
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
print("shape"+str(X_train.shape))
# Build the model
model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(1, activation='sigmoid', input_shape=(X_train.shape[1],))
])

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, epochs=5)

# Predict on test set
y_pred = model.predict(X_test)
print("Accuracy_Error:",mean_squared_error(y_test,y_pred))
# Evaluate the model
test_loss, test_accuracy = model.evaluate(X_test, y_test)
print("Accuracy:", test_accuracy)
