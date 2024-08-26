import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Dropout, Flatten, Dense
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.utils import normalize
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, precision_score, recall_score, f1_score, mean_squared_error
from sklearn.metrics import roc_curve

# Load dataset and labels
dataset = np.load('dataset.npy')
labels = np.load('labels.npy')

print('Dataset shape:', dataset.shape)
print('Labels shape:', labels.shape)

# Define constants
INPUT_SIZE = 64
NUM_CLASSES = 1  # Binary classification

# Data Splitting
X_train, X_test, y_train, y_test = train_test_split(dataset, labels, test_size=0.20, random_state=2523)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25, random_state=2523)

# Normalize the Data
X_train = normalize(X_train, axis=1)
X_val = normalize(X_val, axis=1)
X_test = normalize(X_test, axis=1)

# Define the CNN model
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(INPUT_SIZE, INPUT_SIZE, 3)),
    MaxPooling2D(pool_size=(2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(NUM_CLASSES, activation='sigmoid')
])

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.summary()

# Train the model 
history = model.fit(
    X_train, y_train,
    batch_size=32,
    epochs=10,
    validation_data=(X_val, y_val),
    shuffle=True
)

# Save the model
model.save('cnn_model.h5')

# Load the model 
model = load_model('cnn_model.h5')

# Evaluate the model on the test set
scores = model.evaluate(X_test, y_test, verbose=0)
print('Test loss:', scores[0])
print('Test accuracy:', scores[1])

# Get predictions on the test set
y_pred = model.predict(X_test)
y_pred_binary = (y_pred > 0.5).astype(int)

# Calculate metrics
cm = confusion_matrix(y_test, y_pred_binary)
cr = classification_report(y_test, y_pred_binary, output_dict=True)
acc = accuracy_score(y_test, y_pred_binary)
precision = precision_score(y_test, y_pred_binary)
recall = recall_score(y_test, y_pred_binary)
f1 = f1_score(y_test, y_pred_binary)
mse = mean_squared_error(y_test, y_pred)
max_pixel_value = np.max(y_test)  
psnr = 20 * np.log10(max_pixel_value) - 10 * np.log10(mse)
TP = ((y_pred_binary == 1) & (y_test == 1)).sum()
FP = ((y_pred_binary == 1) & (y_test == 0)).sum()
FN = ((y_pred_binary == 0) & (y_test == 1)).sum()
TN = ((y_pred_binary == 0) & (y_test == 0)).sum()
sensitivity = TP / (TP + FN) if (TP + FN) > 0 else 0
specificity = TN / (TN + FP) if (TN + FP) > 0 else 0
gmean = np.sqrt(sensitivity * specificity)

# Output results
print('Accuracy:', acc)
print('Precision:', precision)
print('Recall:', recall)
print('F1 score:', f1)
print("Mean Squared Error:", mse)
print("Peak Signal-to-Noise Ratio:", psnr)
print("Sensitivity:", sensitivity)
print("Specificity:", specificity)
print("G-Mean:", gmean)

