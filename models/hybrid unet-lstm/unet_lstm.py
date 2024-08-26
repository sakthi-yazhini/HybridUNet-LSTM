import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv3D, MaxPooling3D, Flatten, Dense, BatchNormalization, RepeatVector, LSTM
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import backend as K
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, precision_score, recall_score, f1_score, roc_curve, mean_squared_error

# Define the custom attention layer for inter-video fusion
class Attention(tf.keras.layers.Layer):
    def __init__(self):
        super(Attention, self).__init__()

    def call(self, inputs, **kwargs):
        query, key, value = inputs
        # Compute similarity matrix
        similarity = tf.matmul(query, key, transpose_b=True)
        attention_weights = tf.nn.softmax(similarity, axis=-1)
        # Compute weighted sum
        context = tf.matmul(attention_weights, value)
        return context

# Define the CVA-Net model
def CVA_Net(input_shape):
    # Inputs
    input_frames = Input(shape=input_shape)  # (sequence_length, height, width, channels)
    shuffled_input_frames = Input(shape=input_shape)  # For shuffled video frames

    # Feature extraction for original frames
    x = Conv3D(32, kernel_size=(3, 3, 3), activation='relu')(input_frames)
    x = MaxPooling3D(pool_size=(2, 2, 2))(x)
    x = BatchNormalization()(x)

    x = Conv3D(64, kernel_size=(3, 3, 3), activation='relu')(x)
    x = MaxPooling3D(pool_size=(2, 2, 2))(x)
    x = BatchNormalization()(x)
    x_local = Flatten()(x)

    # Feature extraction for shuffled frames
    y = Conv3D(32, kernel_size=(3, 3, 3), activation='relu')(shuffled_input_frames)
    y = MaxPooling3D(pool_size=(2, 2, 2))(y)
    y = BatchNormalization()(y)

    y = Conv3D(64, kernel_size=(3, 3, 3), activation='relu')(y)
    y = MaxPooling3D(pool_size=(2, 2, 2))(y)
    y = BatchNormalization()(y)
    y_global = Flatten()(y)

    # Inter-video Fusion
    attention_layer = Attention()
    fused_features = attention_layer([x_local, y_global, y_global])

    # Intra-video Fusion
    x_fused = Dense(64, activation='relu')(fused_features)
    x_fused = RepeatVector(input_shape[0])(x_fused)
    x_fused = LSTM(64, activation='relu', return_sequences=False)(x_fused)

    # Output layers
    classification_output = Dense(1, activation='sigmoid')(x_fused)
    
    # Define and compile model
    model = Model(inputs=[input_frames, shuffled_input_frames], outputs=classification_output)
    model.compile(optimizer=Adam(learning_rate=1e-4), loss='binary_crossentropy', metrics=['accuracy'])
    
    return model

# Load dataset and labels
dataset = np.load('dataset.npy')  
labels = np.load('labels.npy')   

print('Dataset: ', len(dataset))
print('Labels: ', len(labels))

# Define constants
INPUT_SIZE = 64
sequence_length = 10

# Convert dataset to sequences of frames
num_sequences = len(dataset) // sequence_length
dataset_sequences = np.array([dataset[i * sequence_length:(i + 1) * sequence_length] for i in range(num_sequences)])
label_sequences = np.array([labels[i * sequence_length:(i + 1) * sequence_length][0] for i in range(num_sequences)])

# Data Splitting
X_train, X_test, y_train, y_test = train_test_split(dataset_sequences, label_sequences, test_size=0.20, random_state=2523)
X_train = X_train.astype('float32') / 255.0
X_test = X_test.astype('float32') / 255.0

# Generate shuffled frames for inter-video fusion
X_train_shuffled = np.array([np.random.permutation(seq) for seq in X_train])
X_test_shuffled = np.array([np.random.permutation(seq) for seq in X_test])

# Define and compile the model
input_shape = (sequence_length, INPUT_SIZE, INPUT_SIZE, 3)
model = CVA_Net(input_shape)

# Model summary
model.summary()

# Train the model
history = model.fit(
    [X_train, X_train_shuffled], 
    y_train, 
    batch_size=32, 
    epochs=10, 
    validation_data=([X_test, X_test_shuffled], y_test),
    shuffle=True
)

# Save the Model
model.save('unet_lstm_model.h5')

# Custom object scope for loading the model
from tensorflow.keras.utils import get_custom_objects

with tf.keras.utils.custom_object_scope({'Attention': Attention}):
    # Load Model
    model = tf.keras.models.load_model('unet_lstm_model.h5')

# Evaluate the model
scores = model.evaluate([X_test, X_test_shuffled], y_test, verbose=0)
print('Test loss:', scores[0])
print('Test accuracy:', scores[1])

# Get predictions
y_pred = model.predict([X_test, X_test_shuffled])
y_pred_binary = (y_pred > 0.5).astype(int)

# Evaluation metrics
cm = confusion_matrix(y_test, y_pred_binary)
cr = classification_report(y_test, y_pred_binary, output_dict=True)
acc = accuracy_score(y_test, y_pred_binary)
precision = precision_score(y_test, y_pred_binary)
recall = recall_score(y_test, y_pred_binary)
f1 = f1_score(y_test, y_pred_binary)
mse = mean_squared_error(y_test, y_pred)
psnr = 20 * np.log10(np.max(y_test)) - 10 * np.log10(mse)
print("Confusion matrix:")
print(cm)
print('Accuracy:', acc)
print('Precision:', precision)
print('Recall:', recall)
print('F1 score:', f1)
print("Mean Squared Error: ", mse)
print("Peak Signal-to-Noise Ratio: ", psnr)

TP = ((y_pred_binary == 1) & (y_test == 1)).sum()
FP = ((y_pred_binary == 1) & (y_test == 0)).sum()
FN = ((y_pred_binary == 0) & (y_test == 1)).sum()
TN = ((y_pred_binary == 0) & (y_test == 0)).sum()
sensitivity = TP / (TP + FN)
specificity = TN / (TN + FP)
gmean = np.sqrt(sensitivity * specificity)
print("Sensitivity:", sensitivity)
print("Specificity:", specificity)
print("G-Mean:", gmean)

