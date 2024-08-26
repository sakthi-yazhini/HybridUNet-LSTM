import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, LayerNormalization, Dropout, Flatten
from tensorflow.keras.models import Model, load_model
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, matthews_corrcoef, cohen_kappa_score
from tensorflow.keras.utils import normalize, get_custom_objects
import tensorflow.keras.backend as K
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, precision_score, recall_score, f1_score, mean_squared_error

# Set seeds for reproducibility
np.random.seed(0)
tf.random.set_seed(0)

# Load dataset and labels
dataset = np.load('dataset.npy')
label = np.load('labels.npy')

print('Dataset: ', len(dataset))
print('Label: ', len(label))

#Data Splitting
X_train, X_test, y_train, y_test = train_test_split(dataset, label, test_size=0.20, random_state=2523)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25, random_state=2523)

# Normalize the Data
X_train = normalize(X_train, axis=1)
X_val = normalize(X_val, axis=1)
X_test = normalize(X_test, axis=1)

# Define transformer blocks and attention mechanisms
class TransformerBlock(tf.keras.layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1):
        super(TransformerBlock, self).__init__()
        self.att = tf.keras.layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.ffn = tf.keras.Sequential(
            [Dense(ff_dim, activation="relu"), Dense(embed_dim),]
        )
        self.layernorm1 = LayerNormalization(epsilon=1e-6)
        self.layernorm2 = LayerNormalization(epsilon=1e-6)
        self.dropout1 = Dropout(rate)
        self.dropout2 = Dropout(rate)

    def call(self, inputs, training):
        attn_output = self.att(inputs, inputs)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)

class TokenAndPositionEmbedding(tf.keras.layers.Layer):
    def __init__(self, maxlen, embed_dim):
        super(TokenAndPositionEmbedding, self).__init__()
        self.maxlen = maxlen
        self.embed_dim = embed_dim
        self.pos_emb = tf.keras.layers.Embedding(input_dim=maxlen, output_dim=embed_dim)
        self.dense = tf.keras.layers.Dense(embed_dim)

    def call(self, x):
        positions = tf.range(start=0, limit=self.maxlen, delta=1)
        positions = self.pos_emb(positions)
        x = self.dense(x)
        return x + positions

# Define the model
embed_dim = 32  
num_heads = 1  
ff_dim = 16  
INPUT_SIZE = 64 

def build_transformer_model(input_shape):
    inputs = Input(shape=input_shape)
    embedding_layer = TokenAndPositionEmbedding(INPUT_SIZE * INPUT_SIZE, embed_dim)
    x = embedding_layer(inputs)
    transformer_block = TransformerBlock(embed_dim, num_heads, ff_dim)
    x = transformer_block(x)
    x = Flatten()(x)
    x = Dense(64, activation="relu")(x)
    x = Dropout(0.1)(x)
    outputs = Dense(1, activation="sigmoid")(x)
    model = Model(inputs=inputs, outputs=outputs)
    return model

# Reshape data for transformer input
X_train_transformer = X_train.reshape(-1, INPUT_SIZE * INPUT_SIZE, 3)
X_val_transformer = X_val.reshape(-1, INPUT_SIZE * INPUT_SIZE, 3)
X_test_transformer = X_test.reshape(-1, INPUT_SIZE * INPUT_SIZE, 3)

model = build_transformer_model((INPUT_SIZE * INPUT_SIZE, 3))

# Compile the model using binary cross-entropy loss
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4), loss='binary_crossentropy', metrics=['accuracy'])

model.summary()

# Train the model
history = model.fit(X_train_transformer, y_train, batch_size=64, epochs=10, validation_data=(X_val_transformer, y_val), shuffle=False)

# Save the model
model.save('transformer_model.h5')

# Register custom layers in custom objects
get_custom_objects().update({
    'TokenAndPositionEmbedding': TokenAndPositionEmbedding,
    'TransformerBlock': TransformerBlock
})

# Load the model using custom objects
model = load_model('transformer_model.h5')

# Evaluate the model
scores = model.evaluate(X_test_transformer, y_test, verbose=0)
print('Test loss:', scores[0])
print('Test accuracy:', scores[1])

# Get predictions
y_pred = model.predict(X_test_transformer)
y_pred_binary = (y_pred > 0.5).astype(int)
cm = confusion_matrix(y_test, y_pred_binary)
cr = classification_report(y_test, y_pred_binary, output_dict=True)
acc = accuracy_score(y_test, y_pred_binary)
precision = precision_score(y_test, y_pred_binary)
recall = recall_score(y_test, y_pred_binary)
f1 = f1_score(y_test, y_pred_binary)
mse = mean_squared_error(y_test, y_pred)
psnr = 20 * np.log10(np.max(y_test)) - 10 * np.log10(mse)
roc_auc = roc_auc_score(y_test, y_pred)
print('Accuracy:', acc)
print('Precision:', precision)
print('Recall:', recall)
print('F1 score:', f1)
print("ROC-AUC:", roc_auc)
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

