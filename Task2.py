# Import necessary libraries
import numpy as np
import pandas as pd
import librosa
import librosa.display
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras import layers, models, callbacks
from sklearn.utils import class_weight

# Load your emotion-labeled audio data into a Pandas DataFrame
# Replace 'your_audio_data.csv' with the actual file path or URL
data = pd.read_csv('your_audio_data.csv')

# Extract features from audio files using librosa library
def extract_features(file_path):
    audio, _ = librosa.load(file_path, res_type='kaiser_fast', duration=3)
    mfccs = librosa.feature.mfcc(y=audio, sr=librosa.get_sr(audio), n_mfcc=13)
    return np.mean(mfccs.T, axis=0)

# Apply feature extraction to all audio files in the dataset
data['features'] = data['audio_file_path'].apply(extract_features)

# Convert emotion labels to numerical values
label_encoder = LabelEncoder()
data['emotion_label'] = label_encoder.fit_transform(data['emotion'])

# Split the data into training and testing sets
X = np.array(data['features'].tolist())
y = np.array(data['emotion_label'])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Calculate class weights to handle imbalanced classes
class_weights = class_weight.compute_class_weight('balanced', np.unique(y_train), y_train)

# Build a simple deep learning model using Keras
model = models.Sequential([
    layers.Dense(128, activation='relu', input_shape=(X_train.shape[1],)),
    layers.Dropout(0.5),
    layers.Dense(64, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(len(label_encoder.classes_), activation='softmax')
])

# Compile the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Define early stopping to prevent overfitting
early_stopping = callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

# Train the model
history = model.fit(X_train, y_train, epochs=20, batch_size=32, validation_split=0.2, 
                    callbacks=[early_stopping], class_weight=dict(enumerate(class_weights)))

# Evaluate the model on the test set
loss, accuracy = model.evaluate(X_test, y_test)
print(f'Test Accuracy: {accuracy}')

# Save the model for future use
model.save('emotion_recognition_model.h5')


