import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import Callback
from sklearn.metrics import f1_score
import matplotlib.pyplot as plt
import numpy as np
from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename
import os
from PIL import Image

# Define global variables for class names and upload folder
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
UPLOAD_FOLDER = 'uploads'

# Helper Functions
def load_and_preprocess_data():
    """Load and preprocess the CIFAR-10 dataset."""
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
    x_train, x_test = x_train / 255.0, x_test / 255.0
    y_train_one_hot = tf.keras.utils.to_categorical(y_train, 10)
    y_test_one_hot = tf.keras.utils.to_categorical(y_test, 10)
    return (x_train, y_train, y_train_one_hot), (x_test, y_test, y_test_one_hot)

def build_model():
    """Build a smaller CNN model for CIFAR-10."""
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(10, activation='softmax')
    ])
    return model

class F1Metric(Callback):
    """Custom callback to compute F1 scores at the end of each epoch."""
    def __init__(self, x_val, y_val_one_hot):
        super().__init__()
        self.x_val = x_val
        self.y_val = np.argmax(y_val_one_hot, axis=1)  # Convert one-hot to class labels
        self.f1_scores = []

    def on_epoch_end(self, epoch, logs=None):
        # Predict on validation data
        y_pred = self.model.predict(self.x_val)
        y_pred_classes = np.argmax(y_pred, axis=1)  # Get predicted class labels
        
        # Compute F1 score
        f1 = f1_score(self.y_val, y_pred_classes, average='macro')
        self.f1_scores.append(f1)
        print(f"Epoch {epoch + 1}: F1 Score = {f1:.4f}")

def plot_metrics(history, f1_scores):
    """Plot accuracy and loss, and a separate scatter plot for F1 scores."""
    # Accuracy and Loss
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Training Accuracy', marker='o')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy', marker='o')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid()

    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Training Loss', marker='o')
    plt.plot(history.history['val_loss'], label='Validation Loss', marker='o')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid()

    plt.tight_layout()
    plt.show()

    # F1 Score Scatter Plot in a Separate Window
    plt.figure()
    epochs = range(1, len(f1_scores) + 1)
    plt.scatter(epochs, f1_scores, color='blue', label='F1 Score')
    plt.title('F1 Score Scatter Plot')
    plt.xlabel('Epoch')
    plt.ylabel('F1 Score')
    plt.xticks(epochs)
    plt.grid()
    plt.legend()
    plt.show()

def train_model(model, x_train, y_train_one_hot, x_test, y_test_one_hot):
    """Train the model with F1 metric tracking."""
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    # Initialize F1Metric callback
    f1_callback = F1Metric(x_test, y_test_one_hot)

    # Train the model
    history = model.fit(x_train, y_train_one_hot, epochs=10, batch_size=64,
                        validation_data=(x_test, y_test_one_hot), callbacks=[f1_callback])

    # Save the model
    model.save('cifar10_simplified_model.keras')
    print("Simplified model saved to cifar10_simplified_model.keras")

    # Plot metrics
    plot_metrics(history, f1_callback.f1_scores)

    return history

# Flask Application
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg'}

def allowed_file(filename):
    """Check if the uploaded file is allowed."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

@app.route('/')
def index():
    """Render the upload form."""
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """Handle image upload and prediction."""
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        # Preprocess the image
        image = Image.open(filepath).resize((32, 32))
        image = np.array(image) / 255.0
        image = np.expand_dims(image, axis=0)

        # Predict
        prediction = model.predict(image)
        predicted_class = np.argmax(prediction, axis=1)[0]
        predicted_label = class_names[predicted_class]

        return jsonify({'predicted_class': predicted_label, 'confidence': float(prediction[0][predicted_class])})

    return jsonify({'error': 'Invalid file format'}), 400

if __name__ == '__main__':
    # Load and preprocess data
    (x_train, y_train, y_train_one_hot), (x_test, y_test, y_test_one_hot) = load_and_preprocess_data()

    # Build and train the model
    model = build_model()
    history = train_model(model, x_train, y_train_one_hot, x_test, y_test_one_hot)

    # Create 'uploads' folder if it doesn't exist
    if not os.path.exists(UPLOAD_FOLDER):
        os.makedirs(UPLOAD_FOLDER)

    # Run Flask app
    app.run(debug=True, use_reloader=False)

