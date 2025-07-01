# Fake Image Detection using Deep Learning

This project aims to detect fake (synthetically generated or manipulated) images using a Convolutional Neural Network (CNN)-based deep learning approach. The model is trained to differentiate between real and fake images using pixel-level patterns and features.

## 📁 Project Structure

- `fake_image_detection_updated.ipynb`: Main Jupyter notebook containing the entire pipeline—from data loading to model training and evaluation.
- `datasets/`: Directory for real and fake image datasets.
- `models/`: Saved trained models for reuse or deployment.

## 📌 Objective

The goal is to build a robust image classification model that can distinguish between:
- **Real images**: Natural, unaltered images.
- **Fake images**: AI-generated or tampered visuals (e.g., GANs, Deepfakes).

## 🛠️ Technologies Used

- Python 3.x
- TensorFlow / Keras
- NumPy & Pandas
- OpenCV
- Matplotlib
- Scikit-learn

## 📊 Dataset

The dataset is organized into two folders:
- Images are resized to 128x128 for faster training.
- Data is split into training, validation, and test sets.

## 🧠 Model Architecture

The CNN model includes:

- Convolutional Layers with ReLU activation
- MaxPooling Layers
- Dropout for regularization
- Dense Layers for classification
- Sigmoid activation for binary output (real vs fake)

```python
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 3)),
    MaxPooling2D(2, 2),
    Dropout(0.2),

    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Dropout(0.2),

    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.3),
    Dense(1, activation='sigmoid')
])
















# Fake-Image-Detection
- Collected and prepared a diverse dataset of real and fake images.
- Resized the images for optimal input to the model.
- Trained a Convolutional Neural Network (CNN) using an appropriate loss function and optimizer.
- Evaluated the model’s performance by calculating the accuracy score.
