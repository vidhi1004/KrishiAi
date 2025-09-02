import numpy as np
import pandas as pd
import tensorflow as tf
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.preprocessing.image import ImageDataGenerator

def train_crop_model():
    df = pd.read_csv("/mnt/data/Crop_recommendation.csv")
    X = df[["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]]
    y = df["label"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestClassifier(n_estimators=100)
    model.fit(X_train, y_train)
    joblib.dump(model, "crop_recommendation.pkl")
    print("Crop Recommendation Model Trained & Saved")

def train_disease_model(dataset_path):
    img_size = 128
    batch_size = 32

    datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)
    train_generator = datagen.flow_from_directory(
        dataset_path,
        target_size=(img_size, img_size),
        batch_size=batch_size,
        class_mode='sparse',
        subset='training'
    )
    val_generator = datagen.flow_from_directory(
        dataset_path,
        target_size=(img_size, img_size),
        batch_size=batch_size,
        class_mode='sparse',
        subset='validation'
    )

    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(img_size, img_size, 3)),
        MaxPooling2D(2, 2),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D(2, 2),
        Flatten(),
        Dense(128, activation='relu'),
        Dense(15, activation='softmax')  # 15 classes in Plant Village dataset
    ])

    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    model.fit(train_generator, validation_data=val_generator, epochs=10)
    model.save("plant_disease_model.h5")
    print("Plant Disease Model Trained & Saved")

# Provide dataset path before running this function
# train_disease_model("/path/to/plant_village_dataset")

if __name__ == "__main__":
    train_crop_model()
    train_disease_model("/path/to/plant_village_dataset")


