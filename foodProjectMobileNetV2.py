import zipfile
import os

zip_file = "/content/drive/MyDrive/images.zip"
new_file_name = "Food101"

os.makedirs(new_file_name,exist_ok=True)

with zipfile.ZipFile(zip_file,"r") as zip_fol:
  zip_fol.extractall(new_file_name)


import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings("ignore")

base_dir = Path("/content/Food101")


def create_dataframe(directory):
    filepaths = []
    labels = []

    for label_dir in directory.iterdir():
        if label_dir.is_dir():
            for img_file in label_dir.iterdir():
                if img_file.suffix.lower() in ['.jpg', '.jpeg', '.png']:
                    filepaths.append(str(img_file))
                    labels.append(label_dir.name)


    if not filepaths or not labels:
        raise ValueError("Filepaths or labels are empty.")
    if len(filepaths) != len(labels):
        raise ValueError("Filepaths and labels must have the same length.")

    data = {'Filepath': filepaths, 'Label': labels}
    dataf = pd.DataFrame(data)
    return dataf

df = create_dataframe(base_dir)
train_df, test_df = train_test_split(df, train_size=0.20, random_state=42)


import tensorflow as tf
from keras.src import layers, models, Model
from keras.src.optimizers import Adam
from keras.src.legacy.preprocessing.image import ImageDataGenerator
from keras.api.applications import MobileNetV2
import matplotlib.pyplot as plt

num_classes = train_df['Label'].nunique()
img_height, img_width = 224, 224
batchSize = 32

train_datagen = ImageDataGenerator(
    rescale=1./255,
    horizontal_flip=True,
    fill_mode='nearest'
)

test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_dataframe(
    train_df,
    x_col='Filepath',
    y_col='Label',
    seed=42,
    target_size=(img_height, img_width),
    batch_size=batchSize,
    class_mode="categorical"
)

pretrained_model = MobileNetV2(
    input_shape=(img_height, img_width, 3),
    include_top=False,
    weights='imagenet'
)

pretrained_model.trainable = False

x = layers.GlobalAveragePooling2D()(pretrained_model.output)
x = layers.Dense(256, activation='relu')(x)
x = layers.BatchNormalization()(x)
x = layers.Dense(128, activation='relu')(x)
x = layers.BatchNormalization()(x)
x = layers.Dense(128, activation='relu')(x)
x = layers.BatchNormalization()(x)
outputs = layers.Dense(num_classes, activation='softmax')(x)

model = Model(pretrained_model.input, outputs, name='MobileNetV2')
optimizer = Adam(learning_rate=0.0001)
model.compile(
    optimizer=optimizer,
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

print(model.summary())

test_generator = test_datagen.flow_from_dataframe(
    dataframe=test_df,
    x_col='Filepath',
    y_col='Label',
    target_size=(img_height, img_width),
    batch_size=batchSize,
    class_mode="categorical"
)

model.fit(
    train_generator,
    steps_per_epoch=len(train_generator),
    epochs=15,
    validation_data=test_generator,
    validation_steps=len(test_generator)
)




loss, accuracy = model.evaluate(test_generator, steps=len(test_generator))
print(f"Test Loss: {loss}")
print(f"Test Accuracy: {accuracy}")
"""
%65
"""
