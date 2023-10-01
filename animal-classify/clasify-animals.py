# %%
# DATASET : https://www.kaggle.com/datasets/alessiocorrado99/animals10

# %%
# usual imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import VGG16
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.applications import VGG16
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D


# %%
path_image = './animals/used'

# %%

# Buat model VGG16 (tanpa lapisan terakhir/final layer)
vgg_model = VGG16(weights='imagenet', include_top=False, input_shape=(150, 150, 3))

# Membekukan lapisan-lapisan model VGG16 agar tidak terlatih
for layer in vgg_model.layers:
    layer.trainable = False

# %%
augmentation = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    horizontal_flip=True,
    vertical_flip=True,
    shear_range=0.2,
    fill_mode='nearest',
    validation_split=0.2,
)

train_data = augmentation.flow_from_directory(
    path_image,
    target_size=(150, 150),
    class_mode='categorical',
    subset='training'
)

validation_data = augmentation.flow_from_directory(
    path_image,
    target_size=(150, 150),
    class_mode='categorical',
    subset='validation'
)

labels = list(train_data.class_indices)

print(f'List of label : {list(train_data.class_indices)}')

# %%
ACCURACY_TRESHOLD = 95e-2


class CallbackTraining(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        v_accuracy = logs.get('val_accuracy')
        accuracy = logs.get('accuracy')
        if accuracy >= ACCURACY_TRESHOLD and v_accuracy >= ACCURACY_TRESHOLD:
            print(
                f'\n Epoch {epoch}\n Accuracy has reach = {logs["accuracy"]*100:.2f}%/n training has been stopped.')
            self.model.stop_training = True


early_stop_callback = EarlyStopping(
    monitor='val_accuracy',
    patience=100,
    restore_best_weights=True
)

# %%
# Buat model VGG16 (tanpa lapisan terakhir/final layer)
vgg_model = VGG16(weights='imagenet', include_top=False, input_shape=(150, 150, 3))

# Membekukan lapisan-lapisan model VGG16 agar tidak terlatih
for layer in vgg_model.layers:
    layer.trainable = False

model_3 = tf.keras.models.Sequential([
    vgg_model,
    tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(3, activation='softmax')
])

# count loss function and optimizer
model_3.compile(
    loss='categorical_crossentropy',
    optimizer=tf.optimizers.legacy.Adam(learning_rate=0.001),
    metrics=['accuracy']
)

# train data
history = model_3.fit(
    train_data,
    epochs=100,
    validation_data=validation_data,
    validation_steps=4,
    verbose=1,
    shuffle=True,
    callbacks=[CallbackTraining(),early_stop_callback]
)

# %%
file_name = 'model_anjing_ayam_laba2.keras'
model_3.save(file_name)

converter = tf.lite.TFLiteConverter.from_keras_model(model_3)
tflite_model = converter.convert()
 
 
with tf.io.gfile.GFile('model_anjing_ayam_laba2.tflite', 'wb') as f:
    f.write(tflite_model)

# %%
plt.title("Epoch Result")
plt.plot(history.history.get('accuracy'))
plt.plot(history.history.get('val_accuracy'))
plt.plot(history.history.get('loss'))
plt.legend(['accuracy','val_accuracy','loss'])
plt.show()

# %%
plt.title("Accuracy per epoch")
plt.plot(history.history.get('accuracy'))
plt.plot(history.history.get('val_accuracy'))
plt.legend(['accuracy','val_accuracy'])
plt.show()


