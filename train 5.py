import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, LeakyReLU
from tensorflow.keras.optimizers.legacy import Adam as LegacyAdam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os
import matplotlib.pyplot as plt

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Define data generators for training and testing
train_dir = 'train_PATH'
test_dir = 'test_PATH'
num_train = 28709
num_test = 7178
batch_size = 64
num_epoch = 30

# Add data augmentation to the training data generator
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=30,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(48, 48),
    batch_size=batch_size,
    color_mode="grayscale",
    class_mode='categorical')

test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(48, 48),
    batch_size=batch_size,
    color_mode="grayscale",
    class_mode='categorical')

# Create the model
model = Sequential()

model.add(Conv2D(64, kernel_size=(3, 3), activation=LeakyReLU(0.1), input_shape=(48, 48, 1))
model.add(Conv2D(128, kernel_size=(3, 3), activation=LeakyReLU(0.1))
model.add(MaxPooling2D(pool_size=(2, 2))
model.add(Dropout(0.25))

model.add(Conv2D(512, kernel_size=(3, 3), activation=LeakyReLU(0.1))
model.add(MaxPooling2D(pool_size=(2, 2))
model.add(Conv2D(512, kernel_size=(3, 3), activation=LeakyReLU(0.1))
model.add(MaxPooling2D(pool_size=(2, 2))
model.add(Dropout(0.5))  # Increased dropout rate

model.add(Flatten())
model.add(Dense(256))
model.add(LeakyReLU(0.1))  # LeakyReLU activation
model.add(Dropout(0.5))  # Increased dropout rate
model.add(Dense(7, activation='softmax'))

# Use the legacy Adam optimizer
optimizer = LegacyAdam(learning_rate=0.0001)

model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

# Define callbacks
checkpoint = ModelCheckpoint('emotion_model.h5', save_best_only=True, verbose=1)
early_stopping = EarlyStopping(patience=5, verbose=1, restore_best_weights=True)

# Train the model
history = model.fit(
    train_generator,
    steps_per_epoch=num_train // batch_size,
    epochs=num_epoch,
    validation_data=test_generator,
    validation_steps=num_test // batch_size,
    callbacks=[checkpoint, early_stopping])

# Evaluate the model on the test data
test_loss, test_accuracy = model.evaluate(test_generator, steps=num_test // batch_size)
print(f'Test Loss: {test_loss:.4f}')
print(f'Test Accuracy: {test_accuracy:.4f}')

# Save the trained model
model.save('emotion_model_real.h5')

# Plot model history
fig, axs = plt.subplots(1, 2, figsize=(15, 5))
axs[0].plot(range(1, len(history.history['accuracy']) + 1), history.history['accuracy'])
axs[0].set_title('Model Accuracy')
axs[0].set_ylabel('Accuracy')
axs[0].set_xlabel('Epoch')

axs[1].plot(range(1, len(history.history['loss']) + 1), history.history['loss'])
axs[1].set_title('Model Loss')
axs[1].set_ylabel('Loss')
axs[1].set_xlabel('Epoch')
plt.show()

model.save_weights('model.h5')
