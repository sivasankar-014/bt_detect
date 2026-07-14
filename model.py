import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
import os

# ==========================================
# Dataset Path
# ==========================================

train_path = os.path.join("data", "Training")
test_path = os.path.join("data", "Testing")

IMG_SIZE = (224, 224)
BATCH_SIZE = 32

# ==========================================
# Load Dataset
# ==========================================

train_data = tf.keras.utils.image_dataset_from_directory(
    train_path,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    color_mode="rgb",
    label_mode="int",
    shuffle=True
)

test_data = tf.keras.utils.image_dataset_from_directory(
    test_path,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    color_mode="rgb",
    label_mode="int",
    shuffle=False
)

print("Classes:", train_data.class_names)

# ==========================================
# Normalize Images
# ==========================================

normalization_layer = tf.keras.layers.Rescaling(1./255)

train_data = train_data.map(lambda x, y: (normalization_layer(x), y))
test_data = test_data.map(lambda x, y: (normalization_layer(x), y))

# ==========================================
# CNN Model
# ==========================================

model = Sequential([

    Conv2D(32, (3,3), activation='relu', input_shape=(224,224,3)),
    MaxPooling2D(),

    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D(),

    Conv2D(128, (3,3), activation='relu'),
    MaxPooling2D(),

    Flatten(),

    Dense(256, activation='relu'),
    Dropout(0.5),

    Dense(1, activation='sigmoid')
])

# ==========================================
# Compile
# ==========================================

model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

model.summary()


# ==========================================
# Train
# ==========================================

history = model.fit(
    train_data,
    validation_data=test_data,
    epochs=10
)

# ==========================================
# Evaluate
# ==========================================

loss, accuracy = model.evaluate(test_data)

print("Test Accuracy:", accuracy)

# ==========================================
# Save Model
# ==========================================

model.save("brain_tumor_model.keras")

print("Model Saved Successfully")