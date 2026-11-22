
import tensorflow as tf
import matplotlib.pyplot as plt
import os

# =========================================
# 1. CONFIGURATION
# =========================================

IMAGE_SIZE = (128, 128)
BATCH_SIZE = 32
EPOCHS = 8
SEED = 42

print("Starting CNN model development...")

# =========================================
# 2. LOAD DATASETS
# =========================================

train_ds = tf.keras.utils.image_dataset_from_directory(
    "data/train",
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="binary",
    seed=SEED
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    "data/validation",
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="binary",
    shuffle=False
)

test_ds = tf.keras.utils.image_dataset_from_directory(
    "data/test",
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="binary",
    shuffle=False
)

class_names = train_ds.class_names

print("Classes:", class_names)

# =========================================
# 3. NORMALIZE IMAGES
# =========================================

AUTOTUNE = tf.data.AUTOTUNE

def normalize(images, labels):
    images = tf.cast(images, tf.float32) / 255.0
    return images, labels

train_ds = train_ds.map(
    normalize,
    num_parallel_calls=AUTOTUNE
).prefetch(AUTOTUNE)

val_ds = val_ds.map(
    normalize,
    num_parallel_calls=AUTOTUNE
).prefetch(AUTOTUNE)

test_ds = test_ds.map(
    normalize,
    num_parallel_calls=AUTOTUNE
).prefetch(AUTOTUNE)

# =========================================
# 4. BUILD CNN MODEL
# =========================================

model = tf.keras.Sequential([

    tf.keras.layers.Input(shape=(128, 128, 3)),

    # First convolution block
    tf.keras.layers.Conv2D(
        32, (3, 3), activation="relu"
    ),
    tf.keras.layers.MaxPooling2D((2, 2)),

    # Second convolution block
    tf.keras.layers.Conv2D(
        64, (3, 3), activation="relu"
    ),
    tf.keras.layers.MaxPooling2D((2, 2)),

    # Third convolution block
    tf.keras.layers.Conv2D(
        128, (3, 3), activation="relu"
    ),
    tf.keras.layers.MaxPooling2D((2, 2)),

    # Classification layers
    tf.keras.layers.Flatten(),

    tf.keras.layers.Dense(
        128, activation="relu"
    ),

    tf.keras.layers.Dropout(0.5),

    tf.keras.layers.Dense(
        1, activation="sigmoid"
    )
])

# =========================================
# 5. COMPILE MODEL
# =========================================

model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

print("\nCNN Model Summary:")
model.summary()

# =========================================
# 6. CALLBACKS
# =========================================

os.makedirs("model", exist_ok=True)
os.makedirs("screenshots", exist_ok=True)

early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor="val_loss",
    patience=3,
    restore_best_weights=True
)

checkpoint = tf.keras.callbacks.ModelCheckpoint(
    "model/best_cats_dogs.keras",
    monitor="val_loss",
    save_best_only=True
)

# =========================================
# 7. TRAIN CNN MODEL
# =========================================

print("\nStarting model training...")

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    callbacks=[early_stopping, checkpoint]
)

# =========================================
# 8. SAVE MODEL
# =========================================

model.save("model/cats_dogs_cnn.keras")

print("\nModel saved successfully!")

# =========================================
# 9. EVALUATE ON TEST DATA
# =========================================

print("\nEvaluating model on test dataset...")

test_loss, test_accuracy = model.evaluate(test_ds)

print("Test Loss:", test_loss)
print("Test Accuracy:", test_accuracy * 100, "%")

# =========================================
# 10. PLOT ACCURACY
# =========================================

plt.figure(figsize=(10, 5))

plt.plot(
    history.history["accuracy"],
    label="Training Accuracy"
)

plt.plot(
    history.history["val_accuracy"],
    label="Validation Accuracy"
)

plt.title("CNN Model Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.grid()

plt.savefig("screenshots/accuracy.png")
plt.show()

# =========================================
# 11. PLOT LOSS
# =========================================

plt.figure(figsize=(10, 5))

plt.plot(
    history.history["loss"],
    label="Training Loss"
)

plt.plot(
    history.history["val_loss"],
    label="Validation Loss"
)

plt.title("CNN Model Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.grid()

plt.savefig("screenshots/loss.png")
plt.show()

# =========================================
# 12. COMPLETION
# =========================================

print("\nCNN TRAINING COMPLETED!")
print("Model and graphs saved successfully.")
