
import tensorflow as tf
import matplotlib.pyplot as plt
from pathlib import Path

# ==================================
# 1. CONFIGURATION
# ==================================

IMAGE_SIZE = (128, 128)
BATCH_SIZE = 32
SEED = 42

print("Starting image preprocessing...")

# ==================================
# 2. DATASET PATHS
# ==================================

train_dir = "data/train"
val_dir = "data/validation"
test_dir = "data/test"

# ==================================
# 3. LOAD DATASETS
# ==================================

train_ds = tf.keras.utils.image_dataset_from_directory(
    train_dir,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="binary",
    seed=SEED,
    shuffle=True
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    val_dir,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="binary",
    shuffle=False
)

test_ds = tf.keras.utils.image_dataset_from_directory(
    test_dir,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="binary",
    shuffle=False
)

class_names = train_ds.class_names

print("\nClass names:", class_names)

# ==================================
# 4. NORMALIZATION
# ==================================

def normalize_images(images, labels):
    images = tf.cast(images, tf.float32) / 255.0
    return images, labels

AUTOTUNE = tf.data.AUTOTUNE

train_ds = train_ds.map(
    normalize_images,
    num_parallel_calls=AUTOTUNE
).prefetch(AUTOTUNE)

val_ds = val_ds.map(
    normalize_images,
    num_parallel_calls=AUTOTUNE
).prefetch(AUTOTUNE)

test_ds = test_ds.map(
    normalize_images,
    num_parallel_calls=AUTOTUNE
).prefetch(AUTOTUNE)

# ==================================
# 5. CHECK IMAGE SHAPE
# ==================================

images, labels = next(iter(train_ds))

print("\nImage batch shape:", images.shape)
print("Label batch shape:", labels.shape)

print("Minimum pixel value:", float(tf.reduce_min(images)))
print("Maximum pixel value:", float(tf.reduce_max(images)))

# ==================================
# 6. VISUALIZE IMAGES
# ==================================

plt.figure(figsize=(10, 8))

for i in range(9):

    plt.subplot(3, 3, i + 1)

    plt.imshow(images[i].numpy())

    label = int(labels[i].numpy()[0])

    plt.title(class_names[label])
    plt.axis("off")

plt.tight_layout()

Path("screenshots").mkdir(exist_ok=True)

plt.savefig(
    "screenshots/preprocessed_samples.png",
    dpi=150
)

plt.show()

# ==================================
# 7. COMPLETION
# ==================================

print("\nImage preprocessing completed!")
print("Training, validation and testing datasets are ready.")
