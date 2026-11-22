
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    ConfusionMatrixDisplay
)

print("Starting model evaluation...")

# 1. Load trained model
model = tf.keras.models.load_model(
    "model/cats_dogs_cnn.keras"
)

# 2. Load test dataset
test_ds = tf.keras.utils.image_dataset_from_directory(
    "data/test",
    image_size=(128, 128),
    batch_size=32,
    label_mode="binary",
    shuffle=False
)

class_names = test_ds.class_names

# 3. Normalize test images
def normalize(images, labels):
    images = tf.cast(images, tf.float32) / 255.0
    return images, labels

test_ds = test_ds.map(normalize)

# 4. Get actual labels
y_true = np.concatenate([
    labels.numpy().ravel()
    for images, labels in test_ds
]).astype(int)

# 5. Predict probabilities
predictions = model.predict(test_ds)

# 6. Convert probabilities into classes
y_pred = (predictions.ravel() >= 0.5).astype(int)

# 7. Generate confusion matrix
cm = confusion_matrix(
    y_true,
    y_pred
)

print("\nConfusion Matrix:")
print(cm)

# 8. Classification report
report = classification_report(
    y_true,
    y_pred,
    target_names=class_names
)

print("\nClassification Report:")
print(report)

# Save report
with open(
    "screenshots/classification_report.txt",
    "w"
) as file:
    file.write(report)

# 9. Plot confusion matrix
display = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=class_names
)

display.plot(cmap="Blues")

plt.title("Cats vs Dogs - Confusion Matrix")
plt.savefig(
    "screenshots/confusion_matrix.png",
    dpi=150
)
plt.show()

print("\nModel evaluation completed!")
