
import random
import shutil
from pathlib import Path
from PIL import Image

SOURCE = Path("data/raw/train")

random.seed(42)

classes = ["cat", "dog"]
splits = {
    "train": 0.70,
    "validation": 0.15,
    "test": 0.15
}

print("Starting dataset preparation...")

if not SOURCE.exists():
    raise FileNotFoundError("Dataset folder not found!")

for split in splits:
    for category in classes:
        Path(f"data/{split}/{category}").mkdir(
            parents=True, exist_ok=True
        )

for category in classes:

    images = []

    for file in SOURCE.rglob("*.jpg"):
        if not file.name.lower().startswith(category + "."):
            continue

        try:
            with Image.open(file) as img:
                img.verify()
            images.append(file)

        except Exception:
            print("Corrupted image:", file.name)

    random.shuffle(images)

    total = len(images)

    train_end = int(total * 0.70)
    val_end = train_end + int(total * 0.15)

    groups = {
        "train": images[:train_end],
        "validation": images[train_end:val_end],
        "test": images[val_end:]
    }

    for split, files in groups.items():

        destination = Path(f"data/{split}/{category}")

        for file in files:
            shutil.copy2(file, destination / file.name)

        print(f"{category} - {split}: {len(files)}")

print("\nDataset preparation completed!")
