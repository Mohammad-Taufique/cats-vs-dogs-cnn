
# Cats vs Dogs Image Classification using CNN

A Deep Learning project that classifies images of cats and dogs using Convolutional Neural Networks (CNN) and TensorFlow.

## Project Overview
This project uses a CNN model to identify whether an uploaded image contains a cat or a dog.

## Technologies Used
- Python
- TensorFlow / Keras
- CNN (Deep Learning)
- NumPy
- Matplotlib
- Scikit-learn
- Streamlit

## Project Features
- Dataset preparation and preprocessing
- CNN model training
- Model evaluation
- Confusion matrix and classification report
- Image prediction through Streamlit web application

## Model Performance
The CNN model was evaluated on 3,750 test images.

- Test Accuracy: 82%
- Cat F1-Score: 0.83
- Dog F1-Score: 0.81

## Project Structure
- app.py
- train_model.py
- preprocess.py
- prepare_dataset.py
- evaluate_model.py
- check_environment.py
- model/
- screenshots/
- requirements.txt

### Evaluation Visualizations

- Training Accuracy: screenshots/accuracy.png
- Confusion Matrix: screenshots/confusion_matrix.png
- Classification Report: screenshots/classification_report.txt  

## Installation

```bash
git clone https://github.com/Mohammad-Taufique/cats-vs-dogs-cnn.git
cd cats-vs-dogs-cnn

pip install -r requirements.txt
```

## Run the Application

```bash
streamlit run app.py
```

## Author
Mohammad Taufique

GitHub: https://github.com/Mohammad-Taufique

