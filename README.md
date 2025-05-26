# Breast Cancer Prediction using Neural Networks

This repository contains a Jupyter Notebook (`untitled.ipynb`) that demonstrates a machine learning project for predicting breast cancer as either **Malignant** or **Benign**. The model is built using a simple Neural Network with **TensorFlow/Keras** and utilizes the **Breast Cancer Wisconsin (Diagnostic) Dataset** from `scikit-learn`.

---

## Table of Contents

- [Project Overview](#project-overview)  
- [Dataset](#dataset)  
- [Methodology](#methodology)  
- [Installation](#installation)  
- [Usage](#usage)  
- [Results](#results)  
- [Contributing](#contributing)  
- [License](#license)  

---

## Project Overview

The goal of this project is to classify breast cancer tumors as either **malignant (cancerous)** or **benign (non-cancerous)** based on various features extracted from digitized images of fine needle aspirates (FNA) of breast masses. A Neural Network model is trained and evaluated for this binary classification task.

---

## Dataset

The project uses the **Breast Cancer Wisconsin (Diagnostic) Dataset**, which is readily available through `sklearn.datasets`. This dataset contains **569 instances** with **30 features**, such as:

- mean radius  
- mean texture  
- mean perimeter  
- mean area  
- mean smoothness  
- mean compactness  
- mean concavity  
- mean concave points  
- mean symmetry  
- mean fractal dimension  

And additional error and worst measurements for these features.

The target variable `label` indicates:

- `0`: Malignant  
- `1`: Benign

---

## Methodology

### Data Loading and Exploration

- Loads the Breast Cancer dataset using `sklearn.datasets.load_breast_cancer()`
- Converts the dataset into a `pandas.DataFrame`
- Performs initial data inspection: `.head()`, `.shape`, `.info()`, `.isnull().sum()`, `.describe()`
- Analyzes the distribution of the target variable (`label`)
- Calculates mean feature values grouped by the label to observe differences

### Data Preprocessing

- Separates features (`X`) and target (`Y`)
- Splits data into training and testing sets with `train_test_split`
- Standardizes feature data using `StandardScaler`

### Model Building (Neural Network)

A sequential model is built using `tensorflow.keras.Sequential` with:

- A `Flatten` layer to prepare input  
- A `Dense` hidden layer with 20 neurons and ReLU activation  
- A `Dense` output layer with 2 neurons (binary classification) and sigmoid activation  

### Model Compilation and Training

- Compiled with:
  - Optimizer: `adam`
  - Loss: `sparse_categorical_crossentropy`
  - Metric: `accuracy`
- Trained on the training data for **10 epochs**, with a **10% validation split**

### Model Evaluation

- Evaluates performance on the test set
- Plots:
  - Training and validation accuracy
  - Training and validation loss

### Prediction System

- Demonstrates how to predict new, unseen data points
- Preprocesses the input data (conversion, reshaping, standardizing)
- Uses the trained model to predict and map output to class labels (Malignant or Benign)

---

## Installation

To run this notebook, ensure you have Python and the required libraries installed. Install them using:

```bash
pip install numpy pandas matplotlib scikit-learn tensorflow keras
````

---

## Usage

Clone the repository:

```bash
git clone https://github.com/your-username/breast-cancer-prediction.git
cd breast-cancer-prediction
```

Open the Jupyter Notebook:

```bash
jupyter notebook untitled.ipynb
```

Run all cells to execute the analysis, train the model, and see the prediction example.

---

## Results

The model achieved an accuracy of approximately **95.6%** on the test dataset. Accuracy and loss plots show effective learning and good generalization to unseen data.

---


