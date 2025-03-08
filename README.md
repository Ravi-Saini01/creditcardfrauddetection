
# Credit Card Fraud Detection

## Overview
This project implements a **credit card fraud detection system** using **Logistic Regression**. The dataset used is the "creditcard.csv" dataset, which contains transactions labeled as **fraudulent (1) or legitimate (0)**.

## Dataset
The dataset is highly **imbalanced**, meaning there are significantly more normal transactions than fraudulent ones. To address this, **undersampling** is used to create a balanced dataset.

### Features:
- **Time**: Seconds elapsed between this transaction and the first transaction.
- **Amount**: Transaction amount.
- **Class**: Target variable (0 = legitimate, 1 = fraudulent).
- **V1 to V28**: Principal components obtained using PCA (to anonymize sensitive data).

## Installation
Make sure you have Python and the required libraries installed. You can install them using:


pip install numpy pandas scikit-learn


## Project Workflow
1. **Load the dataset** from `creditcard.csv`.
2. **Explore the dataset**:
   - Check missing values.
   - Analyze distribution of legitimate vs. fraudulent transactions.
3. **Preprocess the data**:
   - Split data into **legitimate** and **fraudulent** transactions.
   - Perform **undersampling** to balance the dataset.
   - Split into **features (X) and target (Y)**.
   - Split into **training and testing sets (80-20 split)**.
4. **Train a Logistic Regression Model**.
5. **Evaluate the model** using accuracy scores for training and test data.

## Code Implementation

### 1. Load Required Libraries

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


### 2. Load and Explore Dataset

credit_card_data = pd.read_csv('creditcard.csv')
print(credit_card_data.info())
print(credit_card_data.isnull().sum())
print(credit_card_data['Class'].value_counts())


### 3. Handle Data Imbalance (Undersampling)

legit = credit_card_data[credit_card_data.Class == 0]
fraud = credit_card_data[credit_card_data.Class == 1]
legit_sample = legit.sample(n=492)  # Match fraud cases
new_dataset = pd.concat([legit_sample, fraud], axis=0)


### 4. Split Data into Features and Target

X = new_dataset.drop(columns='Class', axis=1)
Y = new_dataset['Class']
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, stratify=Y, random_state=2)


### 5. Train Logistic Regression Model

model = LogisticRegression(max_iter=7000)  # Increased iterations to ensure convergence
model.fit(X_train, Y_train)


### 6. Evaluate Model

X_train_prediction = model.predict(X_train)
training_data_accuracy = accuracy_score(X_train_prediction, Y_train)
print('Accuracy on training data:', training_data_accuracy)

X_test_prediction = model.predict(X_test)
test_data_accuracy = accuracy_score(X_test_prediction, Y_test)
print('Accuracy on test data:', test_data_accuracy)



The model achieves **high accuracy** on both training and test data, demonstrating its ability to detect fraudulent transactions effectively.

## Future Improvements
- Experiment with **different machine learning models** like Random Forest, SVM, or Neural Networks.
- Use **Oversampling techniques (SMOTE)** to handle imbalance instead of undersampling.
- Tune hyperparameters to further improve accuracy.

## Conclusion
This project successfully detects fraudulent credit card transactions using **Logistic Regression**. It demonstrates effective data preprocessing, model training, and evaluation techniques.

## License
This project is for educational purposes and is open-source. Feel free to modify and use it!

