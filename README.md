# Loan Default Prediction Model

This repository contains code for building and evaluating a machine learning model to predict loan defaults based on borrower characteristics and loan terms. The model architecture includes both wide and deep paths, leveraging interactions between categorical and numerical features to enhance prediction performance.

## Loading the Data

The data is loaded from a CSV file located at '/Users/danilomontalvo/Desktop/MLAssignemnts/Lab5/loan_data.csv'. Preliminary analysis includes checking for missing values and dropping unnecessary columns.

## Preprocessing Data

Before training the model, the data is preprocessed as follows:
- Conversion of the target variable 'not.fully.paid' to integer type.
- Separation of categorical and numerical features.
- Standardization of numerical features and one-hot encoding of categorical features using pipelines and column transformers.

## Model Architecture

The model consists of two paths:
1. **Wide Path**: Directly utilizes categorical input.
2. **Deep Path**: Concatenates numerical and one-hot encoded categorical features, followed by two dense layers.

The final output layer utilizes sigmoid activation for binary classification.

## Training the Model

The model is trained using the Adam optimizer and binary cross-entropy loss function. Training history is plotted to visualize the model's performance over epochs.

## Evaluation Metrics

Two primary metrics are used to evaluate the model:
- **Area Under the ROC Curve (AUC-ROC)**: Evaluates the model's ability to discriminate between fully paid and not fully paid loans across various classification thresholds.
- **F1-Score**: Balances precision and recall, providing a robust measure of performance for imbalanced datasets.

## Data Splitting Method

Shuffle split cross-validation is used for model evaluation, ensuring random sampling of data to create multiple training and testing sets. This method mimics real-world scenarios where data arrives in no specific order, providing a stable assessment of the model's performance.

## Evaluating the Model

The model is evaluated on the test set, and the test accuracy is reported.

![Model Architecture](model_architecture.png)
