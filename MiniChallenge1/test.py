import argparse
import sys
import os
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from joblib import dump, load
from sklearn.decomposition import PCA
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler

# Defines a function for training the model.
def run_train(public_dir, model_dir):
    # Ensures the model directory exists, creates it if it doesn't.
    os.makedirs(model_dir, exist_ok=True)
    scaler = StandardScaler()
    pca = PCA(n_components=10)
    # Constructs the path to the training data file.
    train_file = os.path.join(public_dir, 'train_data', 'train.npz')

    # Loads the training data from the .npz file.
    train_data = np.load(train_file)
    # Using PCA 
    train_data_pca = pca.fit_transform(train_data['X_train'])

    scaled_data = scaler.fit_transform(train_data_pca)

    
    # Extracts the features from the training data.
    X_train = scaled_data

    # Extracts the labels from the training data.
    y_train = train_data['y_train']

    # Instantiates the logistic regression model.
    model = LogisticRegression(solver='saga', max_iter=1000)

    # Initial GridSearch
    param_grid = {
        'C': [0.01, 0.1, 1, 10],
        'penalty': ['l1', 'l2']
    }

    grid_search = GridSearchCV(model, param_grid=param_grid, scoring='accuracy')

    # Using GridSearch to find the best model to the training data.
    grid_search.fit(X_train, y_train)

    # Get best model
    best_model = grid_search.best_estimator_

    # Defines the path for saving the trained model.
    model_path = os.path.join(model_dir, 'trained_model.joblib')

    # Saves the trained model to the specified path.
    dump(best_model, model_path)


# Defines a function for making predictions.
def run_predict(model_dir, test_input_dir, output_path):
    # Specifies the path to the trained model.
    model_path = os.path.join(model_dir, 'trained_model.joblib')

    # Constructs the path to the test data file.
    test_file = os.path.join(test_input_dir, 'test.npz')

    # Loads the trained model from file.
    model = load(model_path)

    # Loads the test data from the .npz file.
    test_data = np.load(test_file)

    # Extracts the features from the test data.
    X_test = test_data['X_test']

    # Predicts and saves results.
    pd.DataFrame({'y': model.predict(X_test)}).to_json(output_path, orient='records', lines=True)



run_train("D:/CS116-Python Programming for Machine Learning/CS116.P23 - Mini Challenge 1 (hạn nộp 13-59-59 04-04-2025) (Problem)/public_data", "model")
