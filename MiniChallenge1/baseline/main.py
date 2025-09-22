import argparse
import sys
import os
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from joblib import dump, load
from sklearn.decomposition import PCA
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_validate
from sklearn.feature_selection import mutual_info_classif, RFE
from sklearn.ensemble import RandomForestClassifier


def feature_selection(train_data, option):
    scaler = StandardScaler()
    X_train = train_data['X_train']
    y_train = train_data['y_train']
    X_train = scaler.fit_transform(X_train)

    if option == 1:
        mi_scores = mutual_info_classif(X_train, y_train)
        # Chọn top 10 feature quan trọng nhất
        selected_features = np.argsort(mi_scores)[-10:]

        ### KET QUA: [8, 94, 15, 41, 97, 50, 84, 86, 30, 19]

    elif option == 2:
        # Sử dụng RandomForest làm mô hình đánh giá
        rf = RandomForestClassifier(n_estimators=100, random_state=42)

        # Áp dụng RFE để chọn 10 feature
        rfe = RFE(estimator=rf, n_features_to_select=10)
        rfe.fit(X_train, y_train)

        # Lấy danh sách feature quan trọng
        selected_features = np.where(rfe.support_)[0]

        ### KET QUA: [8, 15, 19, 22, 30, 41, 50, 56, 84, 86]
    elif option == 3:
        logreg = LogisticRegression(penalty='l1', solver='liblinear', C=0.1)  # Điều chỉnh C nếu cần
        logreg.fit(X_train, y_train)

        # Lấy các feature có hệ số khác 0
        selected_features = np.where(logreg.coef_[0] != 0)[0]

        ### KET QUA: [0, 1, 9, 11, 12, 15, 22, 26, 28, 29, 32, 37, 42, 51, 54, 56, 59, 65, 68, 71, 76, 80, 84, 87, 89, 91, 92, 93, 99]
    else:
        pca = PCA(n_components=10)
        X_pca = pca.fit_transform(X_train)

        # Lấy ma trận thành phần chính (components_)
        components = np.abs(pca.components_)  # Lấy giá trị tuyệt đối để đánh giá đóng góp

        # Tìm feature có trọng số cao nhất trong mỗi thành phần chính
        selected_features = []
        for i in range(10):
            most_important_feature = np.argmax(components[i])  # Lấy index của feature quan trọng nhất
            selected_features.append(most_important_feature)

        # Loại bỏ feature trùng lặp để có danh sách feature duy nhất
        selected_features = list(set(selected_features))

        ### KET QUA: [70, 8, 42, 13, 45, 19, 87, 25, 28]

        ### [8, 15, 19, 22, 30, 41, 50, 56, 84, 86]
    return selected_features

slf = sorted(list(set([8, 94, 15, 41, 97, 50, 84, 86, 30, 19, 8, 15, 19, 22, 30, 41, 50, 56, 84, 86, 0, 1, 9, 11, 12, 15, 22, 26, 28, 29, 32, 37, 42, 51, 54, 56, 59, 65, 68, 71, 76, 80, 84, 87, 89, 91, 92, 93, 99, 70, 8, 42, 13, 45, 19, 87, 25, 28])))

def get_selected_data(data, selected_feature):
    return data[:, selected_feature]

# Defines a function for training the model.
def run_train(public_dir, model_dir):
    # Ensures the model directory exists, creates it if it doesn't.
    os.makedirs(model_dir, exist_ok=True)
    
    # Constructs the path to the training data file.
    train_file = os.path.join(public_dir, 'train_data', 'train.npz')

    # Loads the training data from the .npz file.
    train_data = np.load(train_file)

    
    X_train = get_selected_data(train_data['X_train'], slf)
    

    # Extracts the labels from the training data.
    y_train = train_data['y_train']

    # # Instantiates the logistic regression model.
    # model = LogisticRegression(solver='saga', max_iter=1000)

    model = KNeighborsClassifier()

    # Initial GridSearch
    # Param Grid for Logistic Regression
    # param_grid = {
    #     'C': [0.01, 0.1, 1, 10],
    #     'penalty': ['l1', 'l2']
    # }

    # Param Grid for KNN
    param_grid = {
        'n_neighbors': [3, 5, 7, 9],  # Số lượng láng giềng cần thử nghiệm
        'weights': ['uniform', 'distance'],  # Trọng số
        'metric': ['euclidean', 'manhattan', 'minkowski']  # Khoảng cách
    }

    grid_search = GridSearchCV(model, param_grid=param_grid, scoring='accuracy', cv=5, return_train_score=True)

    # Using GridSearch to find the best model to the training data.
    grid_search.fit(X_train, y_train)

    # Chuyển kết quả thành DataFrame để dễ xem
    results = pd.DataFrame(grid_search.cv_results_)

    # Chỉ hiển thị các cột quan trọng
    # print(results[['param_n_neighbors', 'param_weights', 'mean_train_score', 'mean_test_score']])
    print(results[['mean_train_score', 'mean_test_score']])

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
    X_test = get_selected_data(test_data['X_test'], slf)

    # Predicts and saves results.
    pd.DataFrame({'y': model.predict(X_test)}).to_json(output_path, orient='records', lines=True)


# Defines the main function that parses commands and arguments.
def main():
    # Initializes a parser for command-line arguments.
    parser = argparse.ArgumentParser()

    # Creates subparsers for different commands.
    subparsers = parser.add_subparsers(dest='command')

    # Adds a subparser for the 'train' command.
    parser_train = subparsers.add_parser('train')

    # Adds an argument for the directory containing public data.
    parser_train.add_argument('--public_dir', type=str)

    # Adds an argument for the directory to save the model.
    parser_train.add_argument('--model_dir', type=str)

    # Adds a subparser for the 'predict' command.
    parser_predict = subparsers.add_parser('predict')

    # Adds an argument for the directory containing the model.
    parser_predict.add_argument('--model_dir', type=str)

    # Adds an argument for the directory containing test data.
    parser_predict.add_argument('--test_input_dir', type=str)

    # Adds an argument for the path to save prediction results.
    parser_predict.add_argument('--output_path', type=str)

    # Parses the command-line arguments.
    args = parser.parse_args()

    if args.command == 'train':
        # Checks if the 'train' command was given.
        # Calls the function to train the model.
        run_train(args.public_dir, args.model_dir)
    elif args.command == 'predict':
        # Checks if the 'predict' command was given.
        # Calls the function to make predictions.
        run_predict(args.model_dir, args.test_input_dir, args.output_path) 
    else:
        # If no valid command was given, prints the help message.
        # Displays help message for the CLI.
        parser.print_help()

        # Exits the script with a status code indicating an error.
        sys.exit(1)


# Checks if the script is being run as the main program.
if __name__ == "__main__":
    # Calls the main function if the script is executed directly.
    main()
