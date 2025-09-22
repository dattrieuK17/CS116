import argparse
import sys
import os
import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression, Lasso
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.tree import DecisionTreeClassifier
from sklearn.experimental import enable_iterative_imputer  # Kích hoạt IterativeImputer
from sklearn.impute import IterativeImputer, SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.model_selection import cross_val_score, GridSearchCV, StratifiedKFold
from joblib import dump, load
from sklearn.base import BaseEstimator, TransformerMixin

# class LassoImputer(BaseEstimator, TransformerMixin):
#     def __init__(self, alpha=0.01, max_iter=1000):
#         self.alpha = alpha
#         self.max_iter = max_iter
#         self.models = {}
#         self.columns = []

#     def fit(self, X, y=None):
#         self.columns = X.columns
#         X = pd.DataFrame(X, columns=self.columns)

#         for col in self.columns:
#             missing = X[col].isnull()
#             if missing.any():
#                 not_missing = ~missing
#                 X_train = X.loc[not_missing].drop(columns=[col])
#                 y_train = X.loc[not_missing, col]
#                 model = Lasso(alpha=self.alpha, max_iter=self.max_iter)
#                 model.fit(X_train, y_train)
#                 self.models[col] = model

#         return self

#     def transform(self, X):
#         X = pd.DataFrame(X, columns=self.columns)
#         for col in self.models:
#             missing = X[col].isnull()
#             if missing.any():
#                 X_pred = X.loc[missing].drop(columns=[col])
#                 X.loc[missing, col] = self.models[col].predict(X_pred)
#         return X.values

# python baseline/baseline/main.py --public_dir public_data --model_dir save_model
def run_train(public_dir, model_dir):
    os.makedirs(model_dir, exist_ok=True)
    
    # Load training data from JSON
    train_path = os.path.join(public_dir, 'train_data', 'train.json')
    df = pd.read_json(train_path, lines=True)

    # Split features and label
    X = df.drop('two_year_recid', axis=1)
    y = df['two_year_recid']

    # Identify categorical and numerical columns
    cat_features = X.select_dtypes(include=['object', 'category']).columns.tolist()
    num_features = X.select_dtypes(include=[np.number]).columns.tolist()

    # # Preprocessing pipeline
    # column_transformer = ColumnTransformer(
    #     transformers=[
    #         ('num', Pipeline([
    #             ('scaler', StandardScaler()),
    #             ('imputer', IterativeImputer(estimator=Lasso(), max_iter=20, random_state=42))
    #         ]), num_features),
    #         ('cat', Pipeline([
    #             ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False)),
    #             ('imputer', IterativeImputer(estimator=RandomForestClassifier(), max_iter=20, random_state=42))
    #         ]), cat_features)
    #     ]
    # )

    # Preprocessing pipeline
    column_transformer = ColumnTransformer(
        transformers=[
            ('num', Pipeline([
                ('imputer', IterativeImputer(estimator=Lasso(), max_iter=20, random_state=42)),
                ('scaler', StandardScaler())
            ]), num_features),
            ('cat', Pipeline([
                ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False)),
                ('imputer', IterativeImputer(estimator=DecisionTreeClassifier(), max_iter=20, random_state=42))
            ]), cat_features)
        ]
    )
    
#    # Pipeline tổng thể: preprocessor + PCA + RandomForestClassifier
#    ### KET QUA: 
#    ###          Best parameters: {'classifier__max_depth': 10, 'classifier__min_samples_split': 2, 'classifier__n_estimators': 200, 'pca__n_components': 8}
#    ###          Best cross-validation accuracy: 0.6363663660716914
#    ###          Cross-validation accuracy of best model: 0.6363663660716914
#     preprocessor = Pipeline([
#         ('preprocessor', column_transformer),
#         ('pca', PCA(random_state=42)),
#         ('classifier', RandomForestClassifier(random_state=42))
#     ])

#     # Định nghĩa lưới tham số cho GridSearchCV
#     param_grid = {
#         'pca__n_components': [4, 6, 8, 10],  # Số lượng thành phần chính
#         'classifier__n_estimators': [100, 200],  # Số cây trong RandomForest
#         'classifier__max_depth': [None, 10, 20],  # Độ sâu tối đa của cây
#         'classifier__min_samples_split': [2, 5]  # Số mẫu tối thiểu để chia node
#     }

    # Pipeline tổng thể: preprocessor + PCA + XGBClassifier
    ### KET QUA: 
    ###          Best parameters: {'classifier__learning_rate': 0.01, 'classifier__max_depth': 3, 'classifier__n_estimators': 200, 'classifier__subsample': 0.8, 'pca__n_components': 8}
    ###          Best cross-validation accuracy: 0.6454723331599761
    ###          Cross-validation accuracy of best model: 0.6454723331599761
    preprocessor = Pipeline([
        ('preprocessor', column_transformer),
        ('pca', PCA(random_state=42, n_components=8))
    ])

    X_processed = preprocessor.fit_transform(X)
    # best_model = XGBClassifier(
    #     learning_rate=0.01,
    #     max_depth=3,
    #     n_estimators=200,
    #     subsample=0.8,
    #     random_state=42,
    #     eval_metric='logloss')
    
    # best_model.fit(X_processed, y)

    best_model = LogisticRegression(C=0.01, fit_intercept=False, penalty='l1', solver='saga')
    best_model.fit(X_processed, y)

    # Thiết lập Stratified K-Fold để giữ tỷ lệ nhãn
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=22520240)

    # Tính score theo F1 cho từng fold
    scores = cross_val_score(
        best_model,        # có thể là Pipeline hoặc estimator độc lập
        X_processed,       # dữ liệu đã transform
        y,                 # nhãn
        cv=cv,
        scoring='f1',      # đánh giá theo F1
        n_jobs=-1
    )

    print("Cross-validation F1 scores:", scores)
    print("Mean F1 score:", np.mean(scores), "±", np.std(scores))
    # Save model and preprocessor
    dump(best_model, os.path.join(model_dir, 'trained_model.joblib'))
    dump(preprocessor, os.path.join(model_dir, 'preprocessor.joblib'))


def run_predict(model_dir, test_input_dir, output_path):
    # Load model and preprocessor
    model = load(os.path.join(model_dir, 'trained_model.joblib'))
    preprocessor = load(os.path.join(model_dir, 'preprocessor.joblib'))

    # Load test data from JSON
    test_path = os.path.join(test_input_dir, 'test.json')
    df_test = pd.read_json(test_path, lines=True)

    # Transform test data
    X_test = preprocessor.transform(df_test)
    preds = model.predict(X_test)

    # Save predictions with proper column name
    pd.DataFrame({'two_year_recid': preds}).to_json(output_path, orient='records', lines=True)


def main():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest='command')

    # Train command
    parser_train = subparsers.add_parser('train')
    parser_train.add_argument('--public_dir', type=str)
    parser_train.add_argument('--model_dir', type=str)

    # Predict command
    parser_predict = subparsers.add_parser('predict')
    parser_predict.add_argument('--model_dir', type=str)
    parser_predict.add_argument('--test_input_dir', type=str)
    parser_predict.add_argument('--output_path', type=str)

    args = parser.parse_args()

    if args.command == 'train':
        run_train(args.public_dir, args.model_dir)
    elif args.command == 'predict':
        run_predict(args.model_dir, args.test_input_dir, args.output_path)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
