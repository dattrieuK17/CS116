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
from sklearn.preprocessing import OneHotEncoder, StandardScaler, FunctionTransformer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.model_selection import cross_val_score, GridSearchCV, StratifiedKFold
from joblib import dump, load
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score

# import seaborn as sns
# from collections import Counter

# def column_note(col):
#     if col == 'sex':
#         return 'Giới tính'
#     elif col == 'age':
#         return 'Tuổi hiện tại'
#     elif col == 'race':
#         return 'Chủng tộc'
#     elif col == 'juv_fel_count':
#         return 'Số trọng tội khi vị thành niên'
#     elif col == 'juv_misd_count':
#         return 'Số tội nhẹ khi vị thành niên'
#     elif col == 'juv_other_count':
#         return 'Hành vi phạm pháp khác khi vị thành niên'
#     elif col == 'priors_count':
#         return 'Số lần phạm tội trước đó'
#     elif col == 'c_charge_degree':
#         return 'Mức độ cáo buộc hình sự hiện tại'
#     elif col == 'two_year_recid':
#         return 'Có tái phạm trong 2 năm (nhãn đầu ra)'

# def analysis_data(df):
#     # Tạo bảng thông tin cho từng cột
#     column_info = pd.DataFrame({
#         'dtype': df.dtypes,
#         'missing_count': df.isnull().sum(),
#         'missing_ratio (%)': df.isnull().mean() * 100,
#         'n_unique': df.nunique(),
#         'example_values': df.apply(lambda col: col.dropna().unique()[:3])
#     })

#     # Định dạng lại tên cột, sắp xếp theo tỉ lệ missing giảm dần
#     column_info = column_info.sort_values(by='missing_ratio (%)', ascending=False)
#     column_info.reset_index(inplace=True)
#     column_info.rename(columns={'index': 'column_name'}, inplace=True)
#     # Áp dụng vào bảng thống kê
#     column_info['Mô tả'] = column_info['column_name'].apply(column_note)
#     # Hiển thị
#     print(column_info)

#     # Xác định kiểu dữ liệu
#     numerical_cols = df.select_dtypes(include=['number']).columns.tolist()
#     categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()

#     print("\nCategorical columns:", categorical_cols)
#     print("Numerical columns:", numerical_cols)

#     # Đếm số lượng giá trị thiếu (null) cho mỗi mẫu (hàng)
#     missing_counts = df.isnull().sum(axis=1)

#     # Tính số lượng mẫu theo số lượng giá trị thiếu
#     missing_count_distribution = Counter(missing_counts)

    
#     # In kết quả
#     print("\nSố lượng sample trong df:", len(df))
#     for count, num_samples in sorted(missing_count_distribution.items()):
#         if count == 0:
#             print(f"Đủ giá trị: {num_samples}")
#         else:
#             print(f"Thiếu {count}: {num_samples}")
    
#     df = normalize_null2nan(df)
#     # ColumnTransformer xử lý số và chữ như trước
#     ct = ColumnTransformer([
#         ('num', Pipeline([
#             ('imputer', IterativeImputer(estimator=Lasso(), max_iter=10, random_state=22520240)),
#             # ('imputer', KNNImputer(n_neighbors=5, weights="uniform", metric="nan_euclidean"))
#             ('scaler', StandardScaler())
#         ]), numerical_cols),
#         ('cat', Pipeline([
#             ('imputer', SimpleImputer(strategy='most_frequent')),
#             ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False, drop='if_binary'))
#         ]), categorical_cols),
#     ])

#     df_transformed = ct.fit_transform(df)

#     # Lấy tên cột sau biến đổi 
#     feature_names = ct.get_feature_names_out()
#     print(feature_names)
    
#     # Chuyển ndarray về DataFrame
#     df_transformed = pd.DataFrame(df_transformed, columns=feature_names)

#     # Tính ma trận tương quan
#     corr_matrix = df_transformed.corr()

#     # Vẽ heatmap
#     plt.figure(figsize=(10, 8))
#     sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap='coolwarm', square=True)
#     plt.title("Tương quan giữa các biến số")
#     plt.tight_layout()
#     plt.show()


# python baseline/baseline/main.py --public_dir public_data --model_dir save_model
def normalize_null2nan(X):
    X_copy = X.copy()
    cat_features = X.select_dtypes(include=['object', 'category']).columns.tolist()
    for feature in cat_features:
        if feature in X_copy.columns:
            X_copy[feature] = X[feature].replace({None: np.nan})
    return X_copy


def visualize_PCA(public_dir):
    # Load training data from JSON
    train_path = os.path.join(public_dir, 'train_data', 'train.json')
    df = pd.read_json(train_path, lines=True)

    # Split features and label
    X = df.drop('two_year_recid', axis=1)
    y = df['two_year_recid']

     # Identify categorical and numerical columns
    num_features = X.select_dtypes(include=[np.number]).columns.tolist()
    cat_features = X.select_dtypes(include=['object', 'category']).columns.tolist()

    # Preprocessing pipeline
    column_transformer = ColumnTransformer(
        transformers=[
            ('num', Pipeline([
                ('imputer', IterativeImputer(estimator=Lasso(), max_iter=20, random_state=22520240)),
                ('scaler', StandardScaler())
            ]), num_features),
            ('cat', Pipeline([
                ('imputer', SimpleImputer(strategy='most_frequent')), 
                ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False, drop='if_binary'))
            ]), cat_features)
        ]
    )
    normalize_cols = FunctionTransformer(normalize_null2nan, validate=False)

    preprocessor = Pipeline([
        ('normalize_cols', normalize_cols),
        ('preprocessor', column_transformer),
    ])

    X_processed = preprocessor.fit_transform(X)
    pca = PCA()
    X_pca = pca.fit_transform(X_processed)

    explained_variance = np.cumsum(pca.explained_variance_ratio_)

    plt.figure(figsize=(8, 4))
    plt.plot(range(1, len(explained_variance) + 1), explained_variance, marker='o', linestyle='--')
    plt.xlabel('Số lượng thành phần PCA')
    plt.ylabel('Tổng phương sai tích lũy')
    plt.title('Biểu đồ chọn số lượng thành phần PCA')
    plt.grid(True)
    plt.axhline(y=0.95, color='r', linestyle=':')
    plt.show()

    # Bước 3: Áp dụng PCA giữ 95% phương sai
    pca_95 = PCA(n_components=0.95)
    X_pca_95 = pca_95.fit_transform(X_pca)

    print(f"Số thành phần được giữ khi n_components=0.95: {X_pca_95.shape[1]}")

############################################
def predict_with_threshold(model, X, thresh):
    proba = model.predict_proba(X)[:, 1]
    return (proba >= thresh).astype(int)

def find_optimized_threshold(public_dir):
    # 0. Load và tiền xử lý
    train_path = os.path.join(public_dir, 'train_data', 'train.json')
    df = pd.read_json(train_path, lines=True)
    X = df.drop('two_year_recid', axis=1)
    y = df['two_year_recid']

    num_features = X.select_dtypes(include=[np.number]).columns.tolist()
    cat_features = X.select_dtypes(include=['object', 'category']).columns.tolist()

    column_transformer = ColumnTransformer([
        ('num', Pipeline([
            ('imputer', IterativeImputer(estimator=Lasso(), max_iter=20, random_state=22520240)),
            ('scaler', StandardScaler())
        ]), num_features),
        ('cat', Pipeline([
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False, drop='if_binary'))
        ]), cat_features),
    ])

    preprocessor = Pipeline([
        ('normalize', FunctionTransformer(normalize_null2nan, validate=False)),
        ('transform', column_transformer),
        ('pca', PCA(random_state=22520240, n_components=7))
    ])
    X_proc = preprocessor.fit_transform(X)

    # 1. Split train/validation
    X_train, X_val, y_train, y_val = train_test_split(
        X_proc, y, test_size=0.2, stratify=y, random_state=42
    )

    # 2. Train model
    model = LogisticRegression(C=0.5, penalty='l1', fit_intercept=False, solver='saga', random_state=22520240, max_iter=1000, class_weight='balanced')
    model.fit(X_train, y_train)

    # 3. Lấy xác suất trên validation
    probs = model.predict_proba(X_val)[:, 1]

    # 4. Quét thresholds
    thresholds = np.linspace(0.1, 0.9, 81)
    f1_scores = []
    acc_scores = []
    for t in thresholds:
        preds = (probs >= t).astype(int)
        f1_scores.append(f1_score(y_val, preds))
        acc_scores.append(accuracy_score(y_val, preds))

    # 5. Tìm threshold tối ưu
    best_idx_acc = np.argmax(acc_scores)
    best_thresh_acc = thresholds[best_idx_acc]

    best_idx_f1 = np.argmax(f1_scores)
    best_thresh_f1 = thresholds[best_idx_f1]

    # 6. (Tùy chọn) Threshold tối ưu chung
    combined = np.array(f1_scores) + np.array(acc_scores)
    best_idx_comb = np.argmax(combined)
    best_thresh_comb = thresholds[best_idx_comb]

    # 7. In kết quả
    print(f1_scores)
    print(acc_scores)
    print(f"Best for F1      : threshold = {best_thresh_f1:.2f}, F1 = {f1_scores[best_idx_f1]:.4f}")
    print(f"Best for Accuracy: threshold = {best_thresh_acc:.2f}, Acc = {acc_scores[best_idx_acc]:.4f}")
    print(f"Best Combined    : threshold = {best_thresh_comb:.2f}, F1+Acc = {combined[best_idx_comb]:.4f}")
####################################################

def run_train(public_dir, model_dir):
    os.makedirs(model_dir, exist_ok=True)
    
    # Load training data from JSON
    train_path = os.path.join(public_dir, 'train_data', 'train.json')
    df = pd.read_json(train_path, lines=True)

    # Split features and label
    X = df.drop('two_year_recid', axis=1)
    y = df['two_year_recid']

    print(sum(y)/len(y))

    # Identify categorical and numerical columns
    num_features = X.select_dtypes(include=[np.number]).columns.tolist()
    cat_features = X.select_dtypes(include=['object', 'category']).columns.tolist()

    # Preprocessing pipeline
    column_transformer = ColumnTransformer(
        transformers=[
            ('num', Pipeline([
                ('imputer', IterativeImputer(estimator=Lasso(), max_iter=20, random_state=22520240)),
                ('scaler', StandardScaler())
            ]), num_features),
            ('cat', Pipeline([
                ('imputer', SimpleImputer(strategy='most_frequent')), 
                ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False, drop='if_binary'))
            ]), cat_features)
        ]
    )
    normalize_cols = FunctionTransformer(normalize_null2nan, validate=False)

    preprocessor = Pipeline([
        ('normalize_cols', normalize_cols),
        ('preprocessor', column_transformer),
        ('pca', PCA(random_state=22520240, n_components=8))
    ])

    # # Pipeline cho gridsearch logistic regression
    # pipeline = Pipeline([
    #     ('normalize_cols', normalize_cols),
    #     ('preprocessor', column_transformer),
    #     ('pca', PCA(random_state=22520240)),
    #     ('classifier', LogisticRegression(max_iter=1000, random_state=22520240, class_weight='balanced'))

    # ])
    # # Lưới tham số
    # param_grid = {
    #     'pca__n_components': [2, 3, 4, 5, 6, 7, 8, 9, 10],
    #     'classifier__C': [0.01, 0.05, 0.1, 0.15, 0.5, 1],
    #     'classifier__penalty': ['l1'],
    #     'classifier__solver': ['saga']  # Dùng được với 'l2'
    # }


    # # GridSearchCV
    # grid_search = GridSearchCV(
    #     estimator=pipeline,
    #     param_grid=param_grid,
    #     scoring="accuracy",
    #     cv=5,
    #     n_jobs=-1,
    #     verbose=1
    # )

    # # Fit vào dữ liệu huấn luyện
    # grid_search.fit(X, y)

    # # In kết quả
    # print("Best parameters:", grid_search.best_params_)
    # print("Best score:", grid_search.best_score_)

    # Full feature pipeline
    
    X_processed = preprocessor.fit_transform(X)
    print(X_processed[2])
    print(preprocessor.named_steps['preprocessor'].get_feature_names_out())
    # Model training
    model = LogisticRegression(C=0.5, penalty='l1', fit_intercept=False, solver='saga', random_state=22520240, max_iter=1000, class_weight='balanced')
    model.fit(X_processed, y)

    # Save model and preprocessor
    dump(model, os.path.join(model_dir, 'trained_model.joblib'))
    dump(preprocessor, os.path.join(model_dir, 'preprocessor.joblib'))


def run_predict(model_dir, test_input_dir, output_path):
        # Load model và preprocessor
    model = load(os.path.join(model_dir, 'trained_model.joblib'))
    preprocessor = load(os.path.join(model_dir, 'preprocessor.joblib'))

    # Load test data từ JSON
    test_path = os.path.join(test_input_dir, 'test.json')
    df_test = pd.read_json(test_path, lines=True)

    # Transform test data
    X_test = preprocessor.transform(df_test)

    # Lấy xác suất positive class (two_year_recid = 1)
    probs = model.predict_proba(X_test)[:, 1]

    # Áp threshold = 0.35 để chuyển xác suất thành nhãn 0/1
    preds = (probs >= 0.55).astype(int)

    # Lưu kết quả
    pd.DataFrame({'two_year_recid': preds}) \
      .to_json(output_path, orient='records', lines=True)


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

# find_optimized_threshold("public_data")