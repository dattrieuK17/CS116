import argparse
import sys
import os
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.experimental import enable_iterative_imputer  # Kích hoạt IterativeImputer
from sklearn.impute import IterativeImputer, SimpleImputer, KNNImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, FunctionTransformer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_score
from joblib import dump, load
from sklearn.linear_model import BayesianRidge, Lasso
from sklearn.decomposition import PCA

import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter

def column_note(col):
    if col == 'sex':
        return 'Giới tính'
    elif col == 'age':
        return 'Tuổi hiện tại'
    elif col == 'race':
        return 'Chủng tộc'
    elif col == 'juv_fel_count':
        return 'Số trọng tội khi vị thành niên'
    elif col == 'juv_misd_count':
        return 'Số tội nhẹ khi vị thành niên'
    elif col == 'juv_other_count':
        return 'Hành vi phạm pháp khác khi vị thành niên'
    elif col == 'priors_count':
        return 'Số lần phạm tội trước đó'
    elif col == 'c_charge_degree':
        return 'Mức độ cáo buộc hình sự hiện tại'
    elif col == 'two_year_recid':
        return 'Có tái phạm trong 2 năm (nhãn đầu ra)'

def analysis_data(df):
    # Tạo bảng thông tin cho từng cột
    column_info = pd.DataFrame({
        'dtype': df.dtypes,
        'missing_count': df.isnull().sum(),
        'missing_ratio (%)': df.isnull().mean() * 100,
        'n_unique': df.nunique(),
        'example_values': df.apply(lambda col: col.dropna().unique()[:3])
    })

    # Định dạng lại tên cột, sắp xếp theo tỉ lệ missing giảm dần
    column_info = column_info.sort_values(by='missing_ratio (%)', ascending=False)
    column_info.reset_index(inplace=True)
    column_info.rename(columns={'index': 'column_name'}, inplace=True)
    # Áp dụng vào bảng thống kê
    column_info['Mô tả'] = column_info['column_name'].apply(column_note)
    # Hiển thị
    print(column_info)

    # Xác định kiểu dữ liệu
    numerical_cols = df.select_dtypes(include=['number']).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()

    print("\nCategorical columns:", categorical_cols)
    print("Numerical columns:", numerical_cols)

    # Đếm số lượng giá trị thiếu (null) cho mỗi mẫu (hàng)
    missing_counts = df.isnull().sum(axis=1)

    # Tính số lượng mẫu theo số lượng giá trị thiếu
    missing_count_distribution = Counter(missing_counts)

    
    # In kết quả
    print("\nSố lượng sample trong df:", len(df))
    for count, num_samples in sorted(missing_count_distribution.items()):
        if count == 0:
            print(f"Đủ giá trị: {num_samples}")
        else:
            print(f"Thiếu {count}: {num_samples}")
    
    df = normalize_null2nan(df)
    # ColumnTransformer xử lý số và chữ như trước
    ct = ColumnTransformer([
        ('num', Pipeline([
            ('imputer', IterativeImputer(estimator=Lasso(), max_iter=10, random_state=22520240)),
            # ('imputer', KNNImputer(n_neighbors=5, weights="uniform", metric="nan_euclidean"))
            ('scaler', StandardScaler())
        ]), numerical_cols),
        ('cat', Pipeline([
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False, drop='if_binary'))
        ]), categorical_cols),
    ])

    df_transformed = ct.fit_transform(df)

    # Lấy tên cột sau biến đổi 
    feature_names = ct.get_feature_names_out()
    print(feature_names)
    
    # Chuyển ndarray về DataFrame
    df_transformed = pd.DataFrame(df_transformed, columns=feature_names)

    # Tính ma trận tương quan
    corr_matrix = df_transformed.corr()

    # Vẽ heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap='coolwarm', square=True)
    plt.title("Tương quan giữa các biến số")
    plt.tight_layout()
    plt.show()



# python baseline/baseline/main.py train --public_dir public_data --model_dir save_model
num_features = ['age', 'juv_fel_count', 'juv_misd_count', 'juv_other_count', 'priors_count']
cat_features = ['sex', 'race', 'c_charge_degree']
def select_features(X):
    # X có đầy đủ cột bao gồm juvenile_count sau bước kế tiếp
    return X[num_features + cat_features]

def create_juvenile_count(X):
    """Hàm tạo cột juvenile_count từ juv_fel_count, juv_misd_count, juv_other_count"""
    X = X.copy()
    juvenile_cols = ['juv_fel_count', 'juv_misd_count', 'juv_other_count']
    if all(col in X.columns for col in juvenile_cols):
        X['juvenile_count'] = X[juvenile_cols].sum(axis=1)
        X = X.drop(columns=juvenile_cols)
    return X

def normalize_null2nan(df):
    df = df.copy()
    for feature in cat_features:
        if feature in df.columns:
            df[feature] = df[feature].replace({None: np.nan})
    return df
        
def run_train(public_dir, model_dir):
    os.makedirs(model_dir, exist_ok=True)
    
    # Load training data from JSON
    train_path = os.path.join(public_dir, 'train_data', 'train.json')
    df = pd.read_json(train_path, lines=True)

    # Loại bỏ các mẫu thiếu 5 giá trị trở lên
    missing_counts = df.isnull().sum(axis=1)
    df = df[missing_counts < 5].copy()
    print(len(df))
    # Split features and label
    X = df.drop('two_year_recid', axis=1)
    y = df['two_year_recid']

    # Identify categorical and numerical columns
    cat_features = X.select_dtypes(include=['object', 'category']).columns.tolist()
    num_features = X.select_dtypes(include=[np.number]).columns.tolist()


    # add_juvenile = FunctionTransformer(create_juvenile_count, validate=False)
    # bước chọn cột (thay cho lambda)
    select_cols  = FunctionTransformer(select_features, validate=False)
    normalize_cols = FunctionTransformer(normalize_null2nan, validate=False)

    # ColumnTransformer xử lý số và chữ như trước
    ct = ColumnTransformer([
        ('num', Pipeline([
            ('imputer', IterativeImputer(estimator=Lasso(), max_iter=10, random_state=22520240)),
            # ('imputer', KNNImputer(n_neighbors=5, weights="uniform", metric="nan_euclidean"))
            ('scaler', StandardScaler()), 
            ('pca', PCA(n_components=5, whiten=False, random_state=22520240))
        ]), num_features),
        ('cat', Pipeline([
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False, drop='if_binary'))
        ]), cat_features),
    ], remainder='drop')

    # Ghép tất cả vào một pipeline duy nhất
    preprocessor = Pipeline([
        # ('add_juvenile', add_juvenile),
        # ('select_cols',  select_cols),
        ('normalize_col', normalize_cols),
        ('transform',    ct)
    ])
    
    # Full feature pipeline
    X_processed = preprocessor.fit_transform(X)
    print(preprocessor['transform'].get_feature_names_out())
    print(X_processed.shape)

    
    ## 0.6032520611189018
    # # 1. Khởi tạo mô hình cơ bản
    # log_reg = LogisticRegression(solver='saga', max_iter=1000)
    
    # # 2. Khai báo grid các siêu tham số cần tìm
    # param_grid = {
    #     'C': [0.001, 0.01, 0.1, 1, 10],              # Hệ số điều chuẩn
    #     'penalty': ['l1', 'l2'],              # Loại regularization
    #     'fit_intercept': [True, False]        # Có tính hệ số chệch hay không
    # }

    # # 3. GridSearchCV để tìm mô hình tốt nhất với 5-fold cross-validation
    # grid_search = GridSearchCV(
    #     estimator=log_reg,
    #     param_grid=param_grid,
    #     cv=5,
    #     scoring='f1',      # Hoặc 'f1_macro' nếu phân loại đa lớp
    #     verbose=1,
    #     n_jobs=-1
    # )

    # # 4. Huấn luyện mô hình với GridSearch
    # grid_search.fit(X_processed, y)

    # # 5. In kết quả
    # print("Best parameters found:", grid_search.best_params_)
    # print("Best cross-validation f1-score:", grid_search.best_score_)

    # # 6. Truy cập mô hình tốt nhất
    # best_model = grid_search.best_estimator_

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

# train_path = os.path.join("public_data", 'train_data', 'train.json')
# df = pd.read_json(train_path, lines=True)
# analysis_data(df)