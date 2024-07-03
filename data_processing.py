import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
from imblearn.over_sampling import SMOTE, ADASYN
from imblearn.under_sampling import RandomUnderSampler
from sklearn.utils import resample
import numpy as np

def preprocess_data(df, preprocessor=None, fit=True):
    df = df.drop(columns=['ID', 'Name', 'SSN', 'Type_of_Loan', 'Customer_ID', 'Month', 'Changed_Credit_Limit'])

    df['Payment_Behaviour'] = df['Payment_Behaviour'].apply(lambda x: np.NaN if x == "!@9#%8" else x)
    df['Credit_Mix'] = df['Credit_Mix'].apply(lambda x: np.NaN if x == "_" else x)

    for column in df.columns:
        if df[column].isnull().sum() > 0:
            if df[column].dtype == 'object':
                df[column].fillna(df[column].mode()[0], inplace=True)
            else:
                df[column] = df[column].interpolate(method='spline', order=3, limit_direction='both')

    if 'Credit_Score' in df.columns:
        X = df.drop('Credit_Score', axis=1)
        y = df['Credit_Score']
        label_encoder = LabelEncoder()
        y = label_encoder.fit_transform(y)
    else:
        X = df
        y = None

    numeric_features = X.select_dtypes(include=['int64', 'float64']).columns
    categorical_features = X.select_dtypes(include=['object']).columns

    numeric_transformer = Pipeline(steps=[
        ('scaler', StandardScaler())])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))])

    if preprocessor is None:
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, numeric_features),
                ('cat', categorical_transformer, categorical_features)])
        if fit:
            X_processed = preprocessor.fit_transform(X).toarray()
        else:
            X_processed = preprocessor.transform(X).toarray()
    else:
        if fit:
            X_processed = preprocessor.fit_transform(X).toarray()
        else:
            X_processed = preprocessor.transform(X).toarray()

    return X_processed, y, preprocessor

def oversample_minority_classes(X, y):
    data = pd.DataFrame(X)
    data['y'] = y

    data_class_1 = data[data['y'] == 0]
    data_class_2 = data[data['y'] == 1]
    data_class_3 = data[data['y'] == 2]

    max_size = max(len(data_class_1), len(data_class_2), len(data_class_3))

    data_class_1_oversampled = resample(data_class_1, replace=True, n_samples=max_size, random_state=42)
    data_class_2_oversampled = resample(data_class_2, replace=True, n_samples=max_size, random_state=42)
    data_class_3_oversampled = resample(data_class_3, replace=True, n_samples=max_size, random_state=42)

    data_oversampled = pd.concat([data_class_1_oversampled, data_class_2_oversampled, data_class_3_oversampled])

    X_oversampled = data_oversampled.drop('y', axis=1).values
    y_oversampled = data_oversampled['y'].values

    return X_oversampled, y_oversampled

def load_data():
    train_df = pd.read_csv('train.csv', encoding='utf-8', low_memory=False, nrows=2500)
    X_train, y_train, preprocessor = preprocess_data(train_df, fit=True)

    # Originalna distribucija podataka
    plt.figure(figsize=(8, 6))
    plt.pie(pd.Series(y_train).value_counts(), labels=['Poor', 'Standard', 'Good'], autopct='%.0f%%')
    plt.title('Originalna distribucija Credit_Score')
    plt.show()

    # Primena SMOTE na treniranje podatke
    smote = SMOTE(random_state=42)
    X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)
    
    plt.figure(figsize=(8, 6))
    plt.pie(pd.Series(y_train_smote).value_counts(), labels=['Poor', 'Standard', 'Good'], autopct='%.0f%%')
    plt.title('Distribucija Credit_Score posle SMOTE')
    plt.show()

    # Primena ADASYN na treniranje podatke
    adasyn = ADASYN(random_state=42)
    X_train_adasyn, y_train_adasyn = adasyn.fit_resample(X_train, y_train)

    plt.figure(figsize=(8, 6))
    plt.pie(pd.Series(y_train_adasyn).value_counts(), labels=['Poor', 'Standard', 'Good'], autopct='%.0f%%')
    plt.title('Distribucija Credit_Score posle ADASYN')
    plt.show()

    # Primena Random UnderSampler na treniranje podatke
    rus = RandomUnderSampler(random_state=42)
    X_train_rus, y_train_rus = rus.fit_resample(X_train, y_train)

    plt.figure(figsize=(8, 6))
    plt.pie(pd.Series(y_train_rus).value_counts(), labels=['Poor', 'Standard', 'Good'], autopct='%.0f%%')
    plt.title('Distribucija Credit_Score posle Random UnderSampler')
    plt.show()

    # Oversampling manjinskih klasa
    X_train_balanced, y_train_balanced = oversample_minority_classes(X_train_rus, y_train_rus)

    plt.figure(figsize=(8, 6))
    plt.pie(pd.Series(y_train_balanced).value_counts(), labels=['Poor', 'Standard', 'Good'], autopct='%.0f%%')
    plt.title('Distribucija Credit_Score posle oversamplinga')
    plt.show()

    # Podela podataka na trening i validacioni set
    X_train, X_val, y_train, y_val = train_test_split(X_train_balanced, y_train_balanced, test_size=0.15, random_state=42)

    return X_train, X_val, y_train, y_val, preprocessor

def load_test_data(preprocessor):
    test_df = pd.read_csv('test.csv', encoding='utf-8', low_memory=False, nrows=2500)
    X_test, _, _ = preprocess_data(test_df, preprocessor=preprocessor, fit=False)

    return X_test
