import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Učitavanje podataka
train_df = pd.read_csv('train.csv', low_memory=False)
test_df = pd.read_csv('test.csv', low_memory=False)

# Uzmi samo prvih 2500 redova
train_df = train_df.head(2500)
test_df = test_df.head(2500)

# Uklanjanje nepotrebnih kolona
train_df = train_df.drop(columns=['ID', 'Name', 'SSN', 'Type_of_Loan', 'Customer_ID', 'Month'])

# Prikazivanje svih vrijednosti prije interpolacije
plt.figure(figsize=(12, 6))
for column in train_df.columns:
    if train_df[column].dtype in ['int64', 'float64']:
        plt.plot(train_df[column], label=column)
plt.title('Vrijednosti prije interpolacije')
plt.legend()
plt.show()

# Interpolacija nedostajućih vrijednosti
for column in train_df.columns:
    if train_df[column].isnull().sum() > 0:
        if train_df[column].dtype == 'object':
            # Za kategoričke podatke koristićemo SimpleImputer sa strategijom 'most_frequent'
            train_df[column].fillna(train_df[column].mode()[0], inplace=True)
        else:
            # Za numeričke podatke koristićemo linearnu interpolaciju
            train_df[column] = train_df[column].interpolate(method='spline', order=3, limit_direction='both')

# Prikazivanje svih vrednosti posle interpolacije
plt.figure(figsize=(12, 6))
for column in train_df.columns:
    if train_df[column].dtype in ['int64', 'float64']:
        plt.plot(train_df[column], label=column)
plt.title('Vrijednosti posle interpolacije')
plt.legend()
plt.show()


# Label Encoding za ciljno obeležje
label_encoder = LabelEncoder()
train_df['Credit_Score'] = label_encoder.fit_transform(train_df['Credit_Score'])

# Identifikacija numeričkih i kategoričkih kolona nakon uklanjanja nepotrebnih
numeric_features = train_df.select_dtypes(include=['int64', 'float64']).columns
categorical_features = train_df.select_dtypes(include=['object']).columns

# Imputacija i skaliranje za numeričke podatke
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())])

# Imputacija i One-Hot kodiranje za kategoričke podatke
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))])

# Kombinacija transformacija
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)])

# Podela podataka
X = train_df.drop('Credit_Score', axis=1)
y = train_df['Credit_Score']
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.3, random_state=42)

# Modeli
# Gradient Boosting Machines (GBM)
gbm = Pipeline(steps=[('preprocessor', preprocessor),
                      ('classifier', GradientBoostingClassifier())])

# Support Vector Machines (SVM)
svm = Pipeline(steps=[('preprocessor', preprocessor),
                      ('classifier', SVC(kernel='rbf'))])

# Treniranje i evaluacija GBM
gbm.fit(X_train, y_train)
y_pred_gbm = gbm.predict(X_val)
print("GBM Accuracy:", accuracy_score(y_val, y_pred_gbm))
print("GBM Precision:", precision_score(y_val, y_pred_gbm, average='macro'))
print("GBM Recall:", recall_score(y_val, y_pred_gbm, average='macro'))
print("GBM F1 Score:", f1_score(y_val, y_pred_gbm, average='macro'))

# Treniranje i evaluacija SVM
svm.fit(X_train, y_train)
y_pred_svm = svm.predict(X_val)
print("SVM Accuracy:", accuracy_score(y_val, y_pred_svm))
print("SVM Precision:", precision_score(y_val, y_pred_svm, average='macro'))
print("SVM Recall:", recall_score(y_val, y_pred_svm, average='macro'))
print("SVM F1 Score:", f1_score(y_val, y_pred_svm, average='macro'))
