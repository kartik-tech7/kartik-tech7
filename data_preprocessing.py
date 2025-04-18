import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer

def load_and_preprocess_data(filepath):
    data = pd.read_csv(filepath)

    # Handling missing values
    imputer = SimpleImputer(strategy='mean')
    data_imputed = pd.DataFrame(imputer.fit_transform(data), columns=data.columns)

    # Splitting features and target
    X = data_imputed.drop('Potability', axis=1)
    y = data_imputed['Potability']

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    return X_train, X_test, y_train, y_test
