import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def load_and_preprocess_data(file_path):
    # Load the dataset
    df = pd.read_csv(file_path)

    # Rename 'default payment next month' to 'credit_default' for clarity
    df.rename(columns={'default.payment.next.month': 'credit_default'}, inplace=True)

    # Define features and target
    X = df.drop('credit_default', axis=1)
    y = df['credit_default']

    # Standardize the features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Split into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    return X_train, X_test, y_train, y_test
