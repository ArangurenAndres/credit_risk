import xgboost as xgb
from sklearn.metrics import roc_curve, auc, accuracy_score, precision_score
import matplotlib.pyplot as plt
import pandas as pd
import joblib

def train_xgboost(X_train, y_train, X_test, y_test):
    # Create the DMatrix for XGBoost
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dtest = xgb.DMatrix(X_test, label=y_test)
    
    # Set up the parameters for XGBoost
    params = {
        'objective': 'binary:logistic',
        'eval_metric': 'logloss',
        'max_depth': 6,
        'eta': 0.1,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
    }

    # Dictionary to store evaluation results
    evals_result = {}

    # Train the model and capture evaluation results
    evallist = [(dtrain, 'train'), (dtest, 'eval')]
    model = xgb.train(params, dtrain, num_boost_round=100, evals=evallist, 
                      early_stopping_rounds=10, evals_result=evals_result)
    
    # Save the model
    joblib.dump(model, 'trained_models/xgboost_model.pkl')
    print("XGBoost model saved to trained_models/xgboost_model.pkl")

    # Plot training loss
    plot_training_loss(evals_result)

    return model

def plot_training_loss(evals_result):
    # Extract training and evaluation log loss
    epochs = len(evals_result['train']['logloss'])
    x_axis = range(0, epochs)
    
    # Plot log loss for training and evaluation data
    plt.figure(figsize=(10, 6))
    plt.plot(x_axis, evals_result['train']['logloss'], label='Train')
    plt.plot(x_axis, evals_result['eval']['logloss'], label='Test')
    plt.legend()
    plt.ylabel('Log Loss')
    plt.title('XGBoost Training and Test Log Loss')
    plt.grid(True)
    plt.savefig('results/xgboost_training_loss.png')
    plt.show()

def evaluate_xgboost(model, X_test, y_test):
    # Convert the test data to DMatrix
    dtest = xgb.DMatrix(X_test)
    
    # Predict using the trained XGBoost model
    y_pred_proba = model.predict(dtest)
    y_pred_class = (y_pred_proba >= 0.5).astype(int)

    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred_class)
    precision = precision_score(y_test, y_pred_class)
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    roc_auc = auc(fpr, tpr)

    print(f"XGBoost Accuracy: {accuracy:.4f}")
    print(f"XGBoost Precision: {precision:.4f}")
    print(f"XGBoost ROC-AUC: {roc_auc:.4f}")

    # Plot ROC Curve
    plt.figure(figsize=(10, 6))
    plt.plot(fpr, tpr, color='blue', label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='red', linestyle='--')
    plt.title('XGBoost ROC Curve', fontsize=16)
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.legend(loc='lower right')
    plt.grid(True)
    plt.savefig('results/xgboost_roc_curve.png')
    plt.show()

if __name__ == "__main__":
    from data_preprocessing import load_and_preprocess_data

    # Load preprocessed data
    X_train, X_test, y_train, y_test = load_and_preprocess_data("data/credit_card_default.csv")

    # Train XGBoost model
    xgb_model = train_xgboost(X_train, y_train, X_test, y_test)

    # Evaluate XGBoost model
    evaluate_xgboost(xgb_model, X_test, y_test)
