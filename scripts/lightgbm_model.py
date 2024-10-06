import lightgbm as lgb
from sklearn.metrics import roc_curve, auc, accuracy_score, precision_score
import matplotlib.pyplot as plt
import joblib

def train_lightgbm(X_train, y_train, X_test, y_test):
    # Create the LightGBM Dataset
    train_data = lgb.Dataset(X_train, label=y_train)
    test_data = lgb.Dataset(X_test, label=y_test)

    # Set parameters for LightGBM
    params = {
        'objective': 'binary',
        'metric': 'binary_logloss',
        'boosting_type': 'gbdt',
        'num_leaves': 31,
        'learning_rate': 0.05,
        'feature_fraction': 0.9
    }

    # Train the model
    model = lgb.train(params, train_data, valid_sets=[train_data, test_data], num_boost_round=100)

    # Save the model
    joblib.dump(model, 'trained_models/lightgbm_model.pkl')
    print("LightGBM model saved to trained_models/lightgbm_model.pkl")

    return model

def evaluate_lightgbm(model, X_test, y_test):
    # Predict using the trained LightGBM model
    y_pred_proba = model.predict(X_test)
    y_pred_class = (y_pred_proba >= 0.5).astype(int)

    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred_class)
    precision = precision_score(y_test, y_pred_class)
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    roc_auc = auc(fpr, tpr)

    print(f"LightGBM Accuracy: {accuracy:.4f}")
    print(f"LightGBM Precision: {precision:.4f}")
    print(f"LightGBM ROC-AUC: {roc_auc:.4f}")

    # Plot ROC Curve
    plt.figure(figsize=(10, 6))
    plt.plot(fpr, tpr, color='blue', label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='red', linestyle='--')
    plt.title('LightGBM ROC Curve', fontsize=16)
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.legend(loc='lower right')
    plt.grid(True)
    plt.savefig('results/lightgbm_roc_curve.png')
    plt.show()

if __name__ == "__main__":
    from data_preprocessing import load_and_preprocess_data

    # Load preprocessed data
    X_train, X_test, y_train, y_test = load_and_preprocess_data("data/credit_card_default.csv")

    # Train LightGBM model
    lgb_model = train_lightgbm(X_train, y_train, X_test, y_test)

    # Evaluate LightGBM model
    evaluate_lightgbm(lgb_model, X_test, y_test)
