import torch
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, precision_score, accuracy_score
from models.credit_default_model import CreditDefaultModel

def evaluate_model(X_test, y_test, model_path, input_dim):
    # Load the saved model
    model = CreditDefaultModel(input_dim)
    model.load_state_dict(torch.load(model_path))
    print(f"Model loaded from {model_path}")

    # Convert test data to PyTorch tensors
    X_test_tensor = torch.FloatTensor(X_test)
    y_test_tensor = torch.FloatTensor(y_test.values)

    # Set model to evaluation mode
    model.eval()

    # Evaluate model and get predictions
    with torch.no_grad():
        y_pred = model(X_test_tensor).squeeze()

    # Since y_pred is in [0, 1], threshold it to get binary predictions
    y_pred_class = (y_pred >= 0.5).float()

    # Calculate Accuracy
    accuracy = accuracy_score(y_test, y_pred_class)
    print(f"Accuracy: {accuracy:.4f}")

    # Calculate Precision
    precision = precision_score(y_test, y_pred_class)
    print(f"Precision: {precision:.4f}")

    # Calculate ROC-AUC
    fpr, tpr, _ = roc_curve(y_test, y_pred)
    roc_auc = auc(fpr, tpr)
    print(f"ROC-AUC: {roc_auc:.4f}")

    # Plot Pretty ROC Curve
    plt.figure(figsize=(10, 6))
    plt.plot(fpr, tpr, color='blue', linewidth=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='red', linestyle='--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.title('Receiver Operating Characteristic', fontsize=16)
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(loc='lower right', fontsize=12)
    plt.savefig('results/roc_curve_pretty.png', bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    # Load preprocessed data and test model
    from scripts.data_preprocessing import load_and_preprocess_data
    X_train, X_test, y_train, y_test = load_and_preprocess_data("data/credit_card_default.csv")
    
    # Path to the saved model
    model_path = 'trained_models/credit_default_model.pth'
    
    # Input dimension (number of features)
    input_dim = X_train.shape[1]

    # Evaluate the model
    evaluate_model(X_test, y_test, model_path, input_dim)
