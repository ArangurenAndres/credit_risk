from scripts.data_preprocessing import load_and_preprocess_data
from scripts.train_model import train_model
import os
import torch

def main():
    # Load and preprocess the data
    X_train, X_test, y_train, y_test = load_and_preprocess_data("data/credit_card_default.csv")

    # Train the model
    input_dim = X_train.shape[1]
    model, loss_values = train_model(X_train, y_train, input_dim)

    # Create the directory for saving the trained model if it doesn't exist
    os.makedirs('trained_models', exist_ok=True)

    # Save the trained model to the 'trained_models' folder
    model_save_path = 'trained_models/credit_default_model.pth'
    torch.save(model.state_dict(), model_save_path)
    print(f"Model saved to {model_save_path}")

    # Save and plot the training loss (make it pretty!)
    import matplotlib.pyplot as plt
    plt.figure(figsize=(10, 6))
    plt.plot(loss_values, color='blue', linewidth=2, label='Training Loss')
    plt.title('Training Loss Over Epochs', fontsize=16)
    plt.xlabel('Epochs', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(fontsize=12)
    plt.savefig('results/training_loss_pretty.png', bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    main()
