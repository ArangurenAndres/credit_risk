import torch
from torch.utils.data import DataLoader, TensorDataset
from models.credit_default_model import CreditDefaultModel
from tqdm import tqdm

def train_model(X_train, y_train, input_dim, epochs=30, learning_rate=0.001):
    # Prepare DataLoader
    train_data = TensorDataset(torch.FloatTensor(X_train), torch.FloatTensor(y_train.values))
    train_loader = DataLoader(train_data, batch_size=32, shuffle=True)

    # Initialize the model
    model = CreditDefaultModel(input_dim)
    criterion = torch.nn.BCELoss()  # Binary cross-entropy loss
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Training Loop with tqdm progress bar
    loss_values = []
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        
        # Add tqdm progress bar to training loop
        with tqdm(train_loader, unit="batch") as tepoch:
            tepoch.set_description(f"Epoch {epoch+1}/{epochs}")
            for inputs, labels in tepoch:
                optimizer.zero_grad()
                outputs = model(inputs).squeeze()
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()

                # Update tqdm bar with loss information
                tepoch.set_postfix(loss=loss.item())

        loss_values.append(running_loss / len(train_loader))
        print(f"Epoch {epoch+1}/{epochs}, Loss: {running_loss / len(train_loader)}")

    return model, loss_values
