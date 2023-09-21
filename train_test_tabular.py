import pandas as pd
from sklearn.discriminant_analysis import StandardScaler
from sklearn.model_selection import train_test_split
import torch
from src.tabular.dataset import TableDataset
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim


from src.tabular.model import SimpleDeepClassifier


if __name__ == "__main__":
    # Read CSV
    print("1. Reading data...")
    synthetic_df = pd.read_csv("src/tabular/data/ten_thousand_samples.csv")
    # Split input with output
    X = synthetic_df.loc[:, synthetic_df.columns.str.startswith("feature")].values
    y = synthetic_df.loc[:, "label"].values
    classes = set(y)
    number_of_classes = len(classes)
    # Scale all features with Z-normalization
    print("2. Scaling...")
    scaler = StandardScaler()
    scaler.fit(X)
    preprocessed_X = scaler.transform(X)

    # Split data
    train_x, test_x, train_y, test_y = train_test_split(
        preprocessed_X, y, test_size=0.3
    )

    # Create PyTorch Dataset objects for train data and test data
    #
    print("3. Creating loaders...")
    train_loader = DataLoader(
        TableDataset(train_x, train_y), batch_size=16, shuffle=True
    )
    val_loader = DataLoader(TableDataset(test_x, test_y), batch_size=16, shuffle=True)

    # Create Model
    print("4. Creating Model...")
    model = SimpleDeepClassifier(X.shape[1], number_of_classes)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(params=model.parameters(), lr=0.01)

    # Training
    print("5. Train and Validation")
    epochs = 10
    for epoch in range(epochs):
        running_loss = 0.0
        print("• Beginning epoch", epoch)
        print("----- Training -----")
        for i, (batch_X, batch_gt_y) in enumerate(train_loader, start=0):
            optimizer.zero_grad()
            batch_pred_y = model(batch_X)
            # predicted_labels = torch.argmax(batch_pred_y, dim=1)
            loss = criterion(batch_pred_y, batch_gt_y)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if i % 32 == 0:
                print(f"[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 32:.3f}")
                running_loss = 0

        print("----- Validation -----")
        # Validation Loop
        model.eval()  # Set the model in evaluation mode
        val_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for i, (batch_X, batch_gt_y) in enumerate(val_loader, start=0):
                batch_pred_y = model(batch_X)
                val_loss += criterion(batch_pred_y, batch_gt_y)
                predicted_labels = torch.argmax(batch_pred_y, dim=1)
                total += batch_gt_y.size(0)
                correct += (predicted_labels == batch_gt_y).sum().item()

            val_accuracy = correct / total
            print(
                f"Epoch [{epoch + 1}/{epochs}] "
                f"Training Loss: {loss.item():.4f} "
                f"Validation Loss: {val_loss / len(val_loader):.4f} "
                f"Validation Accuracy: {val_accuracy:.2%}"
            )

    print("6. Saving model...")
    torch.save(model.state_dict(), "simple_deep_classifier_on_synthetic.pt")

    print("Done")
