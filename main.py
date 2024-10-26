

import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import glob
import os

torch.manual_seed(42)

# Data loading functions
def load_data(folder_path, player_name):
    file_paths = glob.glob(os.path.join(folder_path, '*.csv'))
    dfs = [pd.read_csv(file_path) for file_path in file_paths]
    data = pd.concat(dfs, ignore_index=True)
    return data[data['Player'] == player_name]

# Data preparation functions
def prepare_data(data, input_features, target_columns):
    X = data[input_features]
    y = data[target_columns]

    scaler_X = StandardScaler()
    scaler_y = StandardScaler()
    X = scaler_X.fit_transform(X)
    y = scaler_y.fit_transform(y)

    return torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.float32), scaler_X, scaler_y

# Model definition
class SimpleMultiOutputModel(nn.Module):
    def __init__(self, input_size, output_size):
        super(SimpleMultiOutputModel, self).__init__()
        self.fc1 = nn.Linear(input_size, 8)
        self.fc2 = nn.Linear(8, 16)
        self.fc3 = nn.Linear(16, output_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Model training function with early stopping
def train_model(model, X_train, y_train, X_test, y_test, criterion, optimizer, epochs=100, patience=10):
    best_loss = float('inf')
    patience_count = 0

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        outputs = model(X_train)
        loss = criterion(outputs, y_train)
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            test_outputs = model(X_test)
            test_loss = criterion(test_outputs, y_test)

            if test_loss < best_loss:
                best_loss = test_loss
                patience_count = 0
            else:
                patience_count += 1
            if patience_count >= patience:
                print(f"Early stopping at epoch {epoch + 1}. Best test loss: {best_loss:.4f}")
                break

        if (epoch) % 10 == 0:
            print(f'Epoch: {epoch} | Loss: {loss.item():.4f} | Test Loss: {test_loss.item():.4f}')

# Prediction function
def predict_for_matchweek_10(model, data, input_features, scaler_X, scaler_y):
    avg_data = data[input_features].mean().to_frame().T
    avg_scaled = scaler_X.transform(avg_data)
    avg_tensor = torch.tensor(avg_scaled, dtype=torch.float32)

    model.eval()
    with torch.no_grad():
        predictions = model(avg_tensor)
    predictions_original_scale = scaler_y.inverse_transform(predictions.numpy())

    print("Predictions for Matchweek 10 (based on average stats from Matchweeks 1-9):")
    print(predictions_original_scale)

# Main pipeline
if __name__ == "__main__":
    folder_path = 'data/normalized_per_90_min'
    player_name = 'Dennis Cirkin'
    input_features = [
        'Performance_Touches', 'Passes_Att', 'Passes_Cmp', 'Passes_PrgP', 'Carries_Carries',
        'Take-Ons_Att', 'Take-Ons_Succ', 'Performance_Int', 'Performance_Tkl',
        'Performance_Blocks', 'Expected_xG', 'Expected_xAG', 'Performance_CrdY', 'Performance_CrdR'
    ]
    target_columns = ['Fouls_Won', 'Fouls_Committed', 'Performance_Sh', 'Performance_SoT']

    data = load_data(folder_path, player_name)
    X, y, scaler_X, scaler_y = prepare_data(data, input_features, target_columns)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = SimpleMultiOutputModel(X_train.shape[1], y_train.shape[1])
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0005, weight_decay=1e-2)

    train_model(model, X_train, y_train, X_test, y_test, criterion, optimizer)
    predict_for_matchweek_10(model, data, input_features, scaler_X, scaler_y)
