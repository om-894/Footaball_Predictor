

import numpy as np  # Unused import
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder  # Replaced StandardScaler for target with MinMaxScaler
import matplotlib.pyplot as plt  # Unused import
import glob
import os

# Set random seed for reproducibility
torch.manual_seed(42)

class SimpleMultiOutputModel(nn.Module):
    def __init__(self, input_size, output_size):
        super(SimpleMultiOutputModel, self).__init__()
        self.fc1 = nn.Linear(input_size, 8)  # Reduced neurons for simplicity
        self.fc2 = nn.Linear(8, 16)
        self.fc3 = nn.Linear(16, output_size)  # Output layer for predictions

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))  # Ensure non-negative outputs
        return x

# Function to preprocess and combine multiple files
def preprocess_data(file_paths, input_features, target_features):
    # Combine data from multiple files
    combined_data = pd.DataFrame()

    for file_path in file_paths:
        data = pd.read_csv(file_path)
        combined_data = pd.concat([combined_data, data], ignore_index=True)

    # Encode categorical variables (e.g., player names)
    label_encoders = {}
    for column in ['Player']:  # If there are more categorical columns, add them here
        le = LabelEncoder()
        combined_data[column] = le.fit_transform(combined_data[column])
        label_encoders[column] = le

    # Scale the input features and target variables
    scaler_X = StandardScaler()
    scaler_y = MinMaxScaler()  # Changed from StandardScaler to MinMaxScaler for target variables

    input_data = scaler_X.fit_transform(combined_data[input_features])
    target_data = scaler_y.fit_transform(combined_data[target_features])

    return input_data, target_data, scaler_X, scaler_y, combined_data, label_encoders

# Function to train the model
def train_model(input_data, target_data, epochs=1000, early_stopping_tolerance=0.001, patience=50):
    input_dim = input_data.shape[1]
    output_dim = target_data.shape[1]

    # Create a simple PyTorch model
    model = SimpleMultiOutputModel(input_dim, output_dim)

    # Loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0005, weight_decay=1e-2)

    best_loss = float('inf')
    epochs_no_improve = 0

    for epoch in range(epochs):
        # Forward pass
        inputs = torch.tensor(input_data, dtype=torch.float32)
        targets = torch.tensor(target_data, dtype=torch.float32)
        outputs = model(inputs)

        # Calculate loss
        loss = criterion(outputs, targets)

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Validation (for simplicity, using training data as validation here)
        with torch.no_grad():
            test_loss = criterion(model(inputs), targets).item()

        # Early stopping
        if test_loss < best_loss - early_stopping_tolerance:
            best_loss = test_loss
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= patience:
            print(f"Early stopping at epoch {epoch}. Best test loss: {best_loss:.4f}")
            break

        if epoch % 100 == 0:
            print(f"Epoch: {epoch} | Loss: {loss:.4f} | Test Loss: {test_loss:.4f}")

    return model

# Function to make predictions for a player
def predict_for_player(player_data, model, scaler_X, scaler_y):
    # Scale player data
    player_data_scaled = scaler_X.transform(player_data)
    player_data_tensor = torch.tensor(player_data_scaled, dtype=torch.float32)

    # Make predictions
    model.eval()
    with torch.no_grad():
        predictions = model(player_data_tensor)

    # Inverse transform predictions to the original scale
    predictions_original_scale = scaler_y.inverse_transform(predictions.numpy())

    # Clip predictions to avoid negative values
    predictions_original_scale = np.clip(predictions_original_scale, 0, None)

    return predictions_original_scale

# Main function to handle the whole process
def main():
    # List of file paths (replace with your actual file paths)
    file_paths = [
    'data/data_raw/sunderland_md1_player_stats.csv',
    'data/data_raw/sunderland_md2_player_stats.csv',
    'data/data_raw/sunderland_md3_player_stats.csv',
    'data/data_raw/sunderland_md4_player_stats.csv',
    'data/data_raw/sunderland_md5_player_stats.csv',
    'data/data_raw/sunderland_md6_player_stats.csv',
    'data/data_raw/sunderland_md7_player_stats.csv',
    'data/data_raw/sunderland_md8_player_stats.csv',
    'data/data_raw/sunderland_md9_player_stats.csv'
]

    # Define input and target features (replace with actual feature names)
    input_features = [
        'Min',
        'Performance_Touches',
        'Passes_Att',
        'Passes_Cmp',
        'Passes_PrgP',
        'Carries_Carries',
        'Take-Ons_Att',
        'Take-Ons_Succ',
        'Performance_Int',   # Interceptions
        'Performance_Tkl',   # Tackles
        'Performance_Blocks',
        'Expected_xG',       # Expected Goals
        'Expected_xAG',      # Expected Assists
        'Performance_CrdY',  # Yellow Cards
        'Performance_CrdR'   # Red Cards
    ]
    
    target_features = ['Fouls_Won', 'Fouls_Committed', 'Performance_Sh', 'Performance_SoT']  # Example targets

    # Preprocess the data from multiple files
    input_data, target_data, scaler_X, scaler_y, combined_data, label_encoders = preprocess_data(file_paths, input_features, target_features)

    # Train the model
    model = train_model(input_data, target_data)

    # Predict for a specific player
    player_name = "Chris Rigg"  # Replace with the player you want to predict for
    player_row = combined_data[combined_data['Player'] == label_encoders['Player'].transform([player_name])[0]].iloc[0]
    player_data = player_row[input_features].to_frame().T

    predictions = predict_for_player(player_data, model, scaler_X, scaler_y)
    print(f"Predictions for {player_name}:")
    print(predictions)

# Call the main function to run the process
if __name__ == "__main__":
    main()
