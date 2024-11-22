import json
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import xgboost as xgb
from torch.utils.data import DataLoader, TensorDataset, random_split
from transformers import AutoTokenizer, AutoModel
from sklearn.preprocessing import MultiLabelBinarizer, LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, classification_report, mean_absolute_error
from warnings import simplefilter # Filter out Pandas performance warnings

# Global
simplefilter(action="ignore", category=pd.errors.PerformanceWarning)
tokenizer = AutoTokenizer.from_pretrained("google-bert/bert-base-uncased")
bert_model = AutoModel.from_pretrained("google-bert/bert-base-uncased")

# From running data-scraping and cleaning code
def get_data():
    with open('../Data/posts-cleaned.json', "r") as f:
        data = json.load(f)
    
    return data

def get_bert_embedding(sentence, tokenizer, model):
    inputs = tokenizer(sentence, return_tensors="pt", truncation=True, padding=True, max_length=256)
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).squeeze().numpy()

# Define the neural network
class ControversyNet(nn.Module):
    def __init__(self, input_size):
        super(ControversyNet, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_size, 128),     # Input to hidden layer
            nn.ReLU(),                      # Activation
            nn.Dropout(0.3),                # Dropout for regularization
            nn.Linear(128, 64),             # Hidden to another hidden layer
            nn.ReLU(),
            nn.Linear(64, 1)                # Output layer
        )
    
    def forward(self, x):
        return self.fc(x)

def main():
    # Obtain posts
    data_init = get_data()
    df = pd.DataFrame.from_dict(data_init)

    # --------------
    # Assemble data for training
    # --------------

    # Filter rows with controversy_score == [0] and sample 20% of them
    zero_controversy_sample = df.loc[df['controversy_score'].apply(lambda x: x == [0])].sample(frac=0.20, random_state=42)

    # Keep rows where controversy_score != [0]
    non_zero_controversy = df.loc[df['controversy_score'].apply(lambda x: x != [0])]

    # Combine the sampled [0] rows with the non-zero controversy rows
    df = pd.concat([non_zero_controversy, zero_controversy_sample], ignore_index=True)

    # Expand rows based on the controversy_score list
    df = df.explode('controversy_score', ignore_index=True)

    # Ensure the controversy_score is now numeric
    df['controversy_score'] = df['controversy_score'].astype(float)

    # --------------
    # Feature initialization
    # --------------

    # BERT Description Transformer
    df['description'] = df['description'].apply(lambda x: get_bert_embedding(x, tokenizer, bert_model))

    # List of categories to process (categories containing a list of strings)
    multi_cats = ['attributes', 'impacts', 'tags']

    # Iterate through each category
    for cat in multi_cats:
        # Flatten the list of strings in the current column into unique values
        all_unique_values = set([item for sublist in df[cat] for item in sublist])
        temp_df = pd.DataFrame()
        # For each unique value, create a new boolean column
        for value in all_unique_values:
            # Mark the presence (True/False) of the value in each row for this column
            temp_df[f"{cat}_{value}"] = df[cat].apply(lambda x: value in x)
        
        df = pd.concat([df, temp_df], axis=1)

    # One-hot encoding for the categories
    ohe_cats = ["language", "type", "severity", "quick_fix"]
    df = pd.get_dummies(df, columns=ohe_cats, drop_first=True)

    #with open("testing", 'w') as file:
    #    file.write(df.drop(columns=['controversy_score']).select_dtypes(include=['uint8', 'int64', 'float64', 'bool']).to_json(orient = "records"))

    # Combine BERT-transformed description, one-hot, multi-hot, and numeric features
    X = np.hstack([
        np.vstack(df['description'].to_list()),                                                                               # Semantic embeddings                                         
        df.drop(columns=['controversy_score']).select_dtypes(include=['uint8', 'int64', 'float64', 'bool']).values  # Encodings and numerics 
    ])

    # Aggregate controversy_score to a single value (mean)
    #df['aggregated_controversy_score'] = df['controversy_score'].apply(np.mean)
    df['aggregated_controversy_score'] = df['controversy_score']
    df['aggregated_controversy_score'] = np.sqrt(df['aggregated_controversy_score']) # Stretch values
    y = df['aggregated_controversy_score'].values

    # Define bin edges and labels
    #bin_edges = [0.0, 0.05, 0.15, 0.18, 0.21, 0.23, 0.26, 0.35, 1.0]
    #bin_labels = ['none', 'very low', 'low', 'low-medium', 'medium', 'medium-high', 'high', 'very high']
    df['aggregated_controversy_score'].hist(bins=50)
    plt.title('Distribution of controversy scores')
    plt.ylabel('Frequency')
    plt.xlabel('Controversy')
    bin_edges = [0.0, 0.15, 0.3, 0.5, 1.00]
    bin_labels = ['insignificant', 'low', 'medium', 'high']
    
    # Assign controversy scores to bins
    df['controversy_bin'] = pd.cut(
        df['aggregated_controversy_score'], bins=bin_edges, labels=bin_labels, include_lowest=True
    )

    # Encode bins as integers for model training
    label_encoder = LabelEncoder()
    df['controversy_bin_encoded'] = label_encoder.fit_transform(df['controversy_bin'])

    print(f"Label encoder classes: {label_encoder.classes_}")
    print(f"INFO: Distribution of target (encoded bins)... \n---\n {np.unique(df['controversy_bin_encoded'], return_counts=True)} \n---\n")
    print(f"Learning will utilize {len(df.index)} entries.")

    # --------------
    # Model training
    # --------------

    X = X.astype(np.float64)

    #print("TEST\n")
    #print(X.dtype)
    #print(">---<\n")
    #print(X.astype(np.float64))
    #print("TEST2\n")
    #print(df.dtypes)
    #print("TEST3\n")


    # Convert X and y to PyTorch tensors
    X_tensor = torch.from_numpy(X)
    y_tensor = torch.tensor(y, dtype=torch.float32)

    # Combine into a dataset
    dataset = TensorDataset(X_tensor, y_tensor)

    # Split dataset into training and testing sets
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

    # Define data loaders for batching
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    # Initialize model, loss function, and optimizer
    input_size = X.shape[1]
    model = ControversyNet(input_size)
    criterion = nn.MSELoss()  # Regression task, so use Mean Squared Error
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Training loop
    epochs = 50
    for epoch in range(epochs):
        model.train()                                               # Set model to training mode
        train_loss = 0
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()                                   # Clear gradients
            predictions = model(X_batch.to(torch.float32)).squeeze()   # Forward pass
            loss = criterion(predictions, y_batch)                  # Compute loss
            loss.backward()                                         # Backpropagation
            optimizer.step()                                        # Update weights
            train_loss += loss.item()
        
        train_loss /= len(train_loader)  # Average loss per batch
        print(f"Epoch {epoch + 1}/{epochs}, Train Loss: {train_loss:.4f}")

    # Evaluate the model
    model.eval()  # Set model to evaluation mode

    all_predictions = []
    all_targets = []

    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            predictions = model(X_batch.to(torch.float32)).squeeze()
            all_predictions.extend(predictions.tolist())
            all_targets.extend(y_batch.tolist())

    mae = mean_absolute_error(all_targets, all_predictions)
    print(f"Mean Absolute Error (MAE): {mae:.4f}")

    r2 = r2_score(all_targets, all_predictions)
    print(f"R² Score: {r2:.4f}")

    return

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train the model
    forest_model = RandomForestRegressor(n_estimators=500)
    forest_model.fit(X_train, y_train)

    # Make predictions on the test set
    y_pred = forest_model.predict(X_test)

    # Evaluate
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(f"Test Set MSE: {mse:.4f}")
    print(f"Test Set R2 Score: {r2:.4f}")

    # Bin the continuous predictions and true values
    y_test_binned = pd.cut(y_test, bins=bin_edges, labels=bin_labels, include_lowest=True)
    y_pred_binned = pd.cut(y_pred, bins=bin_edges, labels=bin_labels, include_lowest=True)
    y_train_binned = pd.cut(y_train, bins=bin_edges, labels=bin_labels, include_lowest=True)

    # Ensure the values are of type 'str' for classification evaluation
    y_test_binned = y_test_binned.astype(str)
    y_pred_binned = y_pred_binned.astype(str)
    y_train_binned = y_train_binned.astype(str)

    print("Class distribution in y_train \n:", pd.Series(y_train_binned).value_counts())
    print("Class distribution in y_pred \n:", pd.Series(y_pred_binned).value_counts())
    print("Class distribution in y_test \n:", pd.Series(y_test_binned).value_counts())

    # Evaluate classification performance based on the bins
    print("Classification Report for Binned Predictions:")
    print(classification_report(y_test_binned, y_pred_binned, target_names=bin_labels, labels=bin_labels, zero_division=0))

    # Calculate accuracy based on binned values
    accuracy = np.mean(y_test_binned == y_pred_binned)
    print(f"Accuracy based on binned predictions: {accuracy:.4f}")

main()
