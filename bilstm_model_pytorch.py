import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pickle
import json
import os
from sklearn.metrics import classification_report, accuracy_score
import matplotlib.pyplot as plt

class SimpleBiLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(SimpleBiLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_classes = num_classes
        
        # Bi-LSTM layer
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True, bidirectional=True)
        
        # Output layer
        self.fc = nn.Linear(hidden_size * 2, num_classes)  # *2 for bidirectional
        
    def forward(self, x):
        # x shape: (batch_size, seq_len, input_size)
        lstm_out, _ = self.lstm(x)
        
        # Take the last output from both directions
        last_output = lstm_out[:, -1, :]  # (batch_size, hidden_size * 2)
        
        # Classification
        output = self.fc(last_output)
        return output

def load_training_data(data_dir='./training_data'):
    """Load training data from files."""
    X_train = np.load(os.path.join(data_dir, 'X_train.npy'))
    X_test = np.load(os.path.join(data_dir, 'X_test.npy'))
    y_train = np.load(os.path.join(data_dir, 'y_train.npy'))
    y_test = np.load(os.path.join(data_dir, 'y_test.npy'))
    
    with open(os.path.join(data_dir, 'label_encoder.pkl'), 'rb') as f:
        label_encoder = pickle.load(f)
    
    return X_train, X_test, y_train, y_test, label_encoder

def prepare_data_for_lstm(X, seq_length=7):
    """Reshape data for LSTM: (samples, seq_length, features_per_step)"""
    # Reshape from (samples, 42) to (samples, 7, 6)
    # 42 = 7 features * 6 motion components
    return X.reshape(-1, seq_length, 6)

def train_model(model, train_loader, val_loader, num_epochs=10, learning_rate=0.001):
    """Simple training function"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    train_losses = []
    val_losses = []
    
    for epoch in range(num_epochs):
        # Training
        model.train()
        train_loss = 0.0
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                val_loss += loss.item()
        
        train_losses.append(train_loss / len(train_loader))
        val_losses.append(val_loss / len(val_loader))
        
        print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_losses[-1]:.4f}, Val Loss: {val_losses[-1]:.4f}')
    
    return train_losses, val_losses

def evaluate_model(model, test_loader, label_encoder):
    """Evaluate the trained model"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.eval()
    
    all_predictions = []
    all_labels = []
    
    with torch.no_grad():
        for batch_X, batch_y in test_loader:
            batch_X = batch_X.to(device)
            outputs = model(batch_X)
            _, predicted = torch.max(outputs.data, 1)
            
            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(batch_y.numpy())
    
    # Calculate accuracy
    accuracy = accuracy_score(all_labels, all_predictions)
    
    # Classification report
    report = classification_report(all_labels, all_predictions, 
                                 target_names=label_encoder.classes_, output_dict=True)
    
    return accuracy, report, all_predictions, all_labels

def main():
    """Main function to train and test the simple Bi-LSTM model"""
    print("Loading training data...")
    X_train, X_test, y_train, y_test, label_encoder = load_training_data()
    
    print(f"Training data shape: {X_train.shape}")
    print(f"Test data shape: {X_test.shape}")
    print(f"Number of classes: {len(label_encoder.classes_)}")
    print(f"Classes: {label_encoder.classes_}")
    
    # Prepare data for LSTM
    print("\nPreparing data for LSTM...")
    X_train_lstm = prepare_data_for_lstm(X_train)
    X_test_lstm = prepare_data_for_lstm(X_test)
    
    print(f"LSTM training data shape: {X_train_lstm.shape}")
    print(f"LSTM test data shape: {X_test_lstm.shape}")
    
    # Convert to PyTorch tensors
    X_train_tensor = torch.FloatTensor(X_train_lstm)
    y_train_tensor = torch.LongTensor(y_train)
    X_test_tensor = torch.FloatTensor(X_test_lstm)
    y_test_tensor = torch.LongTensor(y_test)
    
    # Create data loaders
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    # Split training data for validation
    train_size = int(0.8 * len(train_dataset))
    val_size = len(train_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(train_dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    
    # Create model
    print("\nCreating Bi-LSTM model...")
    input_size = 6  # 6 motion components
    hidden_size = 64
    num_classes = len(label_encoder.classes_)
    
    model = SimpleBiLSTM(input_size, hidden_size, num_classes)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Train model
    print("\nTraining model...")
    train_losses, val_losses = train_model(model, train_loader, val_loader, num_epochs=10)
    
    # Evaluate model
    print("\nEvaluating model...")
    accuracy, report, predictions, true_labels = evaluate_model(model, test_loader, label_encoder)
    
    print(f"\nTest Accuracy: {accuracy:.4f}")
    print("\nClassification Report:")
    print(classification_report(true_labels, predictions, target_names=label_encoder.classes_))
    
    # Plot training history
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.title('Training History')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig('results/training_history_pytorch.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Training history saved to results/training_history_pytorch.png")
    
    # Save model
    torch.save(model.state_dict(), 'results/bilstm_model_pytorch.pth')
    print("Model saved to results/bilstm_model_pytorch.pth")
    
    # Save results
    results = {
        'test_accuracy': float(accuracy),
        'classification_report': report,
        'model_info': {
            'input_size': input_size,
            'hidden_size': hidden_size,
            'num_classes': num_classes,
            'classes': label_encoder.classes_.tolist()
        }
    }
    
    with open('results/model_results_pytorch.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print("\n" + "="*50)
    print("TRAINING COMPLETED")
    print("="*50)
    print(f"Test Accuracy: {accuracy:.4f}")
    print(f"Model saved as: results/bilstm_model_pytorch.pth")
    print(f"Results saved as: results/model_results_pytorch.json")
    print(f"Plot saved as: results/training_history_pytorch.png")
    print("="*50)

if __name__ == "__main__":
    main() 