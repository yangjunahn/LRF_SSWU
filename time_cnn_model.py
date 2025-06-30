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

class TimeCNN(nn.Module):
    def __init__(self, input_size, num_classes, num_channels=64):
        super(TimeCNN, self).__init__()
        self.input_size = input_size
        self.num_classes = num_classes
        
        # 1D Convolutional layers for temporal feature extraction
        self.conv1 = nn.Conv1d(input_size, num_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(num_channels, num_channels * 2, kernel_size=3, padding=1)
        self.conv3 = nn.Conv1d(num_channels * 2, num_channels * 4, kernel_size=3, padding=1)
        
        # Batch normalization
        self.bn1 = nn.BatchNorm1d(num_channels)
        self.bn2 = nn.BatchNorm1d(num_channels * 2)
        self.bn3 = nn.BatchNorm1d(num_channels * 4)
        
        # Pooling layers
        self.pool = nn.MaxPool1d(kernel_size=2, stride=1)
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        
        # Dropout for regularization
        self.dropout = nn.Dropout(0.3)
        
        # Fully connected layers
        self.fc1 = nn.Linear(num_channels * 4, num_channels * 2)
        self.fc2 = nn.Linear(num_channels * 2, num_classes)
        
        # Activation functions
        self.relu = nn.ReLU()
        
    def forward(self, x):
        # x shape: (batch_size, seq_len, input_size)
        # Transpose to (batch_size, input_size, seq_len) for 1D convolution
        x = x.transpose(1, 2)
        
        # First convolutional block
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.pool(x)
        x = self.dropout(x)
        
        # Second convolutional block
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.pool(x)
        x = self.dropout(x)
        
        # Third convolutional block
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu(x)
        x = self.global_pool(x)  # Global average pooling
        
        # Flatten
        x = x.view(x.size(0), -1)
        
        # Fully connected layers
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x

def load_training_data(data_dir='./training_data'):
    """Load training data from files."""
    X_train = np.load(os.path.join(data_dir, 'X_train.npy'))
    X_test = np.load(os.path.join(data_dir, 'X_test.npy'))
    y_train = np.load(os.path.join(data_dir, 'y_train.npy'))
    y_test = np.load(os.path.join(data_dir, 'y_test.npy'))
    
    with open(os.path.join(data_dir, 'label_encoder.pkl'), 'rb') as f:
        label_encoder = pickle.load(f)
    
    return X_train, X_test, y_train, y_test, label_encoder

def prepare_data_for_cnn(X, seq_length=7):
    """Reshape data for CNN: (samples, seq_length, features_per_step)"""
    # Reshape from (samples, 42) to (samples, 7, 6)
    # 42 = 7 features * 6 motion components
    return X.reshape(-1, seq_length, 6)

def train_model(model, train_loader, val_loader, num_epochs=10, learning_rate=0.001):
    """Training function for CNN model"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []
    
    for epoch in range(num_epochs):
        # Training
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            train_total += batch_y.size(0)
            train_correct += (predicted == batch_y).sum().item()
        
        # Validation
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                val_loss += loss.item()
                
                _, predicted = torch.max(outputs.data, 1)
                val_total += batch_y.size(0)
                val_correct += (predicted == batch_y).sum().item()
        
        # Calculate metrics
        train_losses.append(train_loss / len(train_loader))
        val_losses.append(val_loss / len(val_loader))
        train_accuracies.append(train_correct / train_total)
        val_accuracies.append(val_correct / val_total)
        
        print(f'Epoch [{epoch+1}/{num_epochs}], '
              f'Train Loss: {train_losses[-1]:.4f}, Val Loss: {val_losses[-1]:.4f}, '
              f'Train Acc: {train_accuracies[-1]:.4f}, Val Acc: {val_accuracies[-1]:.4f}')
    
    return train_losses, val_losses, train_accuracies, val_accuracies

def evaluate_model(model, test_loader, label_encoder):
    """Evaluate the trained model"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.eval()
    
    all_predictions = []
    all_labels = []
    all_probabilities = []
    
    with torch.no_grad():
        for batch_X, batch_y in test_loader:
            batch_X = batch_X.to(device)
            outputs = model(batch_X)
            probabilities = torch.softmax(outputs, dim=1)
            _, predicted = torch.max(outputs.data, 1)
            
            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(batch_y.numpy())
            all_probabilities.extend(probabilities.cpu().numpy())
    
    # Calculate accuracy
    accuracy = accuracy_score(all_labels, all_predictions)
    
    # Classification report
    report = classification_report(all_labels, all_predictions, 
                                 target_names=label_encoder.classes_, output_dict=True)
    
    return accuracy, report, all_predictions, all_labels, all_probabilities

def plot_training_history(train_losses, val_losses, train_accuracies, val_accuracies, save_path='results/training_history_cnn.png'):
    """Plot training history for CNN"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Plot loss
    ax1.plot(train_losses, label='Training Loss')
    ax1.plot(val_losses, label='Validation Loss')
    ax1.set_title('Model Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True)
    
    # Plot accuracy
    ax2.plot(train_accuracies, label='Training Accuracy')
    ax2.plot(val_accuracies, label='Validation Accuracy')
    ax2.set_title('Model Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Training history saved to {save_path}")

def compare_models(bilstm_results, cnn_results, save_path='results/model_comparison.png'):
    """Compare Bi-LSTM and CNN model performances"""
    
    # Extract accuracies
    bilstm_acc = bilstm_results['test_accuracy']
    cnn_acc = cnn_results['test_accuracy']
    
    # Create comparison plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Overall accuracy comparison
    models = ['Bi-LSTM', 'Time-CNN']
    accuracies = [bilstm_acc, cnn_acc]
    colors = ['skyblue', 'lightcoral']
    
    bars = ax1.bar(models, accuracies, color=colors, alpha=0.7)
    ax1.set_title('Model Accuracy Comparison')
    ax1.set_ylabel('Test Accuracy')
    ax1.set_ylim(0, 1)
    
    # Add value labels on bars
    for bar, acc in zip(bars, accuracies):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{acc:.3f}', ha='center', va='bottom', fontweight='bold')
    
    # Per-class F1-score comparison
    bilstm_f1 = [bilstm_results['classification_report'][cls]['f1-score'] 
                 for cls in bilstm_results['classification_report'].keys() 
                 if cls not in ['accuracy', 'macro avg', 'weighted avg']]
    cnn_f1 = [cnn_results['classification_report'][cls]['f1-score'] 
              for cls in cnn_results['classification_report'].keys() 
              if cls not in ['accuracy', 'macro avg', 'weighted avg']]
    
    class_names = [cls for cls in bilstm_results['classification_report'].keys() 
                   if cls not in ['accuracy', 'macro avg', 'weighted avg']]
    
    x = np.arange(len(class_names))
    width = 0.35
    
    ax2.bar(x - width/2, bilstm_f1, width, label='Bi-LSTM', alpha=0.7)
    ax2.bar(x + width/2, cnn_f1, width, label='Time-CNN', alpha=0.7)
    
    ax2.set_title('Per-Class F1-Score Comparison')
    ax2.set_xlabel('Classes')
    ax2.set_ylabel('F1-Score')
    ax2.set_xticks(x)
    ax2.set_xticklabels(class_names, rotation=45, ha='right')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Model comparison saved to {save_path}")

def main():
    """Main function to train and evaluate Time-CNN model"""
    print("Loading training data...")
    X_train, X_test, y_train, y_test, label_encoder = load_training_data()
    
    print(f"Training data shape: {X_train.shape}")
    print(f"Test data shape: {X_test.shape}")
    print(f"Number of classes: {len(label_encoder.classes_)}")
    print(f"Classes: {label_encoder.classes_}")
    
    # Prepare data for CNN
    print("\nPreparing data for CNN...")
    X_train_cnn = prepare_data_for_cnn(X_train)
    X_test_cnn = prepare_data_for_cnn(X_test)
    
    print(f"CNN training data shape: {X_train_cnn.shape}")
    print(f"CNN test data shape: {X_test_cnn.shape}")
    
    # Convert to PyTorch tensors
    X_train_tensor = torch.FloatTensor(X_train_cnn)
    y_train_tensor = torch.LongTensor(y_train)
    X_test_tensor = torch.FloatTensor(X_test_cnn)
    y_test_tensor = torch.LongTensor(y_test)
    
    # Create data loaders
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
    
    # Split training data for validation
    train_size = int(0.8 * len(train_dataset))
    val_size = len(train_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(train_dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    # Create CNN model
    print("\nCreating Time-CNN model...")
    input_size = 6  # 6 motion components
    num_classes = len(label_encoder.classes_)
    num_channels = 64
    
    model = TimeCNN(input_size, num_classes, num_channels)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Train model
    print("\nTraining Time-CNN model...")
    train_losses, val_losses, train_accuracies, val_accuracies = train_model(
        model, train_loader, val_loader, num_epochs=10
    )
    
    # Evaluate model
    print("\nEvaluating Time-CNN model...")
    accuracy, report, predictions, true_labels, probabilities = evaluate_model(
        model, test_loader, label_encoder
    )
    
    print(f"\nTime-CNN Test Accuracy: {accuracy:.4f}")
    print("\nClassification Report:")
    print(classification_report(true_labels, predictions, target_names=label_encoder.classes_))
    
    # Plot training history
    plot_training_history(train_losses, val_losses, train_accuracies, val_accuracies)
    
    # Save model
    torch.save(model.state_dict(), 'results/time_cnn_model.pth')
    print("Time-CNN model saved to results/time_cnn_model.pth")
    
    # Save results
    cnn_results = {
        'test_accuracy': float(accuracy),
        'classification_report': report,
        'model_info': {
            'input_size': input_size,
            'num_channels': num_channels,
            'num_classes': num_classes,
            'classes': label_encoder.classes_.tolist()
        },
        'training_history': {
            'train_losses': train_losses,
            'val_losses': val_losses,
            'train_accuracies': train_accuracies,
            'val_accuracies': val_accuracies
        }
    }
    
    with open('results/cnn_model_results.json', 'w') as f:
        json.dump(cnn_results, f, indent=2)
    
    # Load Bi-LSTM results for comparison
    try:
        with open('results/model_results_pytorch.json', 'r') as f:
            bilstm_results = json.load(f)
        
        # Compare models
        compare_models(bilstm_results, cnn_results)
        
        print(f"\n" + "="*60)
        print("MODEL COMPARISON")
        print("="*60)
        print(f"Bi-LSTM Accuracy: {bilstm_results['test_accuracy']:.4f}")
        print(f"Time-CNN Accuracy: {accuracy:.4f}")
        print(f"Improvement: {accuracy - bilstm_results['test_accuracy']:.4f}")
        print("="*60)
        
    except FileNotFoundError:
        print("Bi-LSTM results not found for comparison")
    
    print("\n" + "="*50)
    print("TIME-CNN TRAINING COMPLETED")
    print("="*50)
    print(f"Test Accuracy: {accuracy:.4f}")
    print(f"Model saved as: results/time_cnn_model.pth")
    print(f"Results saved as: results/cnn_model_results.json")
    print(f"Plots saved as: results/training_history_cnn.png")
    print("="*50)

if __name__ == "__main__":
    main() 