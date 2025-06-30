import numpy as np
import pickle
import json
import os

def load_training_database(data_dir='./training_data'):
    """Load the training database from files."""
    
    # Load numpy arrays
    X_train = np.load(os.path.join(data_dir, 'X_train.npy'))
    X_test = np.load(os.path.join(data_dir, 'X_test.npy'))
    y_train = np.load(os.path.join(data_dir, 'y_train.npy'))
    y_test = np.load(os.path.join(data_dir, 'y_test.npy'))
    
    # Load label encoder
    with open(os.path.join(data_dir, 'label_encoder.pkl'), 'rb') as f:
        label_encoder = pickle.load(f)
    
    # Load metadata
    with open(os.path.join(data_dir, 'metadata_train.json'), 'r') as f:
        metadata_train = json.load(f)
    
    with open(os.path.join(data_dir, 'metadata_test.json'), 'r') as f:
        metadata_test = json.load(f)
    
    # Load dataset info
    with open(os.path.join(data_dir, 'dataset_info.json'), 'r') as f:
        dataset_info = json.load(f)
    
    # Load feature names
    with open(os.path.join(data_dir, 'feature_names.json'), 'r') as f:
        feature_names = json.load(f)
    
    return {
        'X_train': X_train,
        'X_test': X_test,
        'y_train': y_train,
        'y_test': y_test,
        'label_encoder': label_encoder,
        'metadata_train': metadata_train,
        'metadata_test': metadata_test,
        'dataset_info': dataset_info,
        'feature_names': feature_names
    }

def analyze_training_database(data):
    """Analyze the training database."""
    
    print("="*60)
    print("TRAINING DATABASE ANALYSIS")
    print("="*60)
    
    # Basic information
    print(f"Training samples: {data['X_train'].shape[0]}")
    print(f"Test samples: {data['X_test'].shape[0]}")
    print(f"Features per sample: {data['X_train'].shape[1]}")
    print(f"Number of categories: {len(data['dataset_info']['categories'])}")
    
    # Category distribution
    print(f"\nCategories: {data['dataset_info']['categories']}")
    
    # Label distribution in training set
    unique_train, counts_train = np.unique(data['y_train'], return_counts=True)
    print(f"\nTraining set label distribution:")
    for label_idx, count in zip(unique_train, counts_train):
        category_name = data['label_encoder'].inverse_transform([label_idx])[0]
        print(f"  {category_name}: {count} samples")
    
    # Label distribution in test set
    unique_test, counts_test = np.unique(data['y_test'], return_counts=True)
    print(f"\nTest set label distribution:")
    for label_idx, count in zip(unique_test, counts_test):
        category_name = data['label_encoder'].inverse_transform([label_idx])[0]
        print(f"  {category_name}: {count} samples")
    
    # Feature statistics
    print(f"\nFeature statistics (training set):")
    print(f"  Mean: {np.mean(data['X_train'], axis=0)[:5]}...")  # First 5 features
    print(f"  Std:  {np.std(data['X_train'], axis=0)[:5]}...")   # First 5 features
    print(f"  Min:  {np.min(data['X_train'], axis=0)[:5]}...")   # First 5 features
    print(f"  Max:  {np.max(data['X_train'], axis=0)[:5]}...")   # First 5 features
    
    # Sample metadata
    print(f"\nSample metadata (first training sample):")
    print(f"  Original file: {data['metadata_train'][0]['original_file']}")
    print(f"  Category: {data['metadata_train'][0]['category']}")
    print(f"  Window: {data['metadata_train'][0]['window_start']} - {data['metadata_train'][0]['window_end']}")
    print(f"  Total samples in file: {data['metadata_train'][0]['total_samples']}")
    
    # Feature names
    print(f"\nFeature names (first 10):")
    for i, name in enumerate(data['feature_names'][:10]):
        print(f"  {i+1:2d}. {name}")
    print(f"  ... and {len(data['feature_names'])-10} more features")
    
    print("="*60)

def main():
    """Main function to load and analyze training database."""
    
    # Load training database
    print("Loading training database...")
    data = load_training_database()
    
    # Analyze the data
    analyze_training_database(data)
    
    # Example of how to use the data for training
    print(f"\nExample usage for model training:")
    print(f"X_train shape: {data['X_train'].shape}")
    print(f"y_train shape: {data['y_train'].shape}")
    print(f"X_test shape: {data['X_test'].shape}")
    print(f"y_test shape: {data['y_test'].shape}")
    
    # Show how to decode labels
    print(f"\nLabel encoding example:")
    print(f"Encoded labels: {data['y_train'][:5]}")
    decoded_labels = data['label_encoder'].inverse_transform(data['y_train'][:5])
    print(f"Decoded labels: {decoded_labels}")

if __name__ == "__main__":
    main() 