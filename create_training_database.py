import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import pickle
import json
from collections import defaultdict

def extract_category_from_filename(filename):
    """
    Extract category from filename.
    Example: '090_Tm_7.5_HS_3.5.out' -> '090_Tm_7.5_HS_3.5'
    """
    # Remove file extension
    name = filename.replace('.out', '')
    # Return the full name as the category (including the angle)
    return name

def read_data_file(file_path):
    """Read data file with header line"""
    with open(file_path, 'r') as file:
        lines = file.readlines()
    
    # Skip the header line and read data
    data_lines = lines[1:]
    data = []
    for line in data_lines:
        values = [float(v) for v in line.split()]
        data.append(values)
    
    return np.array(data)

def extract_features_from_motion_data(motion_data, window_size=100, overlap=0.5):
    """
    Extract features from motion data using sliding windows.
    
    Args:
        motion_data: numpy array of motion data (time, surge, sway, heave, roll, pitch, yaw)
        window_size: number of time steps in each window
        overlap: overlap ratio between consecutive windows (0-1)
    
    Returns:
        features: numpy array of extracted features
    """
    features = []
    step_size = int(window_size * (1 - overlap))
    
    for i in range(0, len(motion_data) - window_size + 1, step_size):
        window = motion_data[i:i + window_size]
        
        # Extract statistical features for each motion component
        window_features = []
        
        # For each motion component (surge, sway, heave, roll, pitch, yaw)
        for col in range(1, 7):  # Skip time column (col 0)
            motion_series = window[:, col]
            
            # Statistical features
            mean_val = np.mean(motion_series)
            std_val = np.std(motion_series)
            max_val = np.max(motion_series)
            min_val = np.min(motion_series)
            range_val = max_val - min_val
            
            # Frequency domain features (FFT)
            fft_vals = np.fft.fft(motion_series)
            fft_magnitude = np.abs(fft_vals)
            dominant_freq_idx = np.argmax(fft_magnitude[1:len(fft_magnitude)//2]) + 1
            dominant_freq_magnitude = fft_magnitude[dominant_freq_idx]
            
            # Add features for this component
            window_features.extend([
                mean_val, std_val, max_val, min_val, range_val,
                dominant_freq_idx, dominant_freq_magnitude
            ])
        
        features.append(window_features)
    
    return np.array(features)

def create_training_database(dataset_path='./orgin_dataset', 
                           window_size=100, 
                           overlap=0.5,
                           test_size=0.2,
                           random_state=42):
    """
    Create training database from motion and wave data files.
    
    Args:
        dataset_path: path to the dataset folder
        window_size: number of time steps in each window
        overlap: overlap ratio between consecutive windows
        test_size: fraction of data to use for testing
        random_state: random seed for reproducibility
    
    Returns:
        dict: training database with features, labels, and metadata
    """
    
    print("Creating training database...")
    print(f"Dataset path: {dataset_path}")
    print(f"Window size: {window_size}")
    print(f"Overlap: {overlap}")
    
    # Get all files in the dataset directory
    all_files = os.listdir(dataset_path)
    
    # Separate motion files and wave files
    motion_files = [f for f in all_files if f.endswith('.out') and not f.startswith('Wave_')]
    wave_files = [f for f in all_files if f.endswith('.out') and f.startswith('Wave_')]
    
    print(f"Found {len(motion_files)} motion files and {len(wave_files)} wave files")
    
    # Create category mapping
    categories = []
    category_to_files = defaultdict(list)
    
    for motion_file in motion_files:
        category = extract_category_from_filename(motion_file)
        categories.append(category)
        category_to_files[category].append(motion_file)
    
    categories = list(set(categories))  # Remove duplicates
    print(f"Found {len(categories)} unique categories: {categories}")
    
    # Process each category
    all_features = []
    all_labels = []
    all_metadata = []
    
    for category in categories:
        print(f"\nProcessing category: {category}")
        
        # Find all motion files for this category
        category_motion_files = [f for f in motion_files 
                               if extract_category_from_filename(f) == category]
        
        for motion_file in category_motion_files:
            print(f"  Processing {motion_file}")
            
            # Read motion data
            motion_data_path = os.path.join(dataset_path, motion_file)
            motion_data = read_data_file(motion_data_path)
            
            # Extract features
            features = extract_features_from_motion_data(motion_data, window_size, overlap)
            
            # Create labels (all samples from this file belong to the same category)
            labels = [category] * len(features)
            
            # Create metadata
            metadata = [{
                'original_file': motion_file,
                'category': category,
                'window_start': i * int(window_size * (1 - overlap)),
                'window_end': i * int(window_size * (1 - overlap)) + window_size,
                'total_samples': len(motion_data)
            } for i in range(len(features))]
            
            all_features.append(features)
            all_labels.extend(labels)
            all_metadata.extend(metadata)
    
    # Combine all features
    X = np.vstack(all_features)
    y = np.array(all_labels)
    
    print(f"\nTotal samples: {len(X)}")
    print(f"Feature dimension: {X.shape[1]}")
    print(f"Number of categories: {len(set(y))}")
    
    # Encode labels
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    
    # Split into train and test sets
    X_train, X_test, y_train, y_test, metadata_train, metadata_test = train_test_split(
        X, y_encoded, all_metadata, test_size=test_size, 
        random_state=random_state, stratify=y_encoded
    )
    
    # Create training database
    training_database = {
        'X_train': X_train,
        'X_test': X_test,
        'y_train': y_train,
        'y_test': y_test,
        'metadata_train': metadata_train,
        'metadata_test': metadata_test,
        'label_encoder': label_encoder,
        'categories': categories,
        'feature_names': generate_feature_names(),
        'dataset_info': {
            'total_samples': len(X),
            'train_samples': len(X_train),
            'test_samples': len(X_test),
            'n_features': X.shape[1],
            'n_categories': len(categories),
            'window_size': window_size,
            'overlap': overlap,
            'categories': categories
        }
    }
    
    return training_database

def generate_feature_names():
    """Generate feature names for the extracted features."""
    motion_components = ['surge', 'sway', 'heave', 'roll', 'pitch', 'yaw']
    feature_types = ['mean', 'std', 'max', 'min', 'range', 'dominant_freq_idx', 'dominant_freq_magnitude']
    
    feature_names = []
    for component in motion_components:
        for feature_type in feature_types:
            feature_names.append(f"{component}_{feature_type}")
    
    return feature_names

def save_training_database(training_database, output_dir='./training_data'):
    """Save training database to files."""
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Save numpy arrays
    np.save(os.path.join(output_dir, 'X_train.npy'), training_database['X_train'])
    np.save(os.path.join(output_dir, 'X_test.npy'), training_database['X_test'])
    np.save(os.path.join(output_dir, 'y_train.npy'), training_database['y_train'])
    np.save(os.path.join(output_dir, 'y_test.npy'), training_database['y_test'])
    
    # Save label encoder
    with open(os.path.join(output_dir, 'label_encoder.pkl'), 'wb') as f:
        pickle.dump(training_database['label_encoder'], f)
    
    # Save metadata
    with open(os.path.join(output_dir, 'metadata_train.json'), 'w') as f:
        json.dump(training_database['metadata_train'], f, indent=2)
    
    with open(os.path.join(output_dir, 'metadata_test.json'), 'w') as f:
        json.dump(training_database['metadata_test'], f, indent=2)
    
    # Save dataset info
    with open(os.path.join(output_dir, 'dataset_info.json'), 'w') as f:
        json.dump(training_database['dataset_info'], f, indent=2)
    
    # Save feature names
    with open(os.path.join(output_dir, 'feature_names.json'), 'w') as f:
        json.dump(training_database['feature_names'], f, indent=2)
    
    print(f"\nTraining database saved to: {output_dir}")
    print(f"Files created:")
    print(f"  - X_train.npy: Training features")
    print(f"  - X_test.npy: Test features")
    print(f"  - y_train.npy: Training labels")
    print(f"  - y_test.npy: Test labels")
    print(f"  - label_encoder.pkl: Label encoder")
    print(f"  - metadata_train.json: Training metadata")
    print(f"  - metadata_test.json: Test metadata")
    print(f"  - dataset_info.json: Dataset information")
    print(f"  - feature_names.json: Feature names")

def main():
    """Main function to create and save training database."""
    
    # Create training database
    training_database = create_training_database(
        dataset_path='./orgin_dataset',
        window_size=100,  # 100 time steps per window
        overlap=0.5,      # 50% overlap between windows
        test_size=0.2,    # 20% for testing
        random_state=42   # For reproducibility
    )
    
    # Save training database
    save_training_database(training_database, output_dir='./training_data')
    
    # Print summary
    print("\n" + "="*50)
    print("TRAINING DATABASE SUMMARY")
    print("="*50)
    print(f"Total samples: {training_database['dataset_info']['total_samples']}")
    print(f"Training samples: {training_database['dataset_info']['train_samples']}")
    print(f"Test samples: {training_database['dataset_info']['test_samples']}")
    print(f"Features per sample: {training_database['dataset_info']['n_features']}")
    print(f"Categories: {training_database['dataset_info']['n_categories']}")
    print(f"Categories: {training_database['dataset_info']['categories']}")
    print("="*50)

if __name__ == "__main__":
    main() 