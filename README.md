# Motion Data Classification: Bi-LSTM vs Time-CNN Comparison

## 📋 Project Overview

This project implements and compares two deep learning models for classifying motion data patterns in wave-motion systems. The goal is to predict motion categories based on extracted features from time-series motion data.

## 🎯 Problem Statement

Given motion data from different wave conditions (angles: 090°, 135°, 180°; periods: 7.5s, 9.5s, 11.5s; wave heights: 3.5m), the task is to classify motion patterns into 9 distinct categories using machine learning approaches.

## 📊 Dataset Description

The `orgin_dataset/` folder contains motion and wave data files:
- **Motion Data**: `{angle}_Tm_{period}_HS_{height}.out` (e.g., `090_Tm_7.5_HS_3.5.out`)
- **Wave Data**: `Wave_{angle}_Tm_{period}_HS_{height}.out` (e.g., `Wave_090_Tm_7.5_HS_3.5.out`)

### Data Structure
Each motion file contains 7 columns:
1. **Time (s)**: Time series
2. **Surge (m)**: Forward/backward motion
3. **Sway (m)**: Side-to-side motion  
4. **Heave (m)**: Up/down motion
5. **Roll (deg)**: Rotation around longitudinal axis
6. **Pitch (deg)**: Rotation around transverse axis
7. **Yaw (deg)**: Rotation around vertical axis

### Categories
9 unique motion categories based on wave conditions:
- `090_Tm_7.5_HS_3.5`, `090_Tm_9.5_HS_3.5`, `090_Tm_11.5_HS_3.5`
- `135_Tm_7.5_HS_3.5`, `135_Tm_9.5_HS_3.5`, `135_Tm_11.5_HS_3.5`
- `180_Tm_7.5_HS_3.5`, `180_Tm_9.5_HS_3.5`, `180_Tm_11.5_HS_3.5`

## 🏗️ Methodology

### 1. Data Preprocessing
- **Feature Extraction**: Sliding window approach (100 time steps, 50% overlap)
- **Statistical Features**: Mean, std, max, min, range for each motion component
- **Frequency Features**: Dominant frequency index and magnitude (FFT)
- **Total Features**: 42 features per sample (7 features × 6 motion components)

### 2. Model Architectures

#### Bi-LSTM Model
- **Architecture**: Bidirectional LSTM with dense layers
- **Input**: Reshaped to (7 timesteps, 6 motion components)
- **Layers**: 2 Bi-LSTM layers → Dense layers → Softmax output
- **Parameters**: 38,025 trainable parameters

#### Time-CNN Model
- **Architecture**: 1D Convolutional Neural Network
- **Input**: Same as Bi-LSTM
- **Layers**: 3 Conv1D blocks → Global pooling → Dense layers
- **Parameters**: 159,433 trainable parameters

### 3. Training Process
- **Dataset Split**: 80% training, 20% testing
- **Training Samples**: 14,392
- **Test Samples**: 3,599
- **Epochs**: 10 for both models
- **Optimizer**: Adam (learning rate: 0.001)
- **Loss Function**: Cross-entropy

## 📈 Results

### Model Performance Comparison

| Model | Test Accuracy | Parameters | Training Time |
|-------|---------------|------------|---------------|
| **Bi-LSTM** | **66.07%** | 38,025 | ~2 minutes |
| **Time-CNN** | 63.80% | 159,433 | ~3 minutes |

### Key Findings
- **Bi-LSTM outperforms Time-CNN** by 2.27 percentage points
- **Best performing category**: `180_Tm_7.5_HS_3.5` (92% F1-score for both models)
- **Worst performing category**: 
  - Bi-LSTM: `090_Tm_9.5_HS_3.5` (50% F1-score)
  - Time-CNN: `090_Tm_11.5_HS_3.5` (30% F1-score)

### Why Bi-LSTM Performed Better
1. **Sequential Nature**: Motion data has temporal dependencies that LSTM captures better
2. **Bidirectional Processing**: Captures both forward and backward temporal patterns
3. **Parameter Efficiency**: Fewer parameters but better performance
4. **Memory Mechanism**: LSTM's memory cells are better suited for sequential data

## 🚀 Setup and Installation

### Prerequisites
- Python 3.8+
- PyTorch 1.12+
- CUDA (optional, for GPU acceleration)

### Installation Steps

1. **Clone the repository:**
   ```bash
   git clone https://github.com/yangjunahn/LRF_SSWU.git
   cd LRF_SSWU
   ```

2. **Create virtual environment:**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Linux/Mac
   # or
   venv\Scripts\activate  # On Windows
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

## 📁 Project Structure

```
LRF_SSWU/
├── orgin_dataset/              # Original motion and wave data
│   ├── 090_Tm_*.out           # Motion data files
│   ├── 135_Tm_*.out
│   ├── 180_Tm_*.out
│   └── Wave_*.out             # Wave data files
├── training_data/              # Processed training data
│   ├── X_train.npy            # Training features
│   ├── X_test.npy             # Test features
│   ├── y_train.npy            # Training labels
│   ├── y_test.npy             # Test labels
│   └── label_encoder.pkl      # Label encoder
├── create_training_database.py # Data preprocessing script
├── bilstm_model_pytorch.py    # Bi-LSTM model implementation
├── time_cnn_model.py          # Time-CNN model implementation
├── visualize_data.py          # Data visualization script
├── load_training_data.py      # Data loading utilities
├── bilstm_model_pytorch.pth   # Trained Bi-LSTM model
├── time_cnn_model.pth         # Trained Time-CNN model
├── model_comparison.png       # Model performance comparison
├── training_history_*.png     # Training curves
├── requirements.txt           # Python dependencies
└── README.md                  # This file
```

## 🎮 Usage

### 1. Data Visualization
Visualize the original motion and wave data:
```bash
python visualize_data.py
```

### 2. Create Training Database
Process raw data into training format:
```bash
python create_training_database.py
```

### 3. Train Bi-LSTM Model
```bash
python bilstm_model_pytorch.py
```

### 4. Train Time-CNN Model
```bash
python time_cnn_model.py
```

### 5. Load and Analyze Results
```bash
python load_training_data.py
```

## 📊 Generated Outputs

- **Trained Models**: `bilstm_model_pytorch.pth`, `time_cnn_model.pth`
- **Results**: `model_results_pytorch.json`, `cnn_model_results.json`
- **Visualizations**: 
  - `wave_motion_plot.png` - Original data visualization
  - `training_history_pytorch.png` - Bi-LSTM training curves
  - `training_history_cnn.png` - Time-CNN training curves
  - `model_comparison.png` - Performance comparison

## 🔧 Dependencies

- **PyTorch**: Deep learning framework
- **NumPy, Pandas**: Data manipulation
- **Scikit-learn**: Machine learning utilities
- **Matplotlib, Seaborn**: Data visualization
- **Pickle, JSON**: Data serialization

## 📝 Future Work

1. **Hyperparameter Optimization**: Grid search for optimal model parameters
2. **Ensemble Methods**: Combine Bi-LSTM and Time-CNN predictions
3. **Attention Mechanisms**: Implement attention layers for better feature selection
4. **Real-time Classification**: Deploy models for real-time motion prediction
5. **Additional Datasets**: Test on different wave conditions and vessel types

## 👨‍💻 Author

**Yangjun Ahn** - Motion Data Classification Research

## 📄 License

This project is open source and available under the [MIT License](LICENSE).

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

---

**Note**: This project demonstrates the effectiveness of sequential models (Bi-LSTM) over convolutional models (Time-CNN) for motion pattern classification in wave-motion systems.
