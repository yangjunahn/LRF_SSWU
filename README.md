# Wave Motion Analysis and Bi-LSTM Model

This project analyzes wave and motion data using data visualization and implements a Bi-LSTM model for motion prediction.

## Dataset

The `orgin_dataset/` folder contains wave and motion data files:
- `Wave_*.out`: Wave elevation data
- `*_Tm_*_HS_*.out`: Motion data (surge, sway, heave, roll, pitch, yaw)

## Setup

1. **Create a virtual environment (recommended):**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Linux/Mac
   # or
   venv\Scripts\activate  # On Windows
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Data Visualization
Run the visualization script to plot wave elevation and motion data:
```bash
python visualize_data.py
```

### Future Development
- Bi-LSTM model implementation
- Training data preparation
- Model training and evaluation

## Project Structure
```
├── visualize_data.py      # Data visualization script
├── requirements.txt       # Python dependencies
├── README.md             # Project documentation
├── orgin_dataset/        # Original dataset files
└── venv/                 # Virtual environment (created locally)
```

## Dependencies

- **pandas, numpy**: Data manipulation and analysis
- **matplotlib, seaborn, plotly**: Data visualization
- **tensorflow**: Deep learning framework for Bi-LSTM
- **scikit-learn, scipy**: Machine learning utilities
- **jupyter**: Interactive development (optional) 