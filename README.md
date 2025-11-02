# EPL Match Predictor

A machine learning system that predicts English Premier League (EPL) match outcomes using historical data and team performance metrics.

## Overview

This project uses a Random Forest classifier to predict EPL match results (Home Win, Away Win, or Draw) based on:
- Team form (last 5 matches performance)
- Season-to-date averages for shots on target, corners, possession, and chances
- Historical match data from 2020-2023

## Features

- **Temporal Data Processing**: Handles mixed date formats and chronological ordering
- **Feature Engineering**: Creates meaningful predictive features from raw match data
- **Model Evaluation**: Provides comprehensive accuracy metrics and classification reports
- **Robust Date Handling**: Supports both ordinal ("28th May 2023") and numerical ("31/10/2020") date formats

## Project Structure

```
match_predictor/
├── backend/
│   └── predictor.py          # Main prediction model and evaluation script
├── data/
│   └── EPL_2020_2023.csv     # Historical EPL match data
├── frontend/                 # (Future web interface)
├── .venv/                    # Python virtual environment
└── README.md                 # This file
```

## Requirements

- Python 3.8 or higher
- Virtual environment (recommended)

### Dependencies

- pandas 2.3.3
- scikit-learn 1.7.2
- numpy 2.3.4

## Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/realbbravo/match_predictor
   cd match_predictor
   ```

2. **Create a virtual environment:**
   ```bash
   python3 -m venv .venv
   ```

3. **Activate the virtual environment:**
   
   On macOS/Linux:
   ```bash
   source .venv/bin/activate
   ```
   
   On Windows:
   ```bash
   .venv\Scripts\activate
   ```

4. **Install required packages:**
   ```bash
   pip install -r requirements.txt
   ```
   
   Or install manually:
   ```bash
   pip install pandas scikit-learn numpy
   ```

## Usage

### Running the Match Predictor

1. **Navigate to the project directory:**
   ```bash
   cd match_predictor
   ```

2. **Activate the virtual environment:**
   ```bash
   source .venv/bin/activate
   ```

3. **Run the predictor:**
   ```bash
   python backend/predictor.py
   ```

### Expected Output

The script will:
1. Load and process the EPL dataset within the /data folder
2. Perform feature engineering with temporal considerations
3. Split data chronologically (80% training, 20% testing)
4. Train a Random Forest model
5. Display evaluation metrics including:
   - Average Accuracy
   - Balanced Accuracy
   - Detailed Classification Report


### Model Training
- Uses Random Forest with 100 estimators
- Temporal train-test split (chronological order preserved)
- Prevents data leakage by using only historical data for predictions

### Evaluation Metrics
- **Accuracy**: Overall prediction correctness
- **Balanced Accuracy**: Accounts for class imbalance
- **Precision/Recall/F1**: Per-class performance metrics

## Future Enhancements

- [ ] Web-based frontend interface
- [ ] Real-time match prediction API
- [ ] Additional feature engineering (head-to-head records, player statistics)
- [ ] Model hyperparameter optimization
- [ ] Integration with live match data feeds

**Note**: This is an educational project demonstrating machine learning applications in sports analytics. Prediction accuracy may vary and should not be used for gambling purposes.