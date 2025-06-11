# Finance Prediction – Time Series

This repository provides a modular **time-series forecasting system** for financial assets such as stocks, REITs, and bonds.  
The system includes data preprocessing, feature engineering with technical indicators, and several forecasting models for multi-horizon price prediction.

---

## Project Purpose

The project aims to explore and evaluate various forecasting models on financial time-series data using both raw price data and engineered technical indicators.

Key features:

- Data preprocessing and feature engineering for technical indicators  
- Forecasting asset prices using multiple models including Linear Regression, Support Vector Regression (SVR), K-Nearest Neighbors (KNN), and Long Short-Term Memory networks (LSTM)  
- Multi-horizon forecasting for 30, 60, 90, 120, and 150 days ahead  
- Model evaluation with train-test splits and one-day-ahead prediction  
- Portfolio optimization based on predicted prices (using genetic algorithm)  
- Visualization of results and performance metrics  

---

## Technologies Used

| Technology          | Description |
|---------------------|-------------|
| `Python 3.10+`       | Core programming language |
| `pandas`, `numpy`    | Data processing and numerical operations |
| `scikit-learn`       | Machine learning models and utilities (Linear Regression, SVR, KNN) |
| `tensorflow`, `keras`| Deep learning framework for LSTM models |
| `matplotlib`, `seaborn` | Data visualization and plotting |
| `deap`               | Genetic algorithm for portfolio optimization |
| `TA-Lib` / custom code | Technical indicators calculation |

---
## Folder Structure

Finance-Prediction--Time-Series--/
│
├── data/ # Input raw and processed data files
├── models/ # Model training and saved models
├── indicators/ # Scripts for technical indicator calculation
├── optimization/ # Portfolio optimization code using genetic algorithms
├── notebooks/ # Jupyter notebooks for experimentation
├── utils/ # Helper functions and utilities
├── main.py # Main script to run forecasting pipeline
└── requirements.txt # Project dependencies
