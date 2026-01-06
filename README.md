# Stock Price Prediction System

A comprehensive stock price prediction application using machine learning and Streamlit for the NIFTY 500 dataset.

## Features

- 📊 **Data Overview**: Explore and visualize the NIFTY 500 dataset
- 🤖 **Multiple ML Models**: Linear Regression, Random Forest, and XGBoost
- 📈 **Price Predictions**: Predict stock prices for individual companies
- 📉 **Performance Metrics**: Detailed model evaluation and comparison
- 🏢 **Company Analysis**: In-depth analysis of individual companies
- 📱 **Interactive UI**: User-friendly Streamlit interface

## Installation

1. Clone or download this repository

2. Install required packages:
```bash
pip install -r requirements.txt
```

## Usage

1. Make sure the `nifty_500.csv` file is in the project directory

2. Run the Streamlit app:
```bash
streamlit run app.py
```

3. The application will open in your default web browser

## How to Use

1. **Home**: Start by loading the data
2. **Data Overview**: Explore the dataset, view statistics, and visualizations
3. **Model Training**: Train the machine learning models and view performance metrics
4. **Predictions**: Select a company and predict its stock price
5. **Company Analysis**: Get detailed insights about specific companies

## Project Structure

```
stock-prediction/
│
├── app.py                 # Main Streamlit application
├── data_preprocessing.py  # Data loading and preprocessing
├── models.py             # ML models and training
├── requirements.txt      # Python dependencies
├── README.md            # Project documentation
└── nifty_500.csv        # Dataset file
```

## Models Used

1. **Linear Regression**: Simple linear model for baseline predictions
2. **Random Forest**: Ensemble method with multiple decision trees
3. **XGBoost**: Gradient boosting algorithm for improved accuracy

## Features Used for Prediction

- Open, High, Low, Previous Close prices
- Change and Percentage Change
- Share Volume and Trading Value
- 52 Week High and Low
- Historical percentage changes (365 day, 30 day)
- Derived features (Price Range, Volume-Price Ratio, etc.)
- Industry encoding

## Performance Metrics

The application evaluates models using:
- **RMSE** (Root Mean Squared Error)
- **MAE** (Mean Absolute Error)
- **R² Score** (Coefficient of Determination)

## Notes

- This is a snapshot dataset, so predictions are based on feature relationships
- For better predictions, historical time-series data would be required
- The models predict closing prices based on current market features
- Always use predictions as a tool for analysis, not as financial advice

## Requirements

- Python 3.7+
- Streamlit
- Pandas
- NumPy
- Scikit-learn
- XGBoost
- Matplotlib
- Seaborn
- Plotly

## License

This project is for educational purposes only.

## Disclaimer

This application is for educational and research purposes only. Stock market predictions are inherently uncertain, and past performance does not guarantee future results. Always consult with a qualified financial advisor before making investment decisions.

