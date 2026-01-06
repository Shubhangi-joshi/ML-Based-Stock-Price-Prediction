import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from data_preprocessing import DataPreprocessor
from models import StockPricePredictor
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="Stock Price Prediction",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    </style>
""", unsafe_allow_html=True)

# Initialize session state
if 'data_loaded' not in st.session_state:
    st.session_state.data_loaded = False
if 'models_trained' not in st.session_state:
    st.session_state.models_trained = False
if 'preprocessor' not in st.session_state:
    st.session_state.preprocessor = DataPreprocessor()
if 'predictor' not in st.session_state:
    st.session_state.predictor = StockPricePredictor()

def load_and_preprocess_data():
    """Load and preprocess the data"""
    try:
        # Load data
        df = st.session_state.preprocessor.load_data('nifty_500.csv')
        
        # Clean data
        df_clean = st.session_state.preprocessor.clean_data(df)
        
        # Prepare features
        X, y = st.session_state.preprocessor.prepare_features(df_clean)
        
        st.session_state.df = df
        st.session_state.df_clean = df_clean
        st.session_state.X = X
        st.session_state.y = y
        st.session_state.data_loaded = True
        
        return True
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return False

def train_models():
    """Train the machine learning models"""
    try:
        with st.spinner("Training models... This may take a few moments."):
            results, X_test, y_test = st.session_state.predictor.train_models(
                st.session_state.X, st.session_state.y
            )
            
            st.session_state.results = results
            st.session_state.X_test = X_test
            st.session_state.y_test = y_test
            st.session_state.models_trained = True
            
        return True
    except Exception as e:
        st.error(f"Error training models: {str(e)}")
        return False

# Main App
st.markdown('<h1 class="main-header">📈 Stock Price Prediction System</h1>', unsafe_allow_html=True)
st.markdown("---")

# Sidebar
st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Select Page",
    ["Home", "Data Overview", "Model Training", "Predictions", "Company Analysis"]
)

# Home Page
if page == "Home":
    st.header("Welcome to Stock Price Prediction System")
    st.markdown("""
    This application uses machine learning to predict stock prices for NIFTY 500 companies.
    
    **Features:**
    - 📊 Data overview and visualization
    - 🤖 Multiple ML models (Linear Regression, Random Forest, XGBoost)
    - 📈 Price predictions for individual companies
    - 📉 Performance metrics and analysis
    
    **How to use:**
    1. Go to "Data Overview" to explore the dataset
    2. Navigate to "Model Training" to train the models
    3. Use "Predictions" to predict stock prices
    4. Check "Company Analysis" for detailed company insights
    """)
    
    if st.button("Load Data", type="primary"):
        if load_and_preprocess_data():
            st.success("Data loaded successfully!")
            st.balloons()

# Data Overview Page
elif page == "Data Overview":
    st.header("Data Overview")
    
    if not st.session_state.data_loaded:
        if st.button("Load Data"):
            if load_and_preprocess_data():
                st.success("Data loaded successfully!")
        else:
            st.info("Please load the data first.")
    else:
        df = st.session_state.df_clean
        
        # Dataset info
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Companies", len(df))
        col2.metric("Total Industries", df['Industry'].nunique() if 'Industry' in df.columns else "N/A")
        col3.metric("Avg Price", f"₹{df['Last Traded Price'].mean():.2f}")
        col4.metric("Total Volume", f"{df['Share Volume'].sum():,.0f}")
        
        # Display data
        st.subheader("Dataset Preview")
        st.dataframe(df.head(20), use_container_width=True)
        
        # Statistics
        st.subheader("Statistical Summary")
        st.dataframe(df.describe(), use_container_width=True)
        
        # Visualizations
        st.subheader("Data Visualizations")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Price distribution
            fig = px.histogram(df, x='Last Traded Price', nbins=50, 
                             title="Price Distribution")
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Volume distribution
            fig = px.histogram(df, x='Share Volume', nbins=50,
                             title="Volume Distribution")
            st.plotly_chart(fig, use_container_width=True)
        
        # Industry analysis
        if 'Industry' in df.columns:
            st.subheader("Industry Analysis")
            industry_stats = df.groupby('Industry').agg({
                'Last Traded Price': 'mean',
                'Share Volume': 'mean',
                'Company Name': 'count'
            }).round(2)
            industry_stats.columns = ['Avg Price', 'Avg Volume', 'Company Count']
            industry_stats = industry_stats.sort_values('Company Count', ascending=False)
            st.dataframe(industry_stats, use_container_width=True)
            
            # Top industries by company count
            fig = px.bar(industry_stats.head(15), x=industry_stats.head(15).index, 
                        y='Company Count', title="Top 15 Industries by Company Count")
            st.plotly_chart(fig, use_container_width=True)

# Model Training Page
elif page == "Model Training":
    st.header("Model Training & Evaluation")
    
    if not st.session_state.data_loaded:
        st.warning("Please load the data first from the Home page.")
        if st.button("Load Data"):
            if load_and_preprocess_data():
                st.success("Data loaded successfully!")
    else:
        if st.button("Train Models", type="primary"):
            if train_models():
                st.success("Models trained successfully!")
                st.balloons()
        
        if st.session_state.models_trained:
            results = st.session_state.results
            
            st.subheader("Model Performance Comparison")
            
            # Create metrics DataFrame
            metrics_df = pd.DataFrame({
                'Model': list(results.keys()),
                'Train RMSE': [r['train_rmse'] for r in results.values()],
                'Test RMSE': [r['test_rmse'] for r in results.values()],
                'Train MAE': [r['train_mae'] for r in results.values()],
                'Test MAE': [r['test_mae'] for r in results.values()],
                'Train R²': [r['train_r2'] for r in results.values()],
                'Test R²': [r['test_r2'] for r in results.values()]
            })
            
            st.dataframe(metrics_df, use_container_width=True)
            
            # Best model
            best_model, best_score = st.session_state.predictor.get_best_model(results)
            st.success(f"Best Model: **{best_model}** with Test R² Score: **{best_score:.4f}**")
            
            # Visualization
            col1, col2 = st.columns(2)
            
            with col1:
                # R2 Score comparison
                fig = px.bar(metrics_df, x='Model', y='Test R²', 
                           title="Test R² Score by Model")
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # RMSE comparison
                fig = px.bar(metrics_df, x='Model', y='Test RMSE',
                           title="Test RMSE by Model")
                st.plotly_chart(fig, use_container_width=True)
            
            # Prediction vs Actual
            st.subheader("Predictions vs Actual (Test Set)")
            model_selected = st.selectbox("Select Model", list(results.keys()))
            
            y_test = results[model_selected]['y_test']
            y_pred = results[model_selected]['y_pred']
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=y_test, y=y_pred, mode='markers',
                                   name='Predictions', marker=dict(color='blue')))
            fig.add_trace(go.Scatter(x=[y_test.min(), y_test.max()], 
                                   y=[y_test.min(), y_test.max()],
                                   mode='lines', name='Perfect Prediction',
                                   line=dict(color='red', dash='dash')))
            fig.update_layout(
                title=f"Actual vs Predicted Prices - {model_selected}",
                xaxis_title="Actual Price",
                yaxis_title="Predicted Price"
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Feature importance
            if model_selected in st.session_state.predictor.feature_importance:
                st.subheader("Feature Importance")
                importance = st.session_state.predictor.feature_importance[model_selected]
                feature_names = st.session_state.predictor.feature_names
                
                importance_df = pd.DataFrame({
                    'Feature': feature_names,
                    'Importance': importance
                }).sort_values('Importance', ascending=False).head(10)
                
                fig = px.bar(importance_df, x='Importance', y='Feature',
                           orientation='h', title="Top 10 Most Important Features")
                st.plotly_chart(fig, use_container_width=True)

# Predictions Page
elif page == "Predictions":
    st.header("Stock Price Predictions")
    
    if not st.session_state.data_loaded:
        st.warning("Please load the data first.")
    elif not st.session_state.models_trained:
        st.warning("Please train the models first.")
    else:
        df_clean = st.session_state.df_clean
        
        # Company selection
        st.subheader("Select Company")
        company_list = df_clean['Company Name'].tolist()
        selected_company = st.selectbox("Choose a company", company_list)
        
        # Get company data
        company_data = df_clean[df_clean['Company Name'] == selected_company].iloc[0]
        
        # Display company info
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Symbol", company_data.get('Symbol', 'N/A'))
        col2.metric("Industry", company_data.get('Industry', 'N/A'))
        col3.metric("Current Price", f"₹{company_data['Last Traded Price']:.2f}")
        col4.metric("Previous Close", f"₹{company_data.get('Previous Close', 0):.2f}")
        
        # Prepare features for prediction
        X_company, _ = st.session_state.preprocessor.prepare_features(
            df_clean[df_clean['Company Name'] == selected_company]
        )
        
        # Model selection
        st.subheader("Select Model for Prediction")
        model_selected = st.selectbox("Choose a model", 
                                     list(st.session_state.predictor.trained_models.keys()))
        
        # Make prediction
        if st.button("Predict Price", type="primary"):
            prediction = st.session_state.predictor.predict(model_selected, X_company)
            predicted_price = prediction[0]
            current_price = company_data['Last Traded Price']
            change = predicted_price - current_price
            change_pct = (change / current_price) * 100
            
            # Display prediction
            col1, col2, col3 = st.columns(3)
            col1.metric("Current Price", f"₹{current_price:.2f}")
            col2.metric("Predicted Price", f"₹{predicted_price:.2f}", 
                       f"₹{change:.2f} ({change_pct:+.2f}%)")
            col3.metric("52 Week High", f"₹{company_data.get('52 Week High', 0):.2f}")
            
            # Prediction chart
            fig = go.Figure()
            fig.add_trace(go.Bar(x=['Current Price', 'Predicted Price'],
                               y=[current_price, predicted_price],
                               marker_color=['blue', 'green']))
            fig.update_layout(
                title="Price Comparison",
                yaxis_title="Price (₹)",
                showlegend=False
            )
            st.plotly_chart(fig, use_container_width=True)

# Company Analysis Page
elif page == "Company Analysis":
    st.header("Company Analysis")
    
    if not st.session_state.data_loaded:
        st.warning("Please load the data first.")
    else:
        df_clean = st.session_state.df_clean
        
        # Company selection
        company_list = df_clean['Company Name'].tolist()
        selected_company = st.selectbox("Select Company", company_list)
        
        # Get company data
        company_df = df_clean[df_clean['Company Name'] == selected_company]
        company_data = company_df.iloc[0]
        
        # Company details
        st.subheader("Company Details")
        col1, col2 = st.columns(2)
        
        with col1:
            st.write(f"**Symbol:** {company_data.get('Symbol', 'N/A')}")
            st.write(f"**Industry:** {company_data.get('Industry', 'N/A')}")
            st.write(f"**Series:** {company_data.get('Series', 'N/A')}")
            st.write(f"**Current Price:** ₹{company_data['Last Traded Price']:.2f}")
            st.write(f"**Open:** ₹{company_data.get('Open', 0):.2f}")
            st.write(f"**High:** ₹{company_data.get('High', 0):.2f}")
            st.write(f"**Low:** ₹{company_data.get('Low', 0):.2f}")
        
        with col2:
            st.write(f"**Previous Close:** ₹{company_data.get('Previous Close', 0):.2f}")
            st.write(f"**Change:** ₹{company_data.get('Change', 0):.2f}")
            st.write(f"**Change %:** {company_data.get('Percentage Change', 0):.2f}%")
            st.write(f"**Volume:** {company_data.get('Share Volume', 0):,}")
            st.write(f"**52 Week High:** ₹{company_data.get('52 Week High', 0):.2f}")
            st.write(f"**52 Week Low:** ₹{company_data.get('52 Week Low', 0):.2f}")
        
        # Price range chart
        st.subheader("Price Analysis")
        price_data = {
            'Open': company_data.get('Open', 0),
            'High': company_data.get('High', 0),
            'Low': company_data.get('Low', 0),
            'Close': company_data.get('Last Traded Price', 0),
            'Previous Close': company_data.get('Previous Close', 0)
        }
        
        fig = go.Figure(data=go.Scatter(
            x=list(price_data.keys()),
            y=list(price_data.values()),
            mode='lines+markers',
            marker=dict(size=10),
            line=dict(width=2)
        ))
        fig.update_layout(
            title="Price Range",
            yaxis_title="Price (₹)",
            xaxis_title="Price Type"
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Comparison with industry
        if 'Industry' in df_clean.columns:
            st.subheader("Industry Comparison")
            industry = company_data.get('Industry')
            industry_companies = df_clean[df_clean['Industry'] == industry].copy()
            
            col1, col2, col3 = st.columns(3)
            col1.metric("Industry Avg Price", 
                       f"₹{industry_companies['Last Traded Price'].mean():.2f}")
            col2.metric("Industry Median Price",
                       f"₹{industry_companies['Last Traded Price'].median():.2f}")
            col3.metric("Companies in Industry",
                       len(industry_companies))
            
            # Company rank in industry
            industry_sorted = industry_companies.sort_values('Last Traded Price', ascending=False)
            company_price = company_data['Last Traded Price']
            rank = (industry_sorted['Last Traded Price'] >= company_price).sum()
            st.info(f"**Rank in Industry:** {rank} out of {len(industry_companies)} companies")

# Footer
st.markdown("---")
st.markdown("<p style='text-align: center; color: gray;'>Stock Price Prediction System | Powered by Machine Learning</p>", 
           unsafe_allow_html=True)

