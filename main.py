import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn import metrics

# Custom CSS for styling
st.markdown("""
    <style>
        /* Main page styling */
        .main {
            background-color: #f0f2f6;
            padding: 2rem;
        }
        
        /* Sidebar styling */
        .sidebar .sidebar-content {
            background-color: #1a2330;
            color: white;
        }
        
        /* Widget styling */
        .stNumberInput, .stTextInput {
            background-color: white;
            border-radius: 8px;
            padding: 0.5rem;
        }
        
        /* Button styling */
        .stButton>button {
            background-color: #2e86c1;
            color: white;
            border-radius: 8px;
            padding: 0.5rem 1rem;
            transition: all 0.3s ease;
        }
        
        .stButton>button:hover {
            background-color: #1b4f72;
            color: white;
            transform: scale(1.05);
        }
        
        /* Card styling */
        .card {
            background-color: white;
            border-radius: 12px;
            padding: 1.5rem;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
            margin: 1rem 0;
            transition: transform 0.3s ease;
        }
        
        .card:hover {
            transform: translateY(-5px);
        }
        
        /* Dataframe styling */
        .dataframe {
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
        }
        
        .dataframe th {
            background-color: #2e86c1 !important;
            color: white !important;
        }
        
        .dataframe td {
            transition: background-color 0.3s ease;
        }
        
        .dataframe tbody tr:hover td {
            background-color: #f9e79f !important;
            cursor: pointer;
        }
        
        /* Chart styling */
        .stPlot {
            border-radius: 12px;
            overflow: hidden;
        }
    </style>
""", unsafe_allow_html=True)

# Load the data
@st.cache_data
def load_data():
    gold_data = pd.read_csv("gld_price_data.csv")
    return gold_data

gold_data = load_data()

# Preprocess the data
gold_data = gold_data.copy()
gold_data['Date'] = pd.to_datetime(gold_data['Date'])

# Prepare features and target
X = gold_data.drop(['Date', 'GLD'], axis=1)
Y = gold_data['GLD']

# Train the model
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=2)
regressor = RandomForestRegressor(n_estimators=300)
regressor.fit(X_train, Y_train)

# Prediction function
def predict_gold_price(spx, uso, slv, eur_usd):
    input_data = [[spx, uso, slv, eur_usd]]
    prediction = regressor.predict(input_data)
    return prediction[0]

# App Navigation
st.title("💰 Gold Price Prediction App")

# Sidebar Navigation
page = st.sidebar.radio("Select a Page", 
    ("🏠 Home", "🔮 Prediction", "📊 Dataset", "🌡️ Correlation Heatmap", "📈 Distribution", "📉 Model Performance"),
    index=0
)

# Home Page
if page == "🏠 Home":
    st.markdown("""
        <div class="card">
            <h2 style="color: #2e86c1;">Welcome to Gold Price Prediction App! 🌟</h2>
            <p>This app helps predict gold prices using machine learning based on various economic indicators.</p>
        </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
        <div class="card">
            <h3 style="color: #1b4f72;">📌 Navigation Guide</h3>
            <ul>
                <li><b>🔮 Prediction:</b> Predict gold prices using current market values</li>
                <li><b>📊 Dataset:</b> View the historical gold price data</li>
                <li><b>🌡️ Correlation Heatmap:</b> Explore feature relationships</li>
                <li><b>📈 Distribution:</b> Analyze gold price distribution</li>
                <li><b>📉 Model Performance:</b> View model evaluation metrics</li>
            </ul>
        </div>
    """, unsafe_allow_html=True)

# Prediction Page
elif page == "🔮 Prediction":
    st.markdown("""
        <div class="card">
            <h2 style="color: #2e86c1;">Gold Price Prediction 🔮</h2>
            <p>Enter market values to predict current gold price:</p>
        </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    with col1:
        spx = st.number_input("S&P 500 Index (SPX)", value=gold_data['SPX'].mean())
        uso = st.number_input("US Oil Fund (USO)", value=gold_data['USO'].mean())
    with col2:
        slv = st.number_input("Silver Price (SLV)", value=gold_data['SLV'].mean())
        eur_usd = st.number_input("EUR/USD Exchange Rate", value=gold_data['EUR/USD'].mean())
    
    if st.button("Predict Now 🚀"):
        prediction = predict_gold_price(spx, uso, slv, eur_usd)
        st.markdown(f"""
            <div class="card" style="background-color: #f9e79f; color: #1b4f72;">
                <h3>Predicted Gold Price: ${prediction:.2f}</h3>
                <p>Based on current market values</p>
            </div>
        """, unsafe_allow_html=True)

# Dataset Page
elif page == "📊 Dataset":
    st.markdown("""
        <div class="card">
            <h2 style="color: #2e86c1;">Historical Gold Price Data 📅</h2>
            <p>Explore the dataset used for training the model</p>
        </div>
    """, unsafe_allow_html=True)
    st.dataframe(gold_data.head(15).style.set_properties(**{
        'background-color': '#f0f2f6',
        'color': '#1b4f72',
        'border': '1px solid #dfe6e9'
    }))

# Correlation Heatmap Page
elif page == "🌡️ Correlation Heatmap":
    st.markdown("""
        <div class="card">
            <h2 style="color: #2e86c1;">Feature Correlation Matrix 🌡️</h2>
            <p>Explore relationships between different market indicators</p>
        </div>
    """, unsafe_allow_html=True)
    correlation = gold_data.corr(numeric_only=True)
    plt.figure(figsize=(10, 8))
    sns.set_theme(style="white")
    sns.heatmap(correlation, cbar=True, square=True, fmt='.1f', annot=True, 
                annot_kws={'size':10}, cmap='YlGnBu', linewidths=0.5)
    st.pyplot(plt)

# Distribution Page
elif page == "📈 Distribution":
    st.markdown("""
        <div class="card">
            <h2 style="color: #2e86c1;">Gold Price Distribution 📊</h2>
            <p>Analyze historical price distribution patterns</p>
        </div>
    """, unsafe_allow_html=True)
    plt.figure(figsize=(10, 6))
    sns.set_theme(style="darkgrid")
    sns.histplot(gold_data['GLD'], color='#2e86c1', kde=True, bins=50)
    plt.xlabel('Gold Price (GLD)')
    plt.ylabel('Frequency')
    st.pyplot(plt)

# Model Performance Page
elif page == "📉 Model Performance":
    st.markdown("""
        <div class="card">
            <h2 style="color: #2e86c1;">Model Evaluation 📈</h2>
            <p>Random Forest Regressor performance metrics</p>
        </div>
    """, unsafe_allow_html=True)
    
    test_data_prediction = regressor.predict(X_test)
    error_score = metrics.r2_score(Y_test, test_data_prediction)
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"""
            <div class="card" style="background-color: #e8f6f3;">
                <h3 style="color: #1b4f72;">R-squared Score</h3>
                <h1 style="color: #2e86c1;">{error_score:.4f}</h1>
            </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
            <div class="card" style="background-color: #fdedec;">
                <h3 style="color: #1b4f72;">Model Details</h3>
                <p>Algorithm: Random Forest Regressor</p>
                <p>Estimators: 300 trees</p>
                <p>Test Size: 20%</p>
            </div>
        """, unsafe_allow_html=True)
