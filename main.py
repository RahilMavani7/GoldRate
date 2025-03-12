import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn import metrics

# Custom CSS for dark theme and golden effects
st.markdown("""
    <style>
    /* Main dark theme */
    body {
        background-color: #0E1117;
        color: #ffffff;
    }
    
    /* Sidebar styling */
    .sidebar .sidebar-content {
        background-color: #1a1a1a;
        color: white;
    }
    
    /* Widget styling */
    .stNumberInput, .stTextInput, .stSelectbox {
        background-color: #2d2d2d;
        color: white;
        border-radius: 8px;
    }
    
    /* Button styling */
    .stButton>button {
        background-color: #2d2d2d;
        color: #FFD700;
        border: 1px solid #FFD700;
        border-radius: 8px;
        transition: all 0.3s ease;
    }
    
    .stButton>button:hover {
        background-color: #FFD700;
        color: #1a1a1a !important;
        transform: scale(1.05);
    }
    
    /* Card styling */
    .card {
        background-color: #1a1a1a;
        border: 1px solid #2d2d2d;
        border-radius: 12px;
        padding: 1.5rem;
        margin: 1rem 0;
        transition: all 0.3s ease;
    }
    
    .card:hover {
        border-color: #FFD700;
        box-shadow: 0 0 15px rgba(255, 215, 0, 0.2);
    }
    
    /* Dataframe styling */
    .dataframe {
        background-color: #1a1a1a !important;
        color: white !important;
        border-radius: 8px;
    }
    
    .dataframe th {
        background-color: #FFD700 !important;
        color: #1a1a1a !important;
    }
    
    .dataframe td {
        transition: all 0.3s ease;
    }
    
    .dataframe tbody tr:hover td {
        background-color: #2d2d2d !important;
        color: #FFD700 !important;
    }
    
    /* Chart styling */
    .stPlot {
        background-color: #1a1a1a;
        border-radius: 12px;
        padding: 1rem;
    }
    
    /* Text colors */
    h1, h2, h3, h4, h5, h6 {
        color: #FFD700 !important;
    }
    
    p, li {
        color: #cccccc !important;
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
    ("🏠 Home", "🔮 Prediction", "📊 Dataset", "🌡️ Correlation", "📈 Distribution", "📉 Performance"),
    index=0
)

# Home Page
if page == "🏠 Home":
    st.markdown("""
    <div class="card">
        <h2>Welcome to Gold Price Prediction App! 🌟</h2>
        <p>This app predicts gold prices using machine learning based on various market indicators:</p>
        <ul>
            <li>SPX Index</li>
            <li>US Oil Fund (USO)</li>
            <li>Silver Prices (SLV)</li>
            <li>EUR/USD Exchange Rate</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

# Prediction Page
elif page == "🔮 Prediction":
    st.markdown("""
    <div class="card">
        <h2>Gold Price Prediction</h2>
        <p>Enter current market values to predict gold price:</p>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    with col1:
        spx = st.number_input("SPX Value", value=gold_data['SPX'].mean())
        uso = st.number_input("USO Value", value=gold_data['USO'].mean())
    with col2:
        slv = st.number_input("SLV Value", value=gold_data['SLV'].mean())
        eur_usd = st.number_input("EUR/USD Value", value=gold_data['EUR/USD'].mean())
    
    if st.button("✨ Predict Gold Price"):
        prediction = predict_gold_price(spx, uso, slv, eur_usd)
        st.markdown(f"""
        <div class="card" style="border-color: #FFD700;">
            <h3>Predicted Gold Price (GLD):</h3>
            <h1 style="color: #FFD700;">${prediction:.2f}</h1>
        </div>
        """, unsafe_allow_html=True)

# Dataset Page
elif page == "📊 Dataset":
    st.markdown("""
    <div class="card">
        <h2>Gold Price Dataset</h2>
        <p>Historical market data used for predictions:</p>
    </div>
    """, unsafe_allow_html=True)
    st.dataframe(gold_data.head(15).style.set_properties(**{
        'background-color': '#1a1a1a',
        'color': '#ffffff',
        'border': '1px solid #2d2d2d'
    }))

# Correlation Page
elif page == "🌡️ Correlation":
    st.markdown("""
    <div class="card">
        <h2>Market Indicators Correlation</h2>
        <p>Relationship between different financial instruments:</p>
    </div>
    """, unsafe_allow_html=True)
    
    plt.figure(figsize=(10, 8))
    sns.set(style="dark")
    correlation = gold_data.corr(numeric_only=True)
    sns.heatmap(correlation, cbar=True, square=True, 
                fmt='.1f', annot=True, annot_kws={'size':10},
                cmap='YlOrBr', linewidths=0.5)
    st.pyplot(plt)

# Distribution Page
elif page == "📈 Distribution":
    st.markdown("""
    <div class="card">
        <h2>Gold Price Distribution</h2>
        <p>Historical price frequency distribution:</p>
    </div>
    """, unsafe_allow_html=True)
    
    plt.figure(figsize=(10, 6))
    sns.set(style="darkgrid")
    sns.histplot(gold_data['GLD'], color='#FFD700', 
                kde=True, bins=50, alpha=0.8)
    plt.xlabel('Gold Price (GLD)')
    plt.ylabel('Frequency')
    st.pyplot(plt)

# Performance Page
elif page == "📉 Performance":
    st.markdown("""
    <div class="card">
        <h2>Model Performance</h2>
        <p>Random Forest Regressor evaluation metrics:</p>
    </div>
    """, unsafe_allow_html=True)
    
    test_data_prediction = regressor.predict(X_test)
    error_score = metrics.r2_score(Y_test, test_data_prediction)
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"""
        <div class="card">
            <h3>R-squared Score</h3>
            <h1 style="color: #FFD700;">{error_score:.4f}</h1>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="card">
            <h3>Model Details</h3>
            <p>Algorithm: Random Forest</p>
            <p>Estimators: 300 trees</p>
            <p>Test Size: 20%</p>
        </div>
        """, unsafe_allow_html=True)
