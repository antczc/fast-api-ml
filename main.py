from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import pandas as pd
import yfinance as yf

stock = yf.Ticker("AAPL")
df = stock.history(period="1y")
df.to_csv('AAPL_stock_data.csv')

# Splitting data
X_train, X_test, y_train, y_test = train_test_split(df.drop('Close', axis=1), df['Close'], test_size=0.2, random_state=42)

# Instantiate and training model
model = LinearRegression()
model.fit(X_train, y_train)

import joblib
joblib.dump(model, 'aapl_stock_price_model.pkl')

from fastapi import FastAPI
from pydantic import BaseModel
import joblib

# Define data model
class StockData(BaseModel):
    Open: float
    High: float
    Low: float
    Volume: float
    Close: float
    Dividends: float

app = FastAPI()

# Load trained model
model = joblib.load('aapl_stock_price_model.pkl')

@app.post('/predict')
def predict(data: StockData):
    data = data.dict()
    Open = data['Open']
    High = data['High']
    Low = data['Low']
    Volume = data['Volume']
    Close = data['Close']
    Dividends = data['Dividends']
    
    prediction = model.predict([[Open, High, Low, Volume, Close, Dividends]])
    return {
        'prediction': prediction[0]
    }

