
## 🧠 Model Options
- Linear Regression  
- Random Forest Regressor  

## 📈 Output
The script will display:
- Model performance (R² score)
- Plot of actual vs predicted closing prices
- Next day's predicted closing price
import yfinance as yf
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt

# --- Choose stock ---
STOCK = "AAPL"   # You can change to TSLA, MSFT, etc.

# --- Load historical data ---
data = yf.download(STOCK, period="2y")   # last 2 years

# Use selected features
features = data[["Open", "High", "Low", "Volume"]]
target = data["Close"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    features, target, test_size=0.2, shuffle=False
)

# --- Choose your model ---
use_rf = False     # Set to True for Random Forest

if use_rf:
    model = RandomForestRegressor(n_estimators=200)
else:
    model = LinearRegression()

# Train
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Score
score = r2_score(y_test, y_pred)
print(f"Model R² Score: {score:.4f}")

# Predict next day
next_day_features = features.tail(1)
next_close_prediction = model.predict(next_day_features)[0]
print(f"Predicted Next Closing Price for {STOCK}: {next_close_prediction:.2f}")

# --- Plot ---
plt.figure(figsize=(12,5))
plt.plot(y_test.values, label="Actual Close", linewidth=2)
plt.plot(y_pred, label="Predicted Close", linewidth=2)
plt.title(f"Actual vs Predicted Closing Prices ({STOCK})")
plt.xlabel("Days")
plt.ylabel("Price")
plt.legend()
plt.grid()
plt.show()
