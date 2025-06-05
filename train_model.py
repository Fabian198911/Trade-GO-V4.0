# train_model.py
import ccxt
import pandas as pd
import ta
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# 1. Datenquelle definieren (BTC/USDT)
symbol = "BTC/USDT"
exchange = ccxt.bitget()

# 2. OHLCV-Daten abrufen
ohlcv = exchange.fetch_ohlcv(symbol, timeframe='30m', limit=500)
df = pd.DataFrame(ohlcv, columns=['time', 'open', 'high', 'low', 'close', 'volume'])
df['time'] = pd.to_datetime(df['time'], unit='ms')

# 3. Indikatoren berechnen
df['ema5'] = ta.trend.EMAIndicator(df['close'], 5).ema_indicator()
df['ema10'] = ta.trend.EMAIndicator(df['close'], 10).ema_indicator()
df['rsi'] = ta.momentum.RSIIndicator(df['close']).rsi()
stoch = ta.momentum.StochRSIIndicator(df['close'])
df['stoch_k'] = stoch.stochrsi_k()
df['stoch_d'] = stoch.stochrsi_d()
macd = ta.trend.MACD(df['close'])
df['macd_line'] = macd.macd()
df['macd_signal'] = macd.macd_signal()
df['volume_avg'] = df['volume'].rolling(20).mean()
df['adx'] = ta.trend.ADXIndicator(df['high'], df['low'], df['close']).adx()
df['atr'] = ta.volatility.AverageTrueRange(df['high'], df['low'], df['close']).average_true_range()

# 4. Zielvariable: Kurs steigt im nächsten Intervall = LONG (1), sonst SHORT (0)
df['target'] = (df['close'].shift(-1) > df['close']).astype(int)

# 5. Zeilen mit NaN entfernen
df.dropna(inplace=True)

# 6. Feature-Matrix & Zielvariable
features = [
    'ema5', 'ema10', 'rsi', 'stoch_k', 'stoch_d',
    'macd_line', 'macd_signal', 'adx', 'volume',
    'volume_avg', 'atr'
]

X = df[features]
y = df['target']

# 7. Train-Test-Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 8. Modell trainieren
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 9. Modell evaluieren
print("Testgenauigkeit:", model.score(X_test, y_test))

# 10. Modell speichern
joblib.dump(model, "trade_model.pkl")
print("✅ Modell gespeichert als trade_model.pkl")
