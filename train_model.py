import ccxt
import pandas as pd
import ta
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# 1. Daten abrufen
symbol = "BTC/USDT"
exchange = ccxt.bitget()
ohlcv = exchange.fetch_ohlcv(symbol, timeframe='30m', limit=1000)
df = pd.DataFrame(ohlcv, columns=['time', 'open', 'high', 'low', 'close', 'volume'])
df['time'] = pd.to_datetime(df['time'], unit='ms')

# 2. Indikatoren berechnen
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

# 3. Klassisches Score-System
def long_score(row):
    score = 0
    score += row['ema5'] > row['ema10']
    score += 45 < row['rsi'] < 60
    score += row['stoch_k'] > row['stoch_d']
    score += row['close'] > row['close'] - 2 * row['atr']  # wie BB-Lower
    score += row['macd_line'] > row['macd_signal']
    score += row['volume'] > row['volume_avg']
    score += row['adx'] > 20
    return score

def short_score(row):
    score = 0
    score += row['ema5'] < row['ema10']
    score += 40 < row['rsi'] < 55
    score += row['stoch_k'] < row['stoch_d']
    score += row['close'] < row['close'] + 2 * row['atr']  # wie BB-Upper
    score += row['macd_line'] < row['macd_signal']
    score += row['volume'] > row['volume_avg']
    score += row['adx'] > 20
    return score

df['long_score'] = df.apply(long_score, axis=1)
df['short_score'] = df.apply(short_score, axis=1)

# 4. Zielvariable: realistischer Trade-Ausgang
def simulate_trade(row, highs, lows, tp_factor=2, sl_factor=1.5):
    entry = row['close']
    atr = row['atr']
    tp = entry + tp_factor * atr
    sl = entry - sl_factor * atr
    for h, l in zip(highs, lows):
        if l <= sl:
            return -1
        if h >= tp:
            return 1
    return 0

lookahead = 5
results = []

for i in range(len(df) - lookahead):
    row = df.iloc[i]
    highs = df['high'].iloc[i+1:i+1+lookahead].values
    lows = df['low'].iloc[i+1:i+1+lookahead].values
    results.append(simulate_trade(row, highs, lows))

df = df.iloc[:len(results)].copy()
df['target'] = results
df.dropna(inplace=True)

# Nur LONG & SHORT
df = df[df['target'] != 0]

# 5. Feature-Set (inkl. Scores)
features = [
    'ema5', 'ema10', 'rsi', 'stoch_k', 'stoch_d',
    'macd_line', 'macd_signal', 'adx', 'volume',
    'volume_avg', 'atr', 'long_score', 'short_score'
]

X = df[features]
y = df['target']

# 6. Trainieren
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = RandomForestClassifier(n_estimators=150, max_depth=10, random_state=42)
model.fit(X_train, y_train)

# 7. Evaluierung & Speichern
print("✅ Genauigkeit:", model.score(X_test, y_test))
joblib.dump(model, "trade_model.pkl")
print("✅ Modell mit klassischem Wissen + ML gespeichert.")
