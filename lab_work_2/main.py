import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf

# Classical ML
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

# DL
import torch
import torch.nn as nn

# Statistics
from statsmodels.tsa.holtwinters import ExponentialSmoothing


torch.manual_seed(42)
np.random.seed(42)

class SyntheticDataGenerator:
    def __init__(self, length=1000):
        self.length = length

    def generate(self):
        t = np.linspace(0, 50, self.length)

        # Trend (Quadratic + seasonality + noise)
        trend = 0.5 * t**2 + 2 * t + 100
        seasonality = 20 * np.sin(t)

        noise =  np.random.normal(0, 15, self.length)
        data = trend + seasonality + noise

        anomalies_indices = [200, 500, 800]
        for idx in anomalies_indices:
            data[idx] += 150

        print(f"Generated synthetic data with {self.length} points, including anomalies at indices: {anomalies_indices}")
        return data

class CustomAnomalyDetector:
    @staticmethod
    def detect_rolling_z_score(data, window_size=30, threshold=3):
        """
        Custom anomaly detection using rolling Z-score method. It calculates the rolling mean and standard deviation,
        then identifies points that deviate from the mean by more than a specified number of standard deviations.
        In our case, we use a window size of 30 days and a threshold of 3 standard deviations to flag anomalies.
        """
        clean_data = data.copy()
        anomalies_idx = []

        for i in range(window_size, len(data)):
            local_window = data[i - window_size : i] # Get the previous window of data of length window_size

            local_mean = np.mean(local_window)
            local_std = np.std(local_window)
            if local_std == 0: continue

            z_score = (data[i] - local_mean) / local_std # Calculate the Z-score for the current point based on the local mean and std

            if abs(z_score) > threshold: # If the absolute Z-score exceeds the threshold, it's considered an anomaly
                anomalies_idx.append(i)
                clean_data[i] = local_mean # Replace the anomaly with the local mean to smooth the data
        print(f"Detected {len(anomalies_idx)} anomalies using rolling Z-score method.")
        return clean_data, anomalies_idx





def get_stock_data(symbol="BLK", period="5y"):
    print(f"Downloading data for {symbol} for period: {period}")
    try:
        ticker = yf.Ticker(symbol)
        data = ticker.history(period=period, interval="1d")
        close_data = data["Close"].values  # Extract closing prices in 1D array
        print(f"Downloaded {len(close_data)} data points.")
        return close_data
    except Exception as e:
        print(f"Error downloading data: {e}")
        return None


def analyse_and_clean_data(data):
    print("Statistical Summary: ")
    print(f"Mean: {np.mean(data):.2f}")
    print(f"Min: {np.min(data):.2f} Max: {np.max(data):.2f}")

    Q1 = np.percentile(data, 25)
    Q3 = np.percentile(data, 75)
    IQR = Q3 - Q1

    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    print(f"Lower bound: {lower_bound:.2f} Upper bound: {upper_bound:.2f}")

    clean_data = data.copy()
    anomalies = []

    for i in range(len(data)):
        if data[i] < lower_bound or data[i] > upper_bound:
            anomalies.append(i)
            if i > 0:
                clean_data[i] = clean_data[i - 1]  # Replace anomaly with previous value
            else:
                clean_data[i] = clean_data[
                    i + 1
                ]  # If first element is anomaly, replace with next value
    print(f"Number of anomalies with IQR Method: {len(anomalies)}")
    return clean_data


class LSTMModel(nn.Module):
    def __init__(self, input_size=1, hidden_size=64, num_layers=2, output_size=1, dropout=0.2):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(
            input_size, hidden_size, num_layers, batch_first=True, dropout=dropout
        )  # (Batch, Seq, Features)
        self.linear = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # x: (batch_size, seq_length, input_size)
        # lstm out shape: (batch_size, seq_len, hidden_size)
        lstm_out, _ = self.lstm(x)

        # We want to predict the next value, so we take the last output
        last_out = lstm_out[:, -1, :]  # (batch_size, hidden_size)
        out = self.linear(last_out)  # (batch_size, output_size)
        return out


class Predictor:
    def __init__(self, train_data, test_data):
        self.train_data = train_data
        self.test_data = test_data

    def polynomial_approximation(self, degree=2):
        X_train = np.arange(len(self.train_data)).reshape(
            -1, 1
        )  # Reshape for sklearn, n rows, 1 column
        y = self.train_data

        poly = PolynomialFeatures(degree=degree)
        X_poly = poly.fit_transform(
            X_train
        )  # Transform to polynomial features, 1, x1, x1^2. n rows, 3 columns

        model = LinearRegression()
        model.fit(X_poly, y)

        start_idx = len(self.train_data)
        end_idx = start_idx + len(self.test_data)

        X_test = np.arange(start_idx, end_idx).reshape(-1, 1)
        X_test_poly = poly.transform(X_test)
        y_pred = model.predict(X_test_poly)
        return y_pred

    def holt_winters(self):
        # Holt-Winters Exponential Smoothing, it extends simple exponential smoothing to capture trends and
        # seasonality in data.
        model = ExponentialSmoothing(
            self.train_data, trend="add", seasonal=None, initialization_method="estimated"
        ).fit()
        forecast = model.forecast(len(self.test_data))
        return forecast

    def lstm_torch(self, look_back=60, epochs=100, lr=0.005):
        scaler = MinMaxScaler(feature_range=(0, 1))  # Normalizing data
        train_scaled = scaler.fit_transform(self.train_data.reshape(-1, 1))

        X_train_seq, y_train_seq = [], []
        for i in range(len(train_scaled) - look_back):
            X_train_seq.append(train_scaled[i : i + look_back])
            y_train_seq.append(train_scaled[i + look_back])

        X_train_seq = torch.FloatTensor(np.array(X_train_seq))
        y_train_seq = torch.FloatTensor(np.array(y_train_seq))

        model = LSTMModel(input_size=1, hidden_size=64, num_layers=2, output_size=1)
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)

        model.train()
        for epoch in range(epochs):
            optimizer.zero_grad()
            output = model(X_train_seq)
            loss = criterion(output, y_train_seq)
            loss.backward()
            optimizer.step()

            if (epoch + 1) % 5 == 0:
                print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}")

        # Recursive prediction
        model.eval()
        predictions = []

        current_batch = torch.FloatTensor(train_scaled[-look_back:]).reshape(
            1, look_back, 1
        )

        with torch.no_grad():
            for i in range(len(self.test_data)):
                pred_step = model(current_batch)
                predictions.append(pred_step.item())

                # Update the current batch to include the new prediction (1, 60, 1) -> (1, 60, 1)
                new_step = pred_step.reshape(1, 1, 1)
                current_batch = torch.cat((current_batch[:, 1:, :], new_step), dim=1)

        predictions = np.array(predictions).reshape(-1, 1)
        predictions = scaler.inverse_transform(predictions).flatten()
        return predictions


if __name__ == "__main__":

    DATA_MODE = 'SYNTHETIC' # 'SYNTHETIC' or 'REAL'

    # BlackRock, Inc. stock symbol: BLK, 5 years data
    SYMBOL = "BLK"
    TEST_DAYS = 100

    if DATA_MODE == 'REAL':
        close_data = get_stock_data(symbol=SYMBOL, period="5y")
    else:
        gen = SyntheticDataGenerator(length=1200)
        close_data = gen.generate()

    clean_data, anomalies = CustomAnomalyDetector.detect_rolling_z_score(close_data, window_size=30, threshold=3)
    # clean_data = analyse_and_clean_data(close_data)

    train_data = clean_data[:-TEST_DAYS]
    test_data = clean_data[-TEST_DAYS:]

    print(f"Train data length: {len(train_data)}")
    print(f"Test data length: {len(test_data)}")

    predictor = Predictor(train_data, test_data)

    # Polynomial Approximation

    if DATA_MODE == 'REAL':
        degree = 1 # Linear trend for real data
    else:
        degree = 2 # Quadratic for synthetic data
    pol_pred = predictor.polynomial_approximation(degree=degree)
    mse_pol = mean_squared_error(test_data, pol_pred)
    mae_pol = np.mean(np.abs(test_data - pol_pred))
    rmse_pol = np.sqrt(mse_pol)
    print(f"Polynomial Approximation MSE: {mse_pol:.2f}")
    print(f"Polynomial Approximation MAE: {mae_pol:.2f}")
    print(f"Polynomial Approximation RMSE: {rmse_pol:.2f}")

    # Holt-Winters Exponential Smoothing

    holt_winters_pred = predictor.holt_winters()[-TEST_DAYS:]
    mse_hw = mean_squared_error(test_data, holt_winters_pred)
    mae_hw = np.mean(np.abs(test_data - holt_winters_pred))
    rmse_hw = np.sqrt(mse_hw)
    print(f"Holt-Winters MSE: {mse_hw:.2f}")
    print(f"Holt-Winters MAE: {mae_hw:.2f}")
    print(f"Holt-Winters RMSE: {rmse_hw:.2f}")

    # LSTM with PyTorch
    # 90 days look back (approximately 4 months of trading days)
    lstm_pred = predictor.lstm_torch(look_back=90, epochs=50, lr=0.005)
    mse_lstm = mean_squared_error(test_data, lstm_pred)
    mae_lstm = np.mean(np.abs(test_data - lstm_pred))
    rmse_lstm = np.sqrt(mse_lstm)

    print(f"LSTM MSE: {mse_lstm:.2f}")
    print(f"LSTM MAE: {mae_lstm:.2f}")
    print(f"LSTM RMSE: {rmse_lstm:.2f}")

    scores = {
        'Polynomial': rmse_pol,
        'Holt-Winters': rmse_hw,
        'LSTM': rmse_lstm
    }
    best_model = min(scores, key=scores.get)
    print(
        f"Best model based on MSE: {best_model} with MSE: {scores[best_model]:.2f}"
    )

    plt.figure()
    start_point = len(train_data) - 400

    valid_anomalies = [x for x in anomalies if x >= start_point and x < len(train_data)]
    if valid_anomalies:
        plt.scatter(valid_anomalies, [close_data[i] for i in valid_anomalies], color='red', label='Detected Anomalies', zorder=5)
    plt.plot(
        np.arange(start_point, len(train_data)),
        train_data[start_point:],
        label="History Data",
        color="gray",
    )
    plt.plot(
        np.arange(len(train_data), len(train_data) + len(test_data)),
        test_data,
        label="Real Future",
        color="blue",
    )

    x_future = np.arange(len(train_data), len(train_data) + len(test_data))
    plt.plot(x_future, pol_pred, label="Polynomial Prediction", color="red")
    plt.plot(x_future, holt_winters_pred, label="Exp. Smoothing (Holt)", color="green")
    plt.plot(x_future, lstm_pred, label="LSTM Prediction", color="orange")

    plt.title(f"BlackRock Stock Prediction")
    plt.xlabel("Trading Days")
    plt.ylabel("Price USD")
    plt.legend()
    plt.grid()
    plt.show()
