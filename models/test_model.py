import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt

# 1. 数据加载
def load_data(ticker, start_date, end_date):
    """
    从Yahoo Finance加载股票数据。
    """
    print(f"Loading data for {ticker} from {start_date} to {end_date}...")
    stock_df = yf.download(ticker, start=start_date, end=end_date)
    if stock_df.empty:
        raise ValueError("No data found for the given ticker and date range.")
    print("Data loaded successfully!")
    return stock_df['Close']

# 2. 数据预处理
def preprocess_data(data, scaler=None):
    """
    对数据进行归一化处理。
    """
    print("Preprocessing data...")
    if scaler is None:
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_data = scaler.fit_transform(data.values.reshape(-1, 1))
    else:
        scaled_data = scaler.transform(data.values.reshape(-1, 1))
    print("Data preprocessing complete!")
    return scaled_data, scaler

# 3. 准备指定日期的输入
def prepare_input(data, target_date, lookback=120, scaler=None):
    """
    根据指定日期生成模型输入数据。
    """
    target_date = pd.to_datetime(target_date)

    # 检查目标日期是否在数据中
    if target_date not in data.index:
        # 调整到最近的交易日
        adjusted_date = data.index[data.index.searchsorted(target_date, side="right") - 1]
        print(f"Target date {target_date} adjusted to nearest trading day: {adjusted_date}")
        target_date = adjusted_date

    # 检查历史数据是否足够
    target_index = data.index.get_loc(target_date)
    if target_index < lookback:
        raise ValueError(f"Not enough historical data for the lookback period of {lookback} days before {target_date}.")

    # 提取过去 lookback 天的数据
    input_data = data.iloc[target_index - lookback:target_index].values.reshape(-1, 1)

    # 归一化数据
    scaled_input, _ = preprocess_data(pd.DataFrame(input_data), scaler)
    input_array = np.reshape(scaled_input, (1, lookback, 1))
    print(f"Input data for {target_date} prepared successfully!")
    return input_array

# 4. 模型加载
def load_trained_model(model_path):
    """
    加载训练好的LSTM模型。
    """
    print(f"Loading model from {model_path}...")
    model = load_model(model_path)
    print("Model loaded successfully!")
    model.summary()
    return model

# 5. 单天预测
def predict_single_day(model, input_data, scaler):
    """
    使用模型预测指定日期的股票价格。
    """
    print("Predicting stock price...")
    prediction = model.predict(input_data)
    predicted_price = scaler.inverse_transform(prediction)
    print("Prediction complete!")
    return predicted_price[0][0]

# 6. 可视化结果
def visualize_prediction(target_date, actual_data, predicted_price):
    """
    绘制预测结果。
    """
    print("Visualizing prediction...")
    plt.figure(figsize=(10, 6))
    plt.title(f"Stock Price Prediction for {target_date}")
    plt.xlabel("Time")
    plt.ylabel("Close Price USD ($)")
    plt.plot(actual_data[-120:], label="Historical Data")
    plt.axhline(y=predicted_price, color='r', linestyle='--', label="Predicted Price")
    plt.legend()
    plt.show()
    print("Visualization complete!")

# 主测试代码
if __name__ == "__main__":
    # 配置参数
    TICKER = "AAPL"  # 股票代码
    START_DATE = "2024-06-01"  # 数据起始日期
    END_DATE = "2024-12-29"    # 数据结束日期
    TARGET_DATE = "2024-12-30"  # 目标预测日期
    MODEL_PATH = "APPLE_model.keras"  # 模型文件路径
    LOOKBACK = 120  # 回溯天数

    try:
        # 1. 加载数据
        data = load_data(TICKER, START_DATE, END_DATE)
        print(f"Available trading dates: {data.index.min()} to {data.index.max()}")

        # 2. 加载模型
        model = load_trained_model(MODEL_PATH)

        # 3. 预处理数据
        scaled_data, scaler = preprocess_data(data)

        # 4. 准备输入数据
        input_data = prepare_input(data, TARGET_DATE, lookback=LOOKBACK, scaler=scaler)

        # 5. 预测股票价格
        predicted_price = predict_single_day(model, input_data, scaler)
        print(f"Predicted stock price for {TARGET_DATE}: ${predicted_price:.2f}")

        # 6. 可视化结果
        visualize_prediction(TARGET_DATE, data, predicted_price)

    except Exception as e:
        print(f"An error occurred: {e}")