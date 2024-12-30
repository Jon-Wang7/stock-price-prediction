from fastapi import FastAPI, Form, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from datetime import datetime, timedelta
import yfinance as yf
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
import io
import base64
import logging

# 设置日志级别
logging.basicConfig(level=logging.DEBUG)

# FastAPI 配置
app = FastAPI()
templates = Jinja2Templates(directory="templates")
app.mount("/static", StaticFiles(directory="static"), name="static")

# 公司与股票代码
COMPANIES = {
    "apple": "AAPL",
    "amazon": "AMZN",
    "google": "GOOGL",
    "ibm": "IBM",
    "microsoft": "MSFT"
}

# 处理 NaN 值的函数
def handle_nan_values(array):
    """
    处理数组中的 NaN 值，使用前后有效值的平均值替换 NaN。
    """
    for i in range(len(array)):
        if np.isnan(array[i]):
            prev_value = array[i - 1] if i > 0 else None
            next_value = array[i + 1] if i < len(array) - 1 else None

            if prev_value is not None and not np.isnan(prev_value) and next_value is not None and not np.isnan(next_value):
                array[i] = (prev_value + next_value) / 2
            elif prev_value is not None and not np.isnan(prev_value):
                array[i] = prev_value
            elif next_value is not None and not np.isnan(next_value):
                array[i] = next_value
    return array

# 数据加载
def load_data(ticker, start_date, end_date):
    adjusted_start_date = (start_date - pd.Timedelta(days=240)).strftime("%Y-%m-%d")
    stock_df = yf.download(ticker, start=adjusted_start_date, end=end_date)
    if stock_df.empty:
        raise ValueError(f"No data found for ticker {ticker} in the specified date range.")
    stock_df = stock_df.asfreq('B')
    stock_df['Close'].fillna(method='ffill', inplace=True)
    stock_df['Close'].fillna(method='bfill', inplace=True)
    logging.debug(f"Loaded data from {stock_df.index.min()} to {stock_df.index.max()}")
    return process_stock_df(stock_df)

# 数据处理
def process_stock_df(stock_df):
    """
    遍历 stock_df，将包含任何 NaN 值的行移除，返回一个新的 DataFrame。
    """
    # 创建一个空的列表，用于保存不包含 NaN 的行
    stock_df_copy = []

    # 遍历每一行，检查是否有 NaN
    for index, row in stock_df.iterrows():
        if not row.isna().any():  # 如果当前行不包含 NaN
            stock_df_copy.append(row)  # 添加到结果列表

    # 将结果列表转为 DataFrame
    stock_df_copy = pd.DataFrame(stock_df_copy, columns=stock_df.columns)

    return stock_df_copy

# 数据预处理
def preprocess_data(data):
    """
    对数据进行归一化处理。
    """
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data.values.reshape(-1, 1))
    return scaled_data, scaler

# 加载模型
def load_model_file(company):
    """
    加载训练好的模型。
    """
    model_path = f"./models/{company.upper()}_model.keras"
    model = load_model(model_path)
    model.summary()  # 打印模型结构
    return model

# 逐日预测
def predict_range(model, data, scaler, start_date, end_date, lookback=120):
    """
    对指定时间范围的每一天逐日预测。
    """
    predictions = []
    dates = []
    for single_date in pd.date_range(start=start_date, end=end_date):
        if single_date not in data.index:
            continue
        current_index = data.index.get_loc(single_date)
        if current_index < lookback:
            logging.warning(f"Not enough data to predict {single_date}. Skipping...")
            continue

        # 获取前120天的数据作为输入
        input_data = data.iloc[current_index - lookback:current_index].values.reshape(-1, 1)
        if np.isnan(input_data).any():
            logging.warning(f"NaN detected in input data for {single_date}. Handling NaN...")
            input_data = handle_nan_values(input_data)

        scaled_input = scaler.transform(input_data)
        scaled_input = scaled_input.reshape(1, lookback, 1)
        pred = model.predict(scaled_input)

        if np.isnan(pred).any():
            logging.warning(f"NaN detected in prediction for {single_date}. Skipping...")
            continue

        predicted_price = scaler.inverse_transform(pred)[0][0]
        predictions.append(predicted_price)
        dates.append(single_date)

    return pd.Series(predictions, index=dates)

# 绘制图表
def generate_plot(actual, predictions, start_date, end_date):
    """
    绘制预测结果图表。
    """
    plt.figure(figsize=(12, 6))
    actual_dates = actual[start_date:end_date].index
    actual_values = actual[start_date:end_date].values
    plt.plot(actual_dates, actual_values, label="Actual Prices", color="blue")
    predicted_dates = predictions.index
    predicted_values = predictions.values
    plt.plot(predicted_dates, predicted_values, label="Predicted Prices", color="red", linestyle="--")
    plt.title("Stock Price Prediction")
    plt.xlabel("Date")
    plt.ylabel("Price (USD)")
    plt.legend()
    plt.grid()
    buffer = io.BytesIO()
    plt.savefig(buffer, format="png")
    buffer.seek(0)
    img_base64 = base64.b64encode(buffer.read()).decode("utf-8")
    buffer.close()
    return img_base64

# 主页面
@app.get("/", response_class=HTMLResponse)
def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request, "companies": COMPANIES})

# 预测接口
@app.post("/predict", response_class=HTMLResponse)
def predict(
    request: Request,
    company: str = Form(...),
    start_date: str = Form(...),
    end_date: str = Form(...)
):
    try:
        # 转换日期
        start_date = pd.to_datetime(start_date)
        end_date = pd.to_datetime(end_date)

        if start_date >= end_date:
            raise ValueError("Start date must be earlier than end date.")

        # 加载数据
        ticker = COMPANIES[company]
        stock_data = load_data(ticker, start_date, end_date)
        actual_prices = stock_data["Close"]
        if len(actual_prices) < 120:
            raise ValueError("Not enough data: At least 120 days of historical data are required.")

        # 数据预处理
        scaled_data, scaler = preprocess_data(actual_prices)

        # 加载模型并逐日预测
        model = load_model_file(company)

        predictions = predict_range(model, stock_data["Close"], scaler, start_date, end_date)

        # 绘制图表
        plot_base64 = generate_plot(
            actual=actual_prices,
            predictions=predictions,
            start_date=start_date,
            end_date=end_date
        )

        return templates.TemplateResponse("index.html", {
            "request": request,
            "companies": COMPANIES,
            "plot_base64": plot_base64
        })
    except Exception as e:
        logging.error(f"Error occurred: {e}")
        return templates.TemplateResponse("index.html", {
            "request": request,
            "companies": COMPANIES,
            "error": str(e)
        })

# 调试模式运行
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="127.0.0.1",
        port=8000,
        reload=True,
        log_level="debug"
    )