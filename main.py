from fastapi import FastAPI, Form, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
import random

app = FastAPI()

# 设置模板和静态文件目录
templates = Jinja2Templates(directory="templates")
app.mount("/static", StaticFiles(directory="static"), name="static")

# 首页路由
@app.get("/", response_class=HTMLResponse)
def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

# 预测路由
@app.post("/predict", response_class=HTMLResponse)
def predict_stock(request: Request, stock_code: str = Form(...), date: str = Form(...)):
    # 模拟预测价格（实际应调用你的模型）
    predicted_price = round(random.uniform(100, 500), 2)
    return templates.TemplateResponse(
        "index.html",
        {"request": request, "stock_code": stock_code, "date": date, "predicted_price": predicted_price},
    )