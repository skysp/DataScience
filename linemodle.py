import numpy as np
import pandas as pd
import matplotlib.pyplot as mp
import datetime as dt
import matplotlib.dates as md


# 日期格式转换：日月年 to 年月日
def dmy2ymd(dmy):
    dmy = str(dmy, encoding='utf-8')
    time = dt.datetime.strptime(dmy, '%d-%m-%Y').date()
    t = time.strftime('%Y-%m-%d')
    return t


# 读取日期和收盘价
[dates, closing_prices] = \
    np.loadtxt('aapl.csv', usecols=(1, 6),
               unpack=True, dtype='M8[D], f8',
               delimiter=',', converters={1: dmy2ymd})

# 用前5天数据预测第6天的数据
N = 5
# 生成预测值数组：如果窗口为5，只能预测出第11个值，前10个值都无法预测，因为前10个数据都是基础数据
# 前2N个数无法预测，最后再多预测1个；
# 先创建为零数组
pred_prices = np.zeros(closing_prices.size - N * 2 + 1)
# 再填入数据
for i in range(pred_prices.size):
    # 先生成一个n行n列的零矩阵
    a = np.zeros((N, N))
    # 再填充这个矩阵，按行放入数据
    for j in range(N):
        # 填入a的第j行，为收盘价的切片；再加上第i个预测值的偏移量；
        a[j, ] = closing_prices[i + j:i + j + N]
    b = closing_prices[i + N:i + N * 2]
    # 用线性代数的最小二乘法求解线性方程组
    x = np.linalg.lstsq(a, b)[0]
    # print(x)
    # 计算出每一个预测值
    pred_prices[i] = b.dot(x)

# 绘图基本配置
mp.figure('Linear Prediction', facecolor='lightgray')
mp.title('Linear Prediction', fontsize=20)
mp.xlabel('Date', fontsize=14)
mp.ylabel('Price', fontsize=14)
mp.grid(linestyle=':')
mp.tick_params(labelsize=10)
# 绘制刻度定位器
ax = mp.gca()
ax.xaxis.set_major_locator(md.WeekdayLocator(byweekday=md.MO))
ax.xaxis.set_major_formatter(md.DateFormatter('%d %b %Y'))
ax.xaxis.set_minor_locator(md.DayLocator())
# matplotlib中的datetime数据类型绘图是效果最好
dates = dates.astype(md.datetime.datetime)
# 先画出收盘价曲线
mp.plot(dates, closing_prices, 'o-', c='lightgray', label='Closing Price')
# 取最后一天的下一个工作日，并追加到日期当中
dates = np.append(dates, dates[-1] + pd.tseries.offsets.BDay())
# 绘制预测线
mp.plot(dates[N * 2:], pred_prices, 'o-', c='orangered', linewidth=3, label='Predicted Price')

mp.legend()
mp.gcf().autofmt_xdate()
mp.tight_layout()
mp.show()
