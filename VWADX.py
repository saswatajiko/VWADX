# -*- coding: utf-8 -*-
"""
@author: Saswata Banerjee
"""

import pandas as pd, numpy as np, matplotlib.pyplot as pyplot, yfinance as yf
pd.options.mode.chained_assignment = None

N = 14 # Number of Periods for Indicator Parameters
N1 = 8 # Fast Period for MACD
start_date = '2011-01-01'
end_date = '2020-12-31'
duration = 10 # 10 years backtest

# Use with period2 > period1 ONLY
def MACD(data: pd.DataFrame, period1: int, period2: int):
    df = data.copy()
    df['EMA1'] = df['Close'].ewm(alpha = 1 / period1, adjust = True).mean()
    df['EMA2'] = df['Close'].ewm(alpha = 1 / period2, adjust = True).mean()
    df['MACD'] = df['EMA1'] - df['EMA2']
    return df

def VWADX(data: pd.DataFrame, period: int):
    df = data.copy()
    expwts = [0] * period
    for i in range(0, period):
        expwts[i] = (1 - 1 / period) ** (period - i - 1)
    df['H-L'] = df['High'] - df['Low']
    df['H-C'] = np.abs(df['High'] - df['Close'].shift(1))
    df['L-C'] = np.abs(df['Low'] - df['Close'].shift(1))
    df['TR'] = df[['H-L', 'H-C', 'L-C']].max(axis=1)
    del df['H-L'], df['H-C'], df['L-C']
    df['AVWTR'] = df['TR']
    for i in range(period - 1, len(df['TR'])):
        df['AVWTR'][i] = np.average(df['TR'][i - period + 1: i + 1], weights = np.multiply(df['Volume'][i - period + 1: i + 1], expwts[:]))
    df['H-pH'] = df['High'] - df['High'].shift(1)
    df['pL-L'] = df['Low'].shift(1) - df['Low']
    df['+DX'] = np.where(
        (df['H-pH'] > df['pL-L']) & (df['H-pH'] > 0),
        df['H-pH'],
        0.0
    )
    df['-DX'] = np.where(
        (df['H-pH'] < df['pL-L']) & (df['pL-L'] > 0),
        df['pL-L'],
        0.0
    )
    del df['H-pH'], df['pL-L']
    df['VWS+DM'] = df['+DX'].ewm(alpha = 1 / period, adjust = True).mean()
    df['VWS-DM'] = df['-DX'].ewm(alpha = 1 / period, adjust = True).mean()
    for i in range(period - 1, len(df['+DX'])):
        df['VWS+DM'][i] = np.average(df['+DX'][i - period + 1: i + 1], weights = np.multiply(df['Volume'][i - period + 1: i + 1], expwts[:]))
        df['VWS-DM'][i] = np.average(df['-DX'][i - period + 1: i + 1], weights = np.multiply(df['Volume'][i - period + 1: i + 1], expwts[:]))
    df['VW+DMI'] = (df['VWS+DM'] / df['AVWTR'])*100
    df['VW-DMI'] = (df['VWS-DM'] / df['AVWTR'])*100
    del df['VWS+DM'], df['VWS-DM']
    df['VWDX'] = (np.abs(df['VW+DMI'] - df['VW-DMI']) / (df['VW+DMI'] + df['VW-DMI'])) * 100
    df['VWADX'] = df['VWDX'].ewm(alpha = 1 / period, adjust = True).mean()
    for i in range(period - 1, len(df['VWDX'])):
        df['VWADX'][i] = np.average(df['VWDX'][i - period + 1: i + 1], weights = np.multiply(df['Volume'][i - period + 1: i + 1], expwts[:]))
    del df['AVWTR'], df['VWDX'], df['TR'], df['-DX'], df['+DX'], df['VW+DMI'], df['VW-DMI']
    return df

def ADX(data: pd.DataFrame, period: int):
    df = data.copy()
    df['H-L'] = df['High'] - df['Low']
    df['H-C'] = np.abs(df['High'] - df['Close'].shift(1))
    df['L-C'] = np.abs(df['Low'] - df['Close'].shift(1))
    df['TR'] = df[['H-L', 'H-C', 'L-C']].max(axis=1)
    del df['H-L'], df['H-C'], df['L-C']
    df['ATR'] = df['TR'].ewm(alpha = 1 / period, adjust = False).mean()
    df['H-pH'] = df['High'] - df['High'].shift(1)
    df['pL-L'] = df['Low'].shift(1) - df['Low']
    df['+DX'] = np.where(
        (df['H-pH'] > df['pL-L']) & (df['H-pH']>0),
        df['H-pH'],
        0.0
    )
    df['-DX'] = np.where(
        (df['H-pH'] < df['pL-L']) & (df['pL-L']>0),
        df['pL-L'],
        0.0
    )
    del df['H-pH'], df['pL-L']
    df['S+DM'] = df['+DX'].ewm(alpha = 1 / period, adjust = False).mean()
    df['S-DM'] = df['-DX'].ewm(alpha = 1 / period, adjust = False).mean()
    df['+DMI'] = (df['S+DM'] / df['ATR']) * 100
    df['-DMI'] = (df['S-DM'] / df['ATR']) * 100
    del df['S+DM'], df['S-DM']
    df['DX'] = (np.abs(df['+DMI'] - df['-DMI'])/(df['+DMI'] + df['-DMI'])) * 100
    df['ADX'] = df['DX'].ewm(alpha = 1 / period, adjust = False).mean()
    del df['ATR'], df['DX'], df['TR'], df['-DX'], df['+DX'], df['+DMI'], df['-DMI']
    return df

def trend_macd(data: pd.DataFrame):
    df = data.copy()
    df['trend'] = np.sign(df['MACD'])
    df['trend_long'] =  np.where(
        df['trend'] == 1.0,
        1.0,
        0.0
    )
    return df

def tradesig_adx(data: pd.DataFrame):
    df = data.copy()
    df['sig_ADX'] = np.where(
        df['ADX'] > 25,
        1.0,
        0.0
    )
    return df

def tradesig_vwadx(data: pd.DataFrame):
    df = data.copy()
    df['sig_VWADX'] = np.where(
        df['VWADX'] > 50,
        1.0,
        0.0
    )
    return df

def trade(data: pd.DataFrame):
    df = data.copy()
    df = ADX(df, N)
    df = VWADX(df, N)
    df = MACD(df, N1, N)
    df = trend_macd(df)
    df = tradesig_adx(df)
    df = tradesig_vwadx(df)
    df['trade_ADX'] = df['trend_long'] * df['sig_ADX']
    df['trade_VWADX'] = df['trend_long'] * df['sig_VWADX']
    df['ret'] = (df['Adj Close'] - df['Adj Close'].shift(1)) / df['Adj Close'].shift(1)
    df['ret_ADX'] = df['ret'] * df['trade_ADX']
    df['ret_VWADX'] = df['ret'] * df['trade_VWADX']
    df['port'] = [1.0] * len(df['ret'])
    df['port_ADX'] = [1.0] * len(df['ret'])
    df['port_VWADX'] = [1.0] * len(df['ret'])
    for i in range(1, len(df['ret'])):
        df['port'][i] = df['port'][i - 1] * (1 + df['ret'][i])
        df['port_ADX'][i] = df['port_ADX'][i - 1] * (1 + df['ret_ADX'][i])
        df['port_VWADX'][i] = df['port_VWADX'][i - 1] * (1 + df['ret_VWADX'][i])
    del df['ret'], df['ret_ADX'], df['ret_VWADX'], df['trade_ADX'], df['trade_VWADX']
    return df

tickers = 'AXISBANK.NS ADANIPORTS.NS ASIANPAINT.NS BAJAJ-AUTO.NS BAJFINANCE.NS BAJAJFINSV.NS BPCL.NS BHARTIARTL.NS BRITANNIA.NS CIPLA.NS COALINDIA.NS DIVISLAB.NS DRREDDY.NS EICHERMOT.NS GRASIM.NS HCLTECH.NS HDFCBANK.NS HEROMOTOCO.NS HINDALCO.NS HINDUNILVR.NS ICICIBANK.NS ITC.NS IOC.NS INDUSINDBK.NS INFY.NS JSWSTEEL.NS KOTAKBANK.NS LT.NS M&M.NS MARUTI.NS NTPC.NS NESTLEIND.NS ONGC.NS POWERGRID.NS RELIANCE.NS SBIN.NS SHREECEM.NS SUNPHARMA.NS TCS.NS TATACONSUM.NS TATAMOTORS.NS TATASTEEL.NS TECHM.NS TITAN.NS UPL.NS ULTRACEMCO.NS WIPRO.NS'
ticker_list = tickers.split(' ')
stock_data = []
for i in range(len(ticker_list)):
    stock_data.append(yf.download(ticker_list[i], start = start_date, end = end_date))
    stock_data[i] = trade(stock_data[i])

performance = {'Ticker':[], 'Buy_and_Hold_CAGR':[], 'ADX_CAGR':[], 'VWADX_CAGR':[]}
for i in range(len(ticker_list)):
    performance['Ticker'].append(ticker_list[i])
    performance['Buy_and_Hold_CAGR'].append(((stock_data[i]['port'][-1] ** (1 / duration)) - 1) * 100)
    performance['ADX_CAGR'].append(((stock_data[i]['port_ADX'][-1] ** (1 / duration)) - 1) * 100)
    performance['VWADX_CAGR'].append(((stock_data[i]['port_VWADX'][-1] ** (1 / duration)) - 1) * 100)

perf = pd.DataFrame.from_dict(performance)
perf = perf.set_index('Ticker')
perf.to_csv('Strategy Performance.csv')