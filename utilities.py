import pandas as pd 
import numpy as np
import datetime
import sklearn.linear_model as skl_lm
import matplotlib.pyplot as plt 

def CalculateReturns(x):
    df = x.copy()
    for i in range(1, df.shape[0]):
        df.iloc[i] = (df.iloc[i] - df.iloc[i - 1])

    df.drop(df.head(1).index,inplace=True)
    return df 

def StdRolling(x, window):
    rollSd = x.copy()
    assets = x.columns
    for zone in assets:
        rollSd[zone] = x[zone].rolling(window = window).std()

    return rollSd

def DeStandardizeDf(df, distrParams):
    scaledDf = df.drop(columns=['date']).sub(np.array(distrParams['avg']), axis = 'columns')/np.array(distrParams['sd'])
    return scaledDf

def StandardizeDf(df, distrParams = None):
    if distrParams is None:
        distrParams = {'avg' : np.mean(df.drop(columns=['date']), axis = 0), 'sd': np.std(df.drop(columns=['date']))}
        scaledDf = df.drop(columns=['date']).sub(np.array(distrParams['avg']), axis = 'columns')/np.array(distrParams['sd'])
    else:
        scaledDf = df.drop(columns=['date']).sub(np.array(distrParams['avg']), axis = 'columns')/np.array(distrParams['sd'])

    return scaledDf

def RollData(y):
    y = np.roll(y, shift = -1)[0:(y.shape[0] - 1)]
    #y = np.roll(y, shift = -1)
    return y 

def FilterInputs(inputs, startDate, stopDate):
    if inputs.columns[0] == 'Unnamed: 0':
        dfToScale = inputs[(inputs.date<=stopDate) & (inputs.date>=startDate)].iloc[:, 1:].copy()
    else:
        dfToScale = inputs[(inputs.date<=stopDate) & (inputs.date>=startDate)].copy()
    return dfToScale

def SelectLogicOutputs(startDate, stopDate, ticker, logicOutputs):
    y = logicOutputs.loc[(logicOutputs.date<=stopDate) & (logicOutputs.date>=startDate), ticker].values
    y = RollData(y)
    return y 

def SelectAndScaleInputs(dfToScale, alreadyScaled = False):   
    if alreadyScaled is False:
        scaledDf = StandardizeDf(dfToScale)
        # scaledDf = dfToScale.drop(columns=['date'])
    else:
        scaledDf = dfToScale.drop(columns=['date'])

    x = scaledDf.values
    x = x[0:(x.shape[0] - 1), :]
    return x


def SelectAndScale(dfToScale, logicOutputs, startDate, stopDate, ticker, alreadyScaled = False):   
    if alreadyScaled is False:
        scaledDf = StandardizeDf(dfToScale)
        # scaledDf = dfToScale.drop(columns=['date'])
    else:
        scaledDf = dfToScale.drop(columns=['date'])

    x = scaledDf.values
    x = x[0:(x.shape[0] - 1), :]
    y = logicOutputs.loc[(logicOutputs.date<=stopDate) & (logicOutputs.date>=startDate), ticker].values
    y = RollData(y)
    return x, y

def TestTrainSplit(x, y, nTrain):
    xTrain = x[0:nTrain]
    yTrain = y[0:nTrain]
    xTest = x[nTrain:]
    yTest = y[nTrain:]
    return xTrain, xTest, yTrain, yTest

def CalculateDayDirectionSimple(time, priceTs):
    direction = 1 * (priceTs.c.values[time] - priceTs.o.values[time] >= 0.) - 1 * (priceTs.c.values[time] - priceTs.o.values[time] < 0.) 
    ret_t =priceTs.c.values[time] - priceTs.o.values[time]
    return direction, ret_t

def CalculateDayDirectionCustom(time, priceTs):
    direction = 1 * (priceTs.c.values[time] - priceTs.o.values[time] >= 0.) * (priceTs.h.values[time] + priceTs.l.values[time] - 2 * priceTs.o.values[time] > 0.) - \
        1 * (priceTs.c.values[time] - priceTs.o.values[time] < 0.) * (priceTs.h.values[time] + priceTs.l.values[time] - 2 * priceTs.o.values[time] < 0.)
    ret_t =priceTs.c.values[time] - priceTs.o.values[time]
    return direction, ret_t

def CalculateOverallDirection(prices, tickers, method = 'custom'):
    calculationMethod = {
        'custom': CalculateDayDirectionCustom,
        'simple': CalculateDayDirectionSimple
    }
    CalculatePointDirection = calculationMethod[method]
    returnsWithDirsLst = []
    for k, ticker in enumerate(tickers):
        ret = []
        dirs = []
        priceTs = prices.loc[prices.ticker == ticker]
        
        for i in range(0, priceTs.shape[0]):
            #dir_i, ret_i = CalculatePointDirection(i, priceTs)
            dir_i, _ = CalculatePointDirection(i, priceTs)
            ret_i = priceTs.c.values[i]
            ret.append(ret_i)
            dirs.append(dir_i)

        retDirDf = pd.DataFrame(data = np.array([ret, dirs]).T, columns = ['ret', 'dirs'])
        retDirDf['ticker'] = ticker
        retDirDf['date'] = priceTs.date.reset_index()['date']
        #print(p.date.reset_index()['date'])
        retDirDf = retDirDf[['ticker', 'date', 'ret', 'dirs']]
        returnsWithDirsLst.append(retDirDf)
        returnsWithDirs = pd.concat(returnsWithDirsLst, axis = 0)

        #print(f'{k}-th weight: {w[k]}; total ret {np.sum(retDf.ret)}')
    return returnsWithDirs


def LogLoss(beta, x, y, verbose = False):
    logLoss = 0.0
    for row in range(x.shape[0]):
        p_k = []
        w_k = []
        Z = 0.0
        for k, _ in enumerate(nominalValues):
            Z = Z + np.exp(-x[row,:] @ beta[k * (x.shape[1]) : (k + 1) * x.shape[1]])
            w_k.append(np.exp(-x[row,:] @ beta[k * (x.shape[1]) : (k + 1) * x.shape[1]]))
        p_k = w_k / Z
        logLoss = logLoss - np.sum([ (value == y[row]) * np.log(p_k[i]) - (1.0 - (value == y[row])) * np.log(1.0 - p_k[i]) for i, value in enumerate(nominalValues)])
        if verbose == True: print(logLoss)
    return logLoss

def CalculateAccuracy(x, y, beta, prob = False):
    if prob == False:
        p_k = []
        for row in range(x.shape[0]):
            p_k.append([np.exp(-x[row,:] @ b) / np.sum([np.exp(-x[row,:] @ g)  for g in beta]) for b in beta])
    else:
        p_k = beta 
    
    df = pd.DataFrame()
    df[[f'p_{value}' for value in nominalValues]] = p_k
    df['y'] = y
    df['prediction'] = np.sum(p_k*nominalValues, axis = 1)
    df['error'] = 1 - (np.rint(df.prediction).astype(int) == df.y.astype(int))

    return df 









# def CalculatePortfolioReturnsConservatively(w, slBuy, tpBuy, slSell, tpSell):
#     returnsWithPricesLst = []
#     for k, ticker in enumerate(tickers):
#         ret = []
#         p = prices.loc[prices.ticker == ticker]
        
#         for i in range(0, p.shape[0]):
#             lossWithBuy = (w[k] > 0) * (np.absolute(p.l.values[i] - p.o.values[i]) > slBuy) * (p.l.values[i] - p.o.values[i]) * w[k]
#             gainWithBuy = (w[k] > 0) * (np.absolute(p.h.values[i] - p.o.values[i]) > np.absolute(tpBuy)) * \
#                 (np.absolute(p.l.values[i] - p.o.values[i]) < slBuy)* (p.h.values[i] - p.o.values[i]) * w[k]

#             lossWithSell = (w[k] < 0) * (p.h.values[i] - p.o.values[i] > slSell) * (p.h.values[i] - p.o.values[i]) * w[k]
#             gainWithSell = (w[k] < 0) * ((p.l.values[i] - p.o.values[i]) > tpSell) * \
#                 ((p.h.values[i] - p.o.values[i]) < slSell) * (p.l.values[i] - p.o.values[i]) * w[k]

#             ret_i = lossWithBuy + lossWithSell + gainWithBuy + gainWithSell
#             ret.append(ret_i)
#         retDf = pd.DataFrame(data = ret, columns = ['ret'])
#         retDf['ticker'] = ticker
#         retDf['date'] = p.date.reset_index()['date']
#         #print(p.date.reset_index()['date'])
#         retDf = retDf[['ticker', 'date', 'ret']]
#         returnsWithPricesLst.append(retDf)
#         returnsWithPrices = pd.concat(returnsWithPricesLst, axis = 0)

#         print(f'{k}-th weight: {w[k]}; total ret {np.sum(retDf.ret)}')
#     return -np.sum(returnsWithPrices.ret)