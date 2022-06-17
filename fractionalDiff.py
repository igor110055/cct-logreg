import pandas as pd 
import numpy as np

def getWeights(d,lags):
    w=[1]
    for k in range(1,lags):
        w.append(-w[-1]*((d-k+1))/k)
    w=np.array(w).reshape(-1,1)
    return w

def findCutoff(order,cutoff,start_lags): 
    val=np.inf
    lags=start_lags
    while abs(val)>cutoff:
        w = getWeights(order, lags)
        val=w[len(w)-1]
        lags = lags + 1
    return lags

def CalculateDifferencedVariables(series, order, thresholdVal, cutoff = False):
    if cutoff is True:
        lags = (findCutoff(order,thresholdVal,1))
        weights = getWeights(order, lags)
    else: 
        lags = thresholdVal
        weights = getWeights(order, lags)

    res=0
    for k in range(lags):
        res = res + weights[k]*series.shift(k).fillna(0)
    return res[lags:]


def FractionallyDifferentiateInputs(inputs, tickers, order, thresholdVal = 10-6, cutoff = True):

    transformedInputs = inputs.copy()

    for col in tickers:
        transformedInputs[col] = CalculateDifferencedVariables(inputs[col], order, thresholdVal, cutoff=cutoff)

    return transformedInputs