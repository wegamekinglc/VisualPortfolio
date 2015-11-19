# -*- coding: utf-8 -*-
u"""
Created on 2015-11-9

@author: cheng.li
"""

import numpy as np
import pandas as pd
import datetime as dt
from math import sqrt
from math import exp
from PyFin.Math.Accumulators import MovingDrawDown
from PyFin.Math.Accumulators import MovingAlphaBeta
from PyFin.Math.Accumulators import MovingSharp

APPROX_BDAYS_PER_MONTH = 21
APPROX_BDAYS_PER_YEAR = 252.


def aggregatePositons(positionBooks, convert='daily'):

    if convert == 'daily':
        resampled_pos = positionBooks.groupby(
            lambda x: dt.datetime(x.year, x.month, x.day)).last()

    return resampled_pos


def aggregateTranscations(transcations, convert='daily'):
    transcations = transcations[['turnover_volume', 'turnover_value']].abs()
    if convert == 'daily':
        resampled_pos = transcations.groupby(
            lambda x: dt.datetime(x.year, x.month, x.day)).sum()

    return resampled_pos


def calculatePosWeight(pos):

    pos_wo_cash = pos.drop('cash', axis=1)
    longs = pos_wo_cash[pos_wo_cash > 0].sum(axis=1).fillna(0)
    shorts = pos_wo_cash[pos_wo_cash < 0].abs().sum(axis=1).fillna(0)

    cash = pos.cash
    net_liquidation = longs + shorts + cash

    return pos.divide(
        net_liquidation,
        axis='index'
    )


def aggregateReturns(returns, convert='daily'):

    def cumulateReturns(x):
       return x.sum()

    if convert == 'daily':
        return returns.groupby(
            lambda x: dt.datetime(x.year, x.month, x.day)).apply(cumulateReturns)
    if convert == 'monthly':
        return returns.groupby(
            [lambda x: x.year,
             lambda x: x.month]).apply(cumulateReturns)
    if convert == 'yearly':
        return returns.groupby(
            [lambda x: x.year]).apply(cumulateReturns)
    else:
        ValueError('convert must be daily, weekly, monthly or yearly')


def drawDown(returns):

    ddCal = MovingDrawDown(len(returns), 'ret')
    length = len(returns)
    ddSeries = [0.0] * length
    peakSeries = [0] * length
    valleySeries = [0] * length
    recoverySeries = [returns.index[-1]] * length
    for i, value in enumerate(returns):
        ddCal.push({'ret': value})
        res = ddCal.value
        ddSeries[i] = exp(res[0]) - 1.0
        peakSeries[i] = returns.index[res[2]]
        valleySeries[i] = returns.index[i]

    for i, value in enumerate(ddSeries):
        for k in range(i, length):
            if ddSeries[k] == 0.0:
                recoverySeries[i] = returns.index[k]
                break

    df = pd.DataFrame(list(zip(ddSeries, peakSeries, valleySeries, recoverySeries)),
                      index=returns.index,
                      columns=['draw_down', 'peak', 'valley', 'recovery'])
    return df


def annualReturn(returns):
    return returns.mean() * APPROX_BDAYS_PER_YEAR


def annualVolatility(returns):
    return returns.std() * sqrt(APPROX_BDAYS_PER_YEAR)


def sortinoRatio(returns):
    annualRet = annualReturn(returns)
    annualNegativeVol = annualVolatility(returns[returns < 0.0])
    return annualRet / annualNegativeVol


def sharpRatio(returns):
    annualRet = annualReturn(returns)
    annualVol = annualVolatility(returns)
    return annualRet / annualVol


def RollingBeta(returns, benchmarkReturns, month_windows):

    def calculateSingalWindowBete(returns, benchmarkReturns, window):
        res = []
        rbcalc = MovingAlphaBeta(window=window * APPROX_BDAYS_PER_MONTH)
        for pRet, mRet in zip(returns, benchmarkReturns):
            rbcalc.push({'pRet': pRet, 'mRet': mRet, 'riskFree': 0})
            try:
                res.append(rbcalc.result()[1])
            except ZeroDivisionError:
                res.append(np.nan)
        return res

    rtn = [pd.Series(calculateSingalWindowBete(returns, benchmarkReturns, window), index=returns.index)
           for window in month_windows]

    return {"beta_{0}m".format(window): res[APPROX_BDAYS_PER_MONTH*min(month_windows):] for window, res in zip(month_windows, rtn)}


def RollingSharp(returns, month_windows):

    def calculateSingalWindowSharp(returns, window):
        res = []
        rscalc = MovingSharp(window=window * APPROX_BDAYS_PER_MONTH)
        for ret in returns:
            rscalc.push({'ret': ret, 'riskFree': 0})
            try:
                # in PyFin, sharp is not annualized
                res.append(rscalc.result() * sqrt(APPROX_BDAYS_PER_YEAR))
            except ZeroDivisionError:
                res.append(np.nan)
        return res

    rtn = [pd.Series(calculateSingalWindowSharp(returns, window), index=returns.index)
           for window in month_windows]

    return {"sharp_{0}m".format(window): res[APPROX_BDAYS_PER_MONTH*min(month_windows):] for window, res in zip(month_windows, rtn)}
