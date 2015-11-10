# -*- coding: utf-8 -*-
u"""
Created on 2015-11-9

@author: cheng.li
"""

import pandas as pd
import datetime as dt
from math import sqrt
from math import exp
from PyFin.Math.Accumulators import MovingDrawDown

APPROX_BDAYS_PER_YEAR = 252.


def cumReturn(returns):

    dfCum = returns.cumsum()
    return dfCum


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

    return pos.divide(
        pos.abs().sum(axis='columns'),
        axis='rows'
    )


def aggregateReturns(returns, convert='daily'):

    def cumulateReturns(x):
        return cumReturn(x)[-1]

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

