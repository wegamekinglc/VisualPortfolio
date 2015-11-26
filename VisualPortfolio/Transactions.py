# -*- coding: utf-8 -*-
u"""
Created on 2015-11-10

@author: cheng.li
"""


def getTurnOver(transactions, positions, period=None, average=True):

    tradedValue = transactions.turnover_value
    portfolioValue = positions.abs().sum(axis=1)
    portfolioValue[portfolioValue == 0] = portfolioValue.mean()
    if period:
        tradedValue = tradedValue.resample(period, how="sum")
        portfolioValue = portfolioValue.resample(period, how="mean")

    turnover = tradedValue / 2.0 if average else tradedValue
    turnoverRate = turnover.div(portfolioValue, axis='index')
    turnoverRate.fillna(0.0, inplace=True)
    return turnoverRate