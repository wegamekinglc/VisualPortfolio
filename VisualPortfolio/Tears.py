# -*- coding: utf-8 -*-
u"""
Created on 2015-11-9

@author: cheng.li
"""

import os
import numpy as np
import pandas as pd
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
from VisualPortfolio.Timeseries import aggregateReturns
from VisualPortfolio.Timeseries import drawDown
from VisualPortfolio.Plottings import plottingRollingReturn
from VisualPortfolio.Plottings import plottingDrawdownPeriods
from VisualPortfolio.Plottings import plottingRollingBeta
from VisualPortfolio.Plottings import plottingRollingSharp
from VisualPortfolio.Plottings import plottingUnderwater
from VisualPortfolio.Plottings import plottingMonthlyReturnsHeapmap
from VisualPortfolio.Plottings import plottingAnnualReturns
from VisualPortfolio.Plottings import plottingMonthlyRetDist
from VisualPortfolio.Plottings import plotting_context
from VisualPortfolio.Timeseries import annualReturn
from VisualPortfolio.Timeseries import annualVolatility
from VisualPortfolio.Timeseries import sortinoRatio
from VisualPortfolio.Timeseries import sharpRatio
from VisualPortfolio.Timeseries import aggregatePositons
from VisualPortfolio.Timeseries import calculatePosWeight
from VisualPortfolio.Timeseries import aggregateTranscations
from VisualPortfolio.Plottings import plottingExposure
from VisualPortfolio.Plottings import plottingTopExposure
from VisualPortfolio.Plottings import plottingHodings
from VisualPortfolio.Plottings import plottingTurnover
from VisualPortfolio.Timeseries import APPROX_BDAYS_PER_MONTH
from VisualPortfolio.Timeseries import RollingBeta
from VisualPortfolio.Timeseries import RollingSharp
import tushare as ts
from PyFin.api import advanceDateByCalendar
from PyFin.Enums import BizDayConventions


@plotting_context
def createPerformanceTearSheet(prices=None, returns=None, benchmark=None, benchmarkReturns=None, plot=True):

    if prices is not None and not isinstance(prices, pd.Series):
        raise TypeError("prices series should be a pandas time series.")
    elif returns is not None and prices is not None:
        raise ValueError("prices series and returns series can't be both set.")

    if benchmark is not None and not (isinstance(benchmark, pd.Series) or isinstance(benchmark, str)):
        raise TypeError("benchmark series should be a pandas time series or a string ticker.")

    if returns is None:
        returns = np.log(prices / prices.shift(1))
        returns.dropna(inplace=True)
        returns = returns[~np.isinf(returns)]

    if benchmark is not None and isinstance(benchmark, str) and benchmarkReturns is None:
        startDate = advanceDateByCalendar("China.SSE", returns.index[0], '-1b', BizDayConventions.Preceding)
        try:
            token = os.environ['DATAYES_TOKEN']
            ts.set_token(token)
        except KeyError:
            raise ValueError("Please input token or set up DATAYES_TOKEN in the envirement.")

        benchmarkPrices = ts.Market().MktIdxd(indexID=benchmark,
                                              beginDate=startDate.strftime('%Y%m%d'),
                                              endDate=returns.index[-1].strftime("%Y%m%d"),
                                              field='tradeDate,closeIndex')
        benchmarkPrices['tradeDate'] = pd.to_datetime(benchmarkPrices['tradeDate'], format="%Y-%m-%d")
        benchmarkPrices.set_index('tradeDate', inplace=True)
        benchmarkPrices.columns = ['close']
        benchmarkReturns = np.log(benchmarkPrices['close'] / benchmarkPrices['close'].shift(1))
        benchmarkReturns.name = benchmark
        benchmarkReturns.dropna(inplace=True)
        benchmarkReturns.index = pd.to_datetime(benchmarkReturns.index.date)
    elif benchmark is not None and isinstance(benchmark, pd.Series):
        benchmarkReturns = np.log(benchmark / benchmark.shift(1))
        try:
            benchmarkReturns.name = benchmark.name
        except AttributeError:
            benchmarkReturns.name = "benchmark"
        benchmarkReturns.dropna(inplace=True)
        benchmarkReturns.index = pd.to_datetime(benchmarkReturns.index.date)

    aggregateDaily = aggregateReturns(returns)
    drawDownDaily = drawDown(aggregateDaily)

    # perf metric
    annualRet = annualReturn(aggregateDaily)
    annualVol = annualVolatility(aggregateDaily)
    sortino = sortinoRatio(aggregateDaily)
    sharp = sharpRatio(aggregateDaily)
    maxDrawDown = np.min(drawDownDaily['draw_down'])
    winningDays = np.sum(aggregateDaily > 0.)
    lossingDays = np.sum(aggregateDaily < 0.)

    perf_metric = pd.DataFrame([annualRet, annualVol, sortino, sharp, maxDrawDown, winningDays, lossingDays],
                                index=['annual_return',
                                       'annual_volatiltiy',
                                       'sortino_ratio',
                                       'sharp_ratio',
                                       'max_draw_down',
                                       'winning_days',
                                       'lossing_days'],
                                columns=['metrics'])

    perf_df = pd.DataFrame(index=aggregateDaily.index)
    perf_df['daily_return'] = aggregateDaily
    perf_df['daily_cum_return'] = np.exp(aggregateDaily.cumsum()) - 1.0
    perf_df['daily_draw_down'] = drawDownDaily['draw_down']

    if benchmarkReturns is not None:
        perf_df['benchmark_return'] = benchmarkReturns
        perf_df['benchmark_cum_return'] = benchmarkReturns.cumsum()
        perf_df.dropna(inplace=True)
        perf_df['benchmark_cum_return'] = np.exp(perf_df['benchmark_cum_return']
                                                     - perf_df['benchmark_cum_return'][0]) - 1.0
        perf_df['access_return'] = aggregateDaily - benchmarkReturns
        perf_df['access_cum_return'] = (1.0 + perf_df['daily_cum_return']) \
                                           / (1.0 + perf_df['benchmark_cum_return']) - 1.0
        perf_df.fillna(0.0, inplace=True)
        accessDrawDownDaily = drawDown(perf_df['access_return'])
    else:
        accessDrawDownDaily = None

    if 'benchmark_cum_return' in perf_df:
        benchmarkCumReturns = perf_df['benchmark_cum_return']
        benchmarkCumReturns.name = benchmarkReturns.name
        accessCumReturns = perf_df['access_cum_return']
        accessReturns = perf_df['access_return']
        rb = RollingBeta(perf_df['daily_return'], perf_df['benchmark_return'], [1, 3, 6])
        rs = RollingSharp(perf_df['daily_return'], [1, 3, 6])
    else:
        benchmarkCumReturns = None
        accessReturns = None
        accessCumReturns = None

    if len(perf_df['daily_return']) > APPROX_BDAYS_PER_MONTH and benchmarkCumReturns is not None:
        rollingRisk = pd.concat([pd.concat(rs, axis=1), pd.concat(rb, axis=1)], axis=1)
    else:
        rollingRisk = None

    if plot:
        verticalSections = 2
        plt.figure(figsize=(16, 7 * verticalSections))
        gs = gridspec.GridSpec(verticalSections, 3, wspace=0.5, hspace=0.5)

        axRollingReturns = plt.subplot(gs[0, :])
        axDrawDown = plt.subplot(gs[1, :], sharex=axRollingReturns)

        plottingRollingReturn(perf_df['daily_cum_return'], benchmarkCumReturns, axRollingReturns)
        plottingDrawdownPeriods(perf_df['daily_cum_return'], drawDownDaily, 5, axDrawDown)

        if rollingRisk is not None:
            plt.figure(figsize=(16, 7 * verticalSections))
            gs = gridspec.GridSpec(verticalSections, 3, wspace=0.5, hspace=0.5)
            axRollingBeta = plt.subplot(gs[0, :])
            axRollingSharp = plt.subplot(gs[1, :])

            bmName = benchmarkReturns.name
            plottingRollingBeta(rb, bmName, ax=axRollingBeta)
            plottingRollingSharp(rs, ax=axRollingSharp)

        plt.figure(figsize=(16, 7 * verticalSections))
        gs = gridspec.GridSpec(verticalSections, 3, wspace=0.5, hspace=0.5)

        axUnderwater = plt.subplot(gs[0, :])
        axMonthlyHeatmap = plt.subplot(gs[1, 0])
        axAnnualReturns = plt.subplot(gs[1, 1])
        axMonthlyDist = plt.subplot(gs[1, 2])

        plottingUnderwater(drawDownDaily['draw_down'], axUnderwater)
        plottingMonthlyReturnsHeapmap(returns, axMonthlyHeatmap)
        plottingAnnualReturns(returns, axAnnualReturns)
        plottingMonthlyRetDist(returns, axMonthlyDist)

    if accessReturns is not None and plot:
         plt.figure(figsize=(16, 7 * verticalSections))
         gs = gridspec.GridSpec(verticalSections, 3, wspace=0.5, hspace=0.5)
         axRollingAccessReturns = plt.subplot(gs[0, :])
         axAccessDrawDown = plt.subplot(gs[1, :], sharex=axRollingAccessReturns)
         plottingRollingReturn(accessCumReturns, None, axRollingAccessReturns, title='Access Cumulative Returns w.r.t. ' + benchmarkReturns.name)
         plottingDrawdownPeriods(accessCumReturns, accessDrawDownDaily, 5, axAccessDrawDown, title=('Top 5 Drawdown periods w.r.t. ' + benchmarkReturns.name))

         plt.figure(figsize=(16, 7 * verticalSections))
         gs = gridspec.GridSpec(verticalSections, 3, wspace=0.5, hspace=0.5)

         axAccessUnderwater = plt.subplot(gs[0, :])
         axAccessMonthlyHeatmap = plt.subplot(gs[1, 0])
         axAccessAnnualReturns = plt.subplot(gs[1, 1])
         axAccessMonthlyDist = plt.subplot(gs[1, 2])

         plottingUnderwater(accessDrawDownDaily['draw_down'], axAccessUnderwater, title='Underwater Plot w.r.t. '
                                                                                        + benchmarkReturns.name)
         plottingMonthlyReturnsHeapmap(accessReturns, ax=axAccessMonthlyHeatmap, title='Monthly Access Returns (%)')
         plottingAnnualReturns(accessReturns, ax=axAccessAnnualReturns, title='Annual Access Returns')
         plottingMonthlyRetDist(accessReturns, ax=axAccessMonthlyDist, title='Distribution of Monthly Access Returns')

    return perf_metric, perf_df, rollingRisk


@plotting_context
def createPostionTearSheet(position, plot=True):
    positions = aggregatePositons(position)
    positions_weiget = calculatePosWeight(positions)
    if plot:
        verticalSections = 3
        plt.figure(figsize=(16, 7 * verticalSections))
        gs = gridspec.GridSpec(verticalSections, 3, wspace=0.5, hspace=0.5)

        axExposure = plt.subplot(gs[0, :])
        axTopExposure = plt.subplot(gs[1, :], sharex=axExposure)
        axHoldings = plt.subplot(gs[2, :])

        plottingExposure(positions_weiget, axExposure)
        plottingTopExposure(positions_weiget, axTopExposure)
        plottingHodings(positions_weiget, axHoldings)
    return positions


@plotting_context
def createTranscationTearSheet(transactions, positions, plot=True):
    positions = aggregatePositons(positions)
    transcations = aggregateTranscations(transactions)

    if plot:
        verticalSections = 1
        plt.figure(figsize=(16, 7 * verticalSections))
        gs = gridspec.GridSpec(verticalSections, 3, wspace=0.5, hspace=0.5)
        axTurnOver = plt.subplot(gs[0, :])
    else:
        axTurnOver = None

    turnOverRate = plottingTurnover(transcations, positions, axTurnOver)[1]
    turnOverRate.name = 'turnover_rate'
    turnOverRate.index.name = 'date'

    return pd.DataFrame(turnOverRate)


@plotting_context
def createAllTearSheet(positions, transcations, prices=None, returns=None, benchmark=None, plot=True):
    perf_metric, perf_df = createPerformanceTearSheet(prices=prices, returns=returns, benchmark=benchmark, plot=plot)
    createPostionTearSheet(position=positions, plot=plot)
    createTranscationTearSheet(position=positions, transcations=transcations, plot=plot)
    return perf_metric, perf_df
