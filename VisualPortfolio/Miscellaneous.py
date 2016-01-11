# -*- coding: utf-8 -*-

u"""
Created on 2016-1-8

@author: cheng.li
"""

import pandas as pd
from DataAPI import api
from VisualPortfolio.Plottings import plotting_context
from VisualPortfolio.Tears import createPerformanceTearSheet


@plotting_context
def portfolioAnalysis(posDF,
                      startDate,
                      endDate,
                      notional=10000000.,
                      benchmark='000300.zicn',
                      isweight=False):

    secIDs = posDF['instrumentID']

    data = api.GetEquityBarEOD(instrumentIDList=secIDs,
                               startDate=startDate,
                               endDate=endDate,
                               field='closePrice',
                               instrumentIDasCol=True,
                               baseDate='end')

    close_data = data['closePrice']
    close_data = close_data.fillna(method='pad')
    close_data.fillna(value=0., inplace=True)
    columns = close_data.columns

    for instrument in columns:

        if isweight and notional:
            invest_value = posDF[posDF.instrumentID == instrument]['position'].iloc[0] * notional
            volume = int(invest_value / close_data[instrument].values[0])
        else:
            volume = posDF[posDF.instrumentID == instrument]['position'].iloc[0]

        close_data[instrument] *= volume

    prices = close_data.sum(axis=1)

    perf_metric, perf_df, rollingRisk = createPerformanceTearSheet(prices=prices, benchmark=benchmark)
    return perf_metric, perf_df, rollingRisk


if __name__ == "__main__":

    from matplotlib import pyplot as plt

    df = pd.DataFrame(data={'instrumentID': ['600000', '000001'],
                            'position': [0.5, 0.5]})

    portfolioAnalysis(df, startDate='2015-01-01', endDate='2016-01-01', isweight=True)

    plt.show()


