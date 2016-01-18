# -*- coding: utf-8 -*-

u"""
Created on 2016-1-8

@author: cheng.li
"""

import pandas as pd
from DataAPI import api
from VisualPortfolio.Plottings import plotting_context
from VisualPortfolio.Tears import createPerformanceTearSheet
from VisualPortfolio.Env import DataSource
from VisualPortfolio.Env import Settings


def get_equity_eod(instruments, start_date, end_date):
    if Settings.data_source == DataSource.DXDataCenter:
        data = api.GetEquityBarEOD(instrumentIDList=instruments,
                                   startDate=start_date,
                                   endDate=end_date,
                                   field='closePrice',
                                   instrumentIDasCol=True,
                                   baseDate='end')
    elif Settings.data_source == DataSource.DataYes:
        import os
        import tushare as ts

        try:
            ts.set_token(os.environ['DATAYES_TOKEN'])
        except KeyError:
            raise

        mt = ts.Market()
        res = []
        for ins in instruments:
            data = mt.MktEqud(ticker=ins,
                              beginDate=start_date.replace('-', ''),
                              endDate=end_date.replace('-', ''),
                              field='tradeDate,ticker,closePrice')
            res.append(data)

        data = pd.concat(res)
        data['tradeDate'] = pd.to_datetime(data['tradeDate'], format='%Y-%m-%d')
        data['ticker'] = data['ticker'].apply(lambda x: '{0:06d}'.format(x))
        data.set_index(['tradeDate', 'ticker'], inplace=True, verify_integrity=True)
        data = data.unstack(level=-1)

    return data


@plotting_context
def portfolioAnalysis(posDF,
                      startDate,
                      endDate,
                      notional=10000000.,
                      benchmark='000300.zicn',
                      isweight=False):

    secIDs = posDF['instrumentID']

    data = get_equity_eod(instruments=secIDs,
                          start_date=startDate,
                          end_date=endDate)

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
    import pandas as pd
    data = pd.read_excel('d:/basket.xlsx')
    data.instrumentID = data.instrumentID.apply(lambda x: "{0:06d}".format(x))

    Settings.set_source(DataSource.DataYes)
    res = portfolioAnalysis(data, '2006-01-01', '2016-01-15')