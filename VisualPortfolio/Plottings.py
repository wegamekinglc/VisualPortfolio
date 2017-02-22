# -*- coding: utf-8 -*-
u"""
Created on 2015-11-9

@author: cheng.li
"""

from functools import wraps
import seaborn as sns
import matplotlib
from matplotlib.ticker import FuncFormatter
from statsmodels.tsa.stattools import acf
import pandas as pd
import numpy as np
from VisualPortfolio.Timeseries import aggregateReturns
from VisualPortfolio.Transactions import getTurnOver
from VisualPortfolio.Timeseries import aggregatePositons


def get_color_list():
    return ['#0000CD', '#F08080', '#8B0000', '#EE82EE', '#8B4513', '#008B8B',
            '#7B68EE', '#FF1493', '#FF6347', '#DC143C', '#E9967A', '#FF4500',
            '#DA70D6', '#FFA07A', '#8B008B', '#66CDAA', '#3CB371', '#191970',
            '#4B0082', '#0000FF', '#BC8F8F', '#FF8C00', '#FFB6C1', '#4682B4',
            '#FF00FF', '#DB7093', '#FF7F50', '#20B2AA', '#2E8B57', '#DAA520',
            '#FA8072', '#1E90FF', '#BA55D3', '#000000', '#87CEEB', '#5F9EA0',
            '#00BFFF', '#556B2F', '#CD853F', '#FFFF00', '#6495ED', '#483D8B',
            '#A52A2A', '#2F4F4F', '#B22222', '#C71585', '#FF0000', '#9932CC',
            '#00008B', '#00FFFF', '#FFA500', '#FFD700', '#D8BFD8', '#800080',
            '#00CED1', '#FF00FF', '#4169E1', '#9400D3', '#40E0D0', '#B8860B',
            '#808000', '#8FBC8F']


def plotting_context(func):
    @wraps(func)
    def call_w_context(*args, **kwargs):
        set_context = kwargs.pop('set_context', True)
        if set_context:
            with context():
                return func(*args, **kwargs)
        else:
            return func(*args, **kwargs)

    return call_w_context


def context(context='notebook', font_scale=1.5, rc=None):
    if rc is None:
        rc = {}

    rc_default = {'lines.linewidth': 1.5,
                  'axes.facecolor': '0.995',
                  'figure.facecolor': '0.97'}

    # Add defaults if they do not exist
    for name, val in rc_default.items():
        rc.setdefault(name, val)

    return sns.plotting_context(context=context, font_scale=font_scale,
                                rc=rc)


def integer_format(x, pos):
    return '%d' % x


def two_dec_places(x, pos):
    return '%.2f' % x


def percentage(x, pos):
    return '%.2f%%' % (x * 100)


def zero_dec_percentage(x, pos):
    return '%.1f%%' % (x * 100)


def plottingRollingReturn(cumReturns,
                          cumReturnsWithoutTC,
                          benchmarkReturns,
                          other_curves,
                          ax,
                          title='Strategy Cumulative Returns'):
    y_axis_formatter = FuncFormatter(two_dec_places)
    ax.yaxis.set_major_formatter(FuncFormatter(y_axis_formatter))

    cumReturns.plot(lw=3,
                    color='forestgreen',
                    alpha=0.6,
                    label='Strategy',
                    ax=ax)

    if cumReturnsWithoutTC is not None:
        cumReturnsWithoutTC.plot(lw=3,
                                 color='red',
                                 alpha=0.6,
                                 label='Strategy (w/o tc)',
                                 ax=ax)

    color_names = get_color_list()

    if benchmarkReturns is not None:
        benchmarkReturns.plot(lw=2,
                              color='gray',
                              alpha=0.6,
                              label=benchmarkReturns.name,
                              ax=ax)

    if other_curves is not None:
        for i, curve_info in enumerate(zip(*other_curves)):
            marker = curve_info[0]
            line_style = curve_info[1]
            label = curve_info[2]
            series = curve_info[3]
            series.plot(lw=2,
                        marker=marker,
                        markersize=12,
                        linestyle=line_style,
                        color=color_names[i],
                        alpha=0.6,
                        label=label,
                        ax=ax)

    ax.axhline(0.0, linestyle='--', color='black', lw=2)
    ax.set_ylabel('Cumulative returns')
    ax.set_title(title)
    ax.legend(loc='best')
    return ax


def plottingRollingBeta(rb, bmName, ax):
    y_axis_formatter = FuncFormatter(two_dec_places)
    ax.yaxis.set_major_formatter(FuncFormatter(y_axis_formatter))

    ax.set_title("Rolling Portfolio Beta to " + bmName)
    ax.set_ylabel('Beta')

    rb['beta_1m'].plot(color='steelblue', lw=3, alpha=0.6, ax=ax)
    rb['beta_3m'].plot(color='grey', lw=3, alpha=0.4, ax=ax)
    rb['beta_6m'].plot(color='yellow', lw=3, alpha=0.5, ax=ax)
    ax.axhline(rb['beta_1m'].mean(), color='steelblue', linestyle='--', lw=3)
    ax.axhline(0.0, color='black', linestyle='-', lw=2)
    ax.set_xlabel('')
    ax.legend(['1-m',
               '3-m',
               '6-m',
               'average 1-m'],
              loc='best')

    return ax


def plottingRollingSharp(rs, ax):
    y_axis_formatter = FuncFormatter(two_dec_places)
    ax.yaxis.set_major_formatter(FuncFormatter(y_axis_formatter))

    ax.set_title('Rolling Sharpe ratio')
    ax.set_ylabel('Sharp')

    rs['sharp_1m'].plot(color='steelblue', lw=3, alpha=0.6, ax=ax)
    rs['sharp_3m'].plot(color='grey', lw=3, alpha=0.4, ax=ax)
    rs['sharp_6m'].plot(color='yellow', lw=3, alpha=0.5, ax=ax)
    ax.axhline(rs['sharp_1m'].mean(), color='steelblue', linestyle='--', lw=3)
    ax.axhline(0.0, color='black', linestyle='-', lw=2)
    ax.set_xlabel('')
    ax.legend(['1-m',
               '3-m',
               '6-m',
               'average 1-m'],
              loc='best')
    return ax


def plottingDrawdownPeriods(cumReturns,
                            drawDownTable,
                            top,
                            ax,
                            title='Top 5 Drawdown Periods'):
    y_axis_formatter = FuncFormatter(two_dec_places)
    ax.yaxis.set_major_formatter(FuncFormatter(y_axis_formatter))
    cumReturns.plot(ax=ax)
    lim = ax.get_ylim()

    tmp = drawDownTable.sort_values(by='draw_down')
    topDrawdown = tmp.groupby('recovery').first()
    topDrawdown = topDrawdown.sort_values(by='draw_down')[:top]
    colors = sns.cubehelix_palette(len(topDrawdown))[::-1]
    for i in range(len(colors)):
        recovery = topDrawdown.index[i]
        ax.fill_between((topDrawdown['peak'][i], recovery),
                        lim[0],
                        lim[1],
                        alpha=.4,
                        color=colors[i])

    ax.set_title(title)
    ax.set_ylabel('Cumulative returns')
    ax.legend(['Cumulative returns'], loc='best')
    ax.set_xlabel('')
    return ax


def plottingUnderwater(drawDownSeries, ax, title='Underwater Plot'):
    y_axis_formatter = FuncFormatter(percentage)
    ax.yaxis.set_major_formatter(FuncFormatter(y_axis_formatter))
    drawDownSeries.plot(ax=ax, kind='area', color='coral', alpha=0.7)
    ax.set_ylabel('Drawdown')
    ax.set_title(title)
    ax.legend(loc='best')
    ax.set_xlabel('')
    return ax


def plottingMonthlyReturnsHeapmap(returns, ax, title='Monthly Returns (%)'):
    x_axis_formatter = FuncFormatter(integer_format)
    ax.xaxis.set_major_formatter(FuncFormatter(x_axis_formatter))
    monthlyRetTable = pd.DataFrame(aggregateReturns(returns, convert='monthly')[0])
    monthlyRetTable = monthlyRetTable.unstack()
    monthlyRetTable.columns = monthlyRetTable.columns.droplevel()
    sns.heatmap((np.exp(monthlyRetTable.fillna(0)) - 1.0) * 100.0,
                annot=True,
                fmt=".1f",
                annot_kws={"size": 9},
                alpha=1.0,
                center=0.0,
                cbar=False,
                cmap=matplotlib.cm.RdYlGn_r,
                ax=ax)
    ax.set_ylabel('Year')
    ax.set_xlabel('Month')
    ax.set_title(title)
    return ax


def plottingAnnualReturns(returns, ax, title='Annual Returns'):
    x_axis_formatter = FuncFormatter(zero_dec_percentage)
    ax.xaxis.set_major_formatter(FuncFormatter(x_axis_formatter))
    ax.tick_params(axis='x', which='major', labelsize=10)

    annulaReturns = pd.DataFrame(aggregateReturns(returns, convert='yearly')[0])
    annulaReturns = np.exp(annulaReturns) - 1.

    ax.axvline(annulaReturns.values.mean(),
               color='steelblue',
               linestyle='--',
               lw=4,
               alpha=0.7)

    annulaReturns.sort_index(ascending=False).plot(
        ax=ax,
        kind='barh',
        alpha=0.7
    )

    ax.axvline(0.0, color='black', linestyle='-', lw=3)

    ax.set_ylabel('Year')
    ax.set_xlabel('Returns')
    ax.set_title(title)
    ax.legend(['mean'], loc='best')
    return ax


def plottingMonthlyRetDist(returns,
                           ax,
                           title="Distribution of Monthly Returns"):
    x_axis_formatter = FuncFormatter(zero_dec_percentage)
    ax.xaxis.set_major_formatter(FuncFormatter(x_axis_formatter))
    ax.tick_params(axis='x', which='major', labelsize=10)

    monthlyRetTable = aggregateReturns(returns, convert='monthly')[0]
    monthlyRetTable = np.exp(monthlyRetTable) - 1.

    if len(monthlyRetTable) > 1:
        ax.hist(
            monthlyRetTable,
            color='orange',
            alpha=0.8,
            bins=20
        )

    ax.axvline(
        monthlyRetTable.mean(),
        color='steelblue',
        linestyle='--',
        lw=4,
        alpha=1.0
    )

    ax.axvline(0.0, color='black', linestyle='-', lw=3, alpha=0.75)
    ax.legend(['mean'], loc='best')
    ax.set_ylabel('Number of months')
    ax.set_xlabel('Returns')
    ax.set_title(title)
    return ax


def plottingExposure(positions, ax, title="Total non cash exposure (%)"):
    positions = aggregatePositons(positions, convert='daily')
    y_axis_formatter = FuncFormatter(two_dec_places)
    ax.yaxis.set_major_formatter(FuncFormatter(y_axis_formatter))
    if 'cash' in positions:
        positions_without_cash = positions.drop('cash', axis='columns')
    else:
        positions_without_cash = positions
    longs = positions_without_cash[positions_without_cash > 0] \
                .sum(axis=1).fillna(0) * 100
    shorts = positions_without_cash[positions_without_cash < 0] \
                 .abs().sum(axis=1).fillna(0) * 100
    df_long_short = pd.DataFrame({'long': longs,
                                  'short': shorts})
    df_long_short.plot(kind='area',
                       stacked=True,
                       color=['blue', 'green'],
                       linewidth=0., ax=ax)
    ax.set_title(title)
    return ax


def plottingTopExposure(positions,
                        ax,
                        top=10,
                        title="Top 10 securities exposure (%)"):
    positions = aggregatePositons(positions, convert='daily')
    y_axis_formatter = FuncFormatter(two_dec_places)
    ax.yaxis.set_major_formatter(FuncFormatter(y_axis_formatter))
    df_mean = positions.abs().mean()
    df_top = df_mean.nlargest(top)
    (positions[df_top.index] * 100.).plot(ax=ax)
    ax.legend(loc='upper center',
              frameon=True,
              bbox_to_anchor=(0.5, -0.14),
              ncol=5)
    ax.set_title(title)
    return ax


def plottingHodings(positions, ax, freq='M', title="Holdings Analysis"):
    positions = aggregatePositons(positions, convert='daily')
    if 'cash' in positions:
        positions = positions.drop('cash', axis='columns')
    df_holdings = positions.apply(lambda x: np.sum(x != 0), axis='columns')
    df_holdings_by_freq = df_holdings.resample(freq).mean()
    df_holdings.plot(color='steelblue', alpha=0.6, lw=0.5, ax=ax)

    if freq == 'M':
        freq = 'monthly'
    else:
        freq = 'daily'

    df_holdings_by_freq.plot(
        color='orangered',
        alpha=0.5,
        lw=2,
        ax=ax)
    ax.axhline(
        df_holdings.values.mean(),
        color='steelblue',
        ls='--',
        lw=3,
        alpha=1.0)

    ax.set_xlim((positions.index[0], positions.index[-1]))

    ax.legend(['Holdings on each bar',
               'Average {0} holdings'.format(freq),
               'Average whole peirod {0} holdings'.format(freq)],
              loc="best")
    ax.set_title(title)
    ax.set_ylabel('Number of securities holdings')
    ax.set_xlabel('')
    return ax


def plottingPositionACF(positions, ax, title='Position auto correlation function'):
    positions = aggregatePositons(positions, convert='raw')
    if 'cash' in positions:
        positions = positions.drop('cash', axis='columns')

    nlags = 100
    acf_mat = np.zeros((len(positions.columns), nlags+1))
    cols = positions.columns

    for i, col in enumerate(cols):
        acfs = acf(positions[col], nlags=nlags)
        acf_mat[i, 0:len(acfs)] = acfs

    acf_mean = np.mean(acf_mat, axis=0)
    ax.plot(acf_mean,
            color='orangered',
            alpha=0.5,
            lw=2,)
    ax.set_title(title)
    ax.set_ylabel('Auto correlation')
    ax.set_xlabel('lags')
    return ax


def plottingTurnover(transactions, positions, turn_over=None, freq='M', ax=None, title="Turnover Analysis"):
    if turn_over is None:
        df_turnover = getTurnOver(transactions, positions)
    else:
        df_turnover = turn_over

    df_turnover_agreagted = df_turnover.resample(freq).sum().dropna()

    if freq == 'M':
        freq = 'monthly'
    else:
        freq = 'daily'

    if ax:
        y_axis_formatter = FuncFormatter(two_dec_places)
        ax.yaxis.set_major_formatter(FuncFormatter(y_axis_formatter))
        df_turnover.plot(color='steelblue', alpha=1.0, lw=0.5, ax=ax)
        df_turnover_agreagted.plot(
            color='orangered',
            alpha=0.5,
            lw=2,
            ax=ax)
        ax.axhline(
            df_turnover_agreagted.mean(),
            color='steelblue',
            linestyle='--',
            lw=3,
            alpha=1.0)
        ax.legend(['turnover',
                   'Aggregated {0} turnover'.format(freq),
                   'Average {0} turnover'.format(freq)],
                  loc="best")
        ax.set_title(title + ' (aggregated {0})'.format(freq))
        ax.set_ylabel('Turnover')
    return ax, df_turnover


if __name__ == "__main__":

    from matplotlib import pyplot as plt
    from FactorMiner.runner.simplebarrunner import SimpleBarRunner
    from PyFin.api import *

    alpha_list = ['alpha101_40_4',
                  'alpha101_225_10',
                  'alpha101_540_20',
                  'alpha102_40_4',
                  'alpha111_40_10',
                  'alpha111_225_10',
                  'alpha111_600_20',
                  'alpha121_40_15',
                  'alpha121_275_15',
                  'alpha121_600_20',
                  'alpha121_900_20',
                  'alpha122_140_4',
                  'alpha151_40_15',
                  'alpha151_275_15',
                  'alpha151_600_20',
                  'alpha151_900_20',
                  'alpha152_140_5']
    weights_list = [1.,
                    3.,
                    6.,
                    12.,
                    1.,
                    3.,
                    6.,
                    1., 3., 6., 3.,
                    12.,
                    1., 3., 6., 3., 12.]

    weights_list = np.array(weights_list) / np.sum(np.array(weights_list))

    huty_factor = None
    for i, f_name in enumerate(alpha_list):
        if huty_factor:
            huty_factor = huty_factor + weights_list[i] * LAST(f_name)
        else:
            huty_factor = weights_list[i] * LAST(f_name)

    runner = SimpleBarRunner(None,
                             huty_factor,
                             '2014-01-01',
                             '2017-02-01',
                             username='sa',
                             password='A12345678!',
                             server_name='test_w',
                             account='test_mssql_sa',
                             freq=5)

    turn_over, daily_return, positions, risk_stats, detail_series, factor_values \
        = runner.simulate(leverage=None, tc_cost=0.)

    plt.show()