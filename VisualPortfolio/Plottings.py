# -*- coding: utf-8 -*-
u"""
Created on 2015-11-9

@author: cheng.li
"""

import six
import random
from functools import wraps
import seaborn as sns
import matplotlib
from matplotlib.ticker import FuncFormatter
from matplotlib import colors
import pandas as pd
import numpy as np
from VisualPortfolio.Timeseries import aggregateReturns
from VisualPortfolio.Transactions import getTurnOver


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


def plottingHodings(positions, ax, title="Holdings per Day"):
    if 'cash' in positions:
        positions = positions.drop('cash', axis='columns')
    df_holdings = positions.apply(lambda x: np.sum(x != 0), axis='columns')
    df_holdings_by_month = df_holdings.resample('1M').mean()
    df_holdings.plot(color='steelblue', alpha=0.6, lw=0.5, ax=ax)
    df_holdings_by_month.plot(
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

    ax.legend(['Daily holdings',
               'average month daily holdings',
               'average whold peirod daily holdings'],
              loc="best")
    ax.set_title(title)
    ax.set_ylabel('Number of securities holdings')
    ax.set_xlabel('')
    return ax


def plottingTurnover(transactions, positions, turn_over=None, freq='M', ax=None, title="Turnover Analysis"):
    if turn_over is None:
        df_turnover = getTurnOver(transactions, positions)
    else:
        df_turnover = turn_over

    df_turnover_agreagted = df_turnover.resample(freq).sum().dropna()

    if freq == 'M':
        freq = 'Monthly'
    else:
        freq = 'Daily'

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
