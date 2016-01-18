# -*- coding: utf-8 -*-
u"""
Created on 2016-1-18

@author: cheng.li
"""

from enum import IntEnum
from enum import unique


@unique
class DataSource(IntEnum):
    DataYes = 1
    DXDataCenter = 2


class SettingsFactory:

    def __init__(self):
        self._data_source = DataSource.DataYes

    def set_source(self, data_source):
        self._data_source = data_source

    @property
    def data_source(self):
        return self._data_source


Settings = SettingsFactory()
