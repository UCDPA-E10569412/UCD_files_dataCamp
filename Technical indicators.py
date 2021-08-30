# -*- coding: utf-8 -*-
"""
Created on Sun Aug 29 10:33:50 2021

@author: micha
"""

import pandas_ta
# Add EMA to dataframe by appending
# Note: pandas_ta integrates seamlessly into
# our existing dataframe
df.ta.ema(close='adj_close', length=10, append=True)