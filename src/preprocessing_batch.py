#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 11 16:56:44 2022

@author: aouyed
"""

import wind_collocator as wc
import igra_collocator as ic
import vertical_coarsening as vc
from parameters import parameters
from datetime import datetime 

param= parameters()
param.set_alg('tvl1')
param.set_plev_coarse(5)
param.set_Lambda(0.3)
for thresh in [10, 100]:
    param.set_thresh(thresh)
    for month in [1]:
        param.set_month(datetime(2020,month,1))
        vc.main(param)
        #ic.main(param)
        #wc.main(param)
