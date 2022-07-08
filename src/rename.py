#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 27 17:04:16 2022

@author: aouyed
"""

import os
import glob
from tqdm import tqdm

files=glob.glob('../data/interim/dataframes/coarse_*')

for file in tqdm(files):
    dirstring=os.path.dirname(os.path.realpath(file))
    base=os.path.basename(file) 
    new_file=os.path.join(dirstring, 'tvl1_'+ base)
    old_file=os.path.join(dirstring, base)
    os.rename(old_file, new_file)