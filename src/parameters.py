#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 17 11:14:22 2022

@author: aouyed
"""

from datetime import datetime 

class parameters:
  
    
    def __init__(self):
        self.tag=None
        self.plev_coarse=5
        self.thresh=10
        self.month=datetime(2020,1,1)
        self.month_string=self.month.strftime("%B").lower()
        self.geodesics={'swath':[(-35.4, 28.75), (49.74, 66.13),'latitude'],
           'equator':[(-36.5, -127.7),(55.5, -95.5),'latitude']}
        self.pressures=[850, 700, 500, 400]
        self.set_tag(self.plev_coarse, self.thresh, self.month_string)
        


        
    def set_plev_coarse(self, n_layers):
        self.plev_coarse=n_layers
        self.set_tag(self.plev_coarse, self.thresh, self.month_string)

    
    def set_thresh(self, thresh):
        self.thresh=thresh
        self.set_tag(self.plev_coarse, self.thresh, self.month_string)
        
    def set_tag(self, plev_coarse, thresh, month_string):
        self.tag='coarse_' + month_string + '_' + str(plev_coarse)+'_t'+str(thresh)
        
    def set_month(self, date):
        self.month=date
        self.month_string=self.month.strftime("%B").lower()
        self.set_tag(self.plev_coarse, self.thresh, self.month_string)
       
    
        