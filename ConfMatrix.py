#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 16 15:12:07 2023

@author: willalbert
"""

from sklearn.metrics import confusion_matrix
import OSToolBox as ost
import os
import numpy as np


trueFile = "/home/willalbert/Documents/kpConv_Nerf/classesSCALED.ply"
pointCloudTrue = ost.read_ply(trueFile)
true = pointCloudTrue['scalar_Classe']

folder = "/home/willalbert/Documents/GitHub/KPConvPyTorch/test/Log_2023-05-15_19-03-10_CATEG69/predictions/"
dir_list = os.listdir(folder)

conf = []

for i, plyFile in enumerate(dir_list):
    if plyFile[-4:] == ".ply":
        pointCloud = ost.read_ply(folder + plyFile)
        pred = pointCloud['scalar_pre']
        
        conf.append(confusion_matrix(true, pred))
    
with open("/home/willalbert/Documents/GitHub/KPConvPyTorch/test/Log_2023-05-15_19-03-10_CATEG69/predictions/confMat.txt", 'w') as txtFile:
    txtFile.writelines('\n\n'.join(str(i) for i in conf))
    txtFile.close()