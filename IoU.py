#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 19 13:01:52 2023

@author: willalbert
"""

import numpy as np

def IoU_from_confusions(confusions):
    """
    Computes IoU from confusion matrices.
    :param confusions: ([..., n_c, n_c] np.int32). Can be any dimension, the confusion matrices should be described by
    the last axes. n_c = number of classes
    :param ignore_unclassified: (bool). True if the the first class should be ignored in the results
    :return: ([..., n_c] np.float32) IoU score
    """

    # Compute TP, FP, FN. This assume that the second to last axis counts the truths (like the first axis of a
    # confusion matrix), and that the last axis counts the predictions (like the second axis of a confusion matrix)
    TP = np.diagonal(confusions, axis1=-2, axis2=-1)
    TP_plus_FN = np.sum(confusions, axis=-1)
    TP_plus_FP = np.sum(confusions, axis=-2)

    # Compute IoU
    IoU = TP / (TP_plus_FP + TP_plus_FN - TP + 1e-6)

    # Compute mIoU with only the actual classes
    mask = TP_plus_FN < 1e-3
    counts = np.sum(1 - mask, axis=-1, keepdims=True)
    mIoU = np.sum(IoU, axis=-1, keepdims=True) / (counts + 1e-6)

    # If class is absent, place mIoU in place of 0 IoU to get the actual mean later
    IoU += mask * mIoU

    return IoU




# matConfClass = np.array([[752413,479487,589804,266284,0,94368,21054,7626,13,401,0,0,171623,381868,0,6367,18,0,0,0,0,0,0,4563,2745,0,0,0,0],[611630,39291217,892674,197281,18,48327,10930,797,0,20,0,0,83559,420542,0,72841,866,0,0,5,0,0,0,167,9,0,0,0,0],[265278,1644345,9950201,125001,22,33897,54303,7557,4,471,0,0,343145,453600,0,20503,592,0,0,0,0,0,0,807,3209,115,0,0,0],[661700,713957,687348,870838,0,38109,6121,2023,4,17,0,0,70571,213747,0,28430,7,0,0,0,0,0,0,5609,963,0,0,0,0],[4,931,411,12,0,0,0,0,0,0,0,0,0,1340,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0],[36676,731,47635,12647,0,18365288,60331,5431,0,1569,0,3,1260247,38891,0,2777,1577,0,0,0,0,0,0,139137,1662,0,0,0,0],[20431,11131,152669,5992,0,307534,1095205,74515,0,1232,0,0,3399228,62854,0,8103,0,0,0,1,0,0,0,34986,23571,0,0,0,0],[3881,1612,30355,1801,0,74370,75984,382177,91,586,0,77,1038770,46152,0,2338,24,0,0,0,0,0,0,14828,15459,0,0,0,0],[13,11014,16071,54,0,0,1345,162037,3191,38,0,0,169457,49530,0,231,0,0,0,215,0,0,0,0,12,0,0,0,0],[118,92,4207,6,0,12180,3043,2537,0,147716,0,4167,104423,3152,0,1311,86,0,0,0,0,0,0,21,117,476,0,153,0],[0,0,0,0,0,8,0,0,0,703,0,378,21,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],[231,84,396,27,0,2915,412,2059,1,18283,0,19916,36830,14,0,1118,311,0,0,0,0,0,0,321,37,0,0,110,0],[49491,108638,192824,5847,0,689656,125868,47385,144,17083,0,773,50767424,1032690,0,34154,5630,0,0,0,0,0,0,26006,4363,0,0,0,0],[140363,517295,434247,25156,0,99838,25700,21325,50,589,0,0,1356062,7785675,0,2721,1,0,0,0,0,0,0,3284,465,0,0,0,0],[3,240,514,27,0,67,232,281,0,194,0,1928,14867,214,0,2340,531,0,0,1445,0,0,0,122,45,0,0,0,0],[950,113653,14139,13480,0,7177,387,871,0,31,0,0,172836,2595,0,5337742,2199,0,0,522,0,0,0,1111,11,0,0,0,0],[0,3346,177,50,0,0,0,212,0,0,0,0,56280,101,0,63625,236637,0,0,187,0,0,0,0,0,0,0,0,0],[0,24,2,0,0,0,0,0,0,0,0,0,0,0,0,576,15248,0,0,0,0,0,0,0,0,0,0,0,0],[0,155,123,8,0,0,0,216,0,0,0,0,128,6,0,9386,17823,0,0,0,0,0,0,21,0,0,0,0,0],[357,4563,1946,780,0,3404,1412,8407,47,24,0,38,63572,786,0,94233,12609,0,0,3173,0,0,0,8,0,0,0,0,0],[121,743,55,129,0,0,44,0,0,0,0,0,3141,37,0,133,7172,0,0,0,0,0,0,99,0,0,0,0,0],[3,598,668,18,0,69,14,1723,311,83,0,360,19521,254,0,1900,3631,0,0,1493,0,0,0,48,21,0,0,0,0],[46,123,223,2,0,1798,90,1158,14,448,0,337,19603,938,0,333,0,0,0,0,0,0,0,0,0,0,0,0,0],[8671,113,11347,14290,0,921046,80812,6124,0,76,0,0,779519,8161,0,1862,2805,0,0,0,0,0,0,826207,5490,148,0,0,0],[6618,166,11379,1880,0,34794,44014,38613,0,60,0,0,230226,4408,0,464,0,0,0,0,0,0,0,7429,106467,0,0,0,0],[0,0,9,0,0,47,0,0,0,0,0,0,0,0,0,0,1567,0,0,0,0,0,0,72,0,7617,0,0,0],[305,418,4253,67,0,1009,224,5852,1680,5728,0,1171,21258,8100,0,1621,1,0,0,94,0,0,0,5,93,0,0,8,0],[0,0,0,0,0,899,0,0,0,6448,0,4005,8045,0,0,0,0,0,0,0,0,0,0,0,0,0,0,2830,0],[149,0,34,0,0,2863,129,0,0,0,0,0,9759,458,0,0,0,0,0,0,0,0,0,745,0,0,0,0,0]], np.int32)
# iouClass = IoU_from_confusions(matConfClass)
# lblsClass = ['ground','road','sidewalk','parking','rail track','building','wall','fence','guard rail','pole','traffic light','traffic sign','vegetation','terrain','person','car','truck','bus','caravan','trailer','train','motorcycle','bicycle','garage','gate','stop','smallpole','lamp','unknown_constr']


# matConfCateg = np.array([[58000856,225847,117358,1024,668898,1471097,129630,0],[132293,20255286,159979,1796,2049525,47510,9021,0],[275067,473941,2022681,1993,4837681,162944,11376,0],[10997,17666,16613,221831,185444,11480,10425,0],[356800,715662,177760,17856,50767424,1032690,39784,0],[1117061,103122,47540,589,1356062,7785675,2722,0],[154801,11820,11607,93,295957,3525,5801265,0],[1681,1915,3331,1228,39124,1192,7357,0]])
# iouCateg = IoU_from_confusions(matConfCateg)
# lblsCateg = ['Flat','Building','Gate','Pole','Veget','Terrain','Vehicle','Moto/Bike']


# # CLASSES
# print("\n\n{:#>16}{:#<6}".format(" CLASSES ",""))
# print("{:#>14}{:#<8}".format(" IoU ",""))
# for i, elem in enumerate(lblsClass):
#     print("{:>14}: {:.4f}".format(elem, iouClass[i]))
    
# iouClassSansZero = []
# nomClassSansZero = []
# for i, elem in enumerate(lblsClass):
#     if iouClass[i] > 0.0001:
#         iouClassSansZero.append(iouClass[i])
#         nomClassSansZero.append(elem)

# iouClassSansZero = np.array(iouClassSansZero)

# print("\n{:#>14}{:#<8}".format(" mIoU ",""))
# print("{}: {:.4f}".format(" Class 0 inl.", np.mean(iouClass)))
# print("{}: {:.4f}".format(" Class 0 exl.", np.mean(iouClassSansZero)))

# # CATÉGORIES
# print("\n\n{:#>17}{:#<5}".format(" CATÉGORIES ",""))
# print("{:#>14}{:#<8}".format(" IoU ",""))
# for i, elem in enumerate(lblsCateg):
#     print("{:>14}: {:.4f}".format(elem, iouCateg[i]))
    

# print("\n{:#>14}{:#<8}".format(" mIoU ",""))
# print("{}: {:.4f}".format("Moto/Bike inl.", np.mean(iouCateg)))
# print("{}: {:.4f}".format("Moto/Bike exl.", np.mean(iouCateg[:-1])))


# ####### CLASSES ######
# ######### IoU ########
#         ground: 0.1641
#           road: 0.8684
#       sidewalk: 0.6220
#        parking: 0.2193
#     rail track: 0.0000
#       building: 0.8218
#           wall: 0.1918
#          fence: 0.1833
#     guard rail: 0.0077
#           pole: 0.4372
#  traffic light: 0.0000
#   traffic sign: 0.2068
#     vegetation: 0.8118
#        terrain: 0.5924
#         person: 0.0000
#            car: 0.8859
#          truck: 0.5461
#            bus: 0.0000
#        caravan: 0.0000
#        trailer: 0.0159
#          train: 0.0000
#     motorcycle: 0.0000
#        bicycle: 0.0000
#         garage: 0.2843
#           gate: 0.1954
#           stop: 0.7578
#      smallpole: 0.0000
#           lamp: 0.1258
# unknown_constr: 0.0000

# ######## mIoU ########
#  Class 0 inl.: 0.2737
#  Class 0 exl.: 0.4178


# ##### CATÉGORIES #####
# ######### IoU ########
#           Flat: 0.9256
#       Building: 0.8368
#           Gate: 0.2431
#           Pole: 0.4445
#          Veget: 0.8118
#        Terrain: 0.5924
#        Vehicle: 0.8940
#      Moto/Bike: 0.0000

# ######## mIoU ########
# Moto/Bike inl.: 0.5935
# Moto/Bike exl.: 0.6783


###################################################################
###################################################################
###################################################################
###################################################################




matConfCateg = np.array([[58726,5433,8486,33],[8741,338074,12,279],[28356,5501,17401,1],[1851,29,0,7882]])
iouCateg = IoU_from_confusions(matConfCateg)
lblsCateg = ['Plat','Constr','Nature','Vehicule']


print("\n\n{:#>17}{:#<5}".format(" CATÉGORIES ",""))
print("{:#>14}{:#<8}".format(" IoU ",""))
for i, elem in enumerate(lblsCateg):
    print("{:>14}: {:.4f}".format(elem, iouCateg[i]))
    

print("\n{:#>14}{:#<8}".format(" mIoU ",""))
print("{:14.4f}".format(np.mean(iouCateg)))

##### CATÉGORIES #####
######### IoU ########
#         Plat: 0.5261
#       Constr: 0.9442
#       Nature: 0.2912
#     Vehicule: 0.7823

######## mIoU ########
#               0.6359