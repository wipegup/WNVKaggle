##########
# This file is meant to be a collection of preprocessing functions
# For the west-nile data
##########

import shapefile
from scipy.spatial import cKDTree
import numpy as np
import pandas as pd
from os import listdir
from os.path import isfile, join
from sklearn.decomposition import TruncatedSVD


###########
# Calculations and Functions for water/park Data
##########

# Park and water shapefile locations
parkDir = './AddData/Parks/'
waterDir = './AddData/Water/'

#######
# Builds, then returns a dictionary of parksize, parkFinder, and waterFinder
#######
def buildWaterAndParkDicts(parkDir, waterDir):

    # Park and water shapefile names
    parks = [f for f in listdir(parkDir) if isfile(join(parkDir,f)) if f.count('.csv') ==0]
    water = [f for f in listdir(waterDir) if isfile(join(waterDir,f))if f.count('.csv') ==0]

    parkShape = parkDir + parks[0].split('.')[0]
    waterShape = waterDir + water[0].split('.')[0]

    # Read in shapefiles, then the shapes/records
    psf = shapefile.Reader(parkShape)
    wsf = shapefile.Reader(waterShape)

    parkSR = psf.shapeRecords()
    waterSR = wsf.shapeRecords()

    # Create a parksize dictionary, keyed on parkname
    parkSize = {}
    for s in parkSR:
        parkSize[s.record[4]] = s.record[19]

    # Create cKDTree functions in Dict
    # Key: Identifier (number for water, park name for park)
    # Value: cKDTree function built on all the points associated with water/park feature

    waterFinder={}
    for i, s in enumerate(waterSR):
        waterFinder[i] = cKDTree(s.shape.points)

    parkFinder = {}
    for s in parkSR:
        parkFinder[s.record[4]] = cKDTree(s.shape.points)

    return parkSize, parkFinder, waterFinder

######
# Function returns both a TrunatedSVD object and a transformed DataFrame
# Which describes the relationship between locations and parks
######

def yeildParkSVD(parkSize, parkFinder, uniqueLocs, TruncSVD = 'calc', comps = 4):

    parkDist = {}

    for l in uniqueLocs:
        parkDist[l] = {}
        for k in parkFinder:
            parkDist[l][k] = (parkFinder[k].query(l,1)[0], parkSize[k])

    parkDF = pd.DataFrame()
    parkDF = parkDF.from_dict(parkDist)
    parkDF = parkDF.transpose()
    parkDF.index = [idx for idx in parkDF.index]

    for c in parkDF:
        parkDF[c+' Area'] = [e[1] for e in parkDF[c]]
        parkDF[c] = [e[0] for e in parkDF[c]]

    if TruncSVD == 'calc':
        TruncSVD = TruncatedSVD(n_components = comps)
        TruncSVD.fit(parkDF)

    toRet = TruncSVD.transform(parkDF)

    toRet = pd.DataFrame(toRet, index = parkDF.index)
    return toRet, TruncSVD

######
# Function returns both a TrunatedSVD object and a transformed DataFrame
# Which describes the relationship between locations and water
######

def yeildWaterSVD(waterFinder, uniqueLocs, TruncSVD = 'calc', comps = 4):

    waterDist = {}

    for l in uniqueLocs:
        waterDist[l] = {}
        for k in waterFinder:
            waterDist[l][k] = waterFinder[k].query(l,1)[0]

    waterDF = pd.DataFrame()
    waterDF = waterDF.from_dict(waterDist)
    waterDF = waterDF.transpose()
    waterDF.index = [idx for idx in waterDF.index]

    if TruncSVD == 'calc':
        TruncSVD = TruncatedSVD(n_components = comps)
        TruncSVD.fit(waterDF)

    toRet = TruncSVD.transform(waterDF)

    toRet = pd.DataFrame(toRet, index = waterDF.index)
    return toRet, TruncSVD
