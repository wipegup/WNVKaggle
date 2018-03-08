import shapefile
from scipy.spatial import cKDTree
import numpy as np
import pandas as pd
from os import listdir
from os.path import isfile, join
from sklearn.decomposition import TruncatedSVD
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import FunctionTransformer
from sklearn.linear_model import LinearRegression

#class ChicagoPreprocessor(object):
#    def __init__(train, test = None):
#        self.parkFinder =


def agg_on_species(train, test):

    noAgg = [c for c in train.columns if c not in ['NumMosquitos','WnvPresent']]

    agg = train.groupby(noAgg)['NumMosquitos', 'WnvPresent'].sum()

    for i, c in enumerate(noAgg):
        agg[c] = agg.index.map(lambda x:x[i])

    agg.index = range(0,len(agg))
    agg['WnvPresent'] = (agg['WnvPresent'].map(lambda x:x>0)).astype(int)
    return agg, test

def InitPrepross(train, test):

    def location_add(df):
        df['Location'] = [(df.loc[idx,'Longitude'], df.loc[idx, 'Latitude'])
                            for idx in df.index]
        return df

    def change_date(df):
        df['Date'] = pd.to_datetime(df['Date'])

        return df

    def drop_unused(df):
        for col in ['Address','Block','Street',
              'AddressNumberAndStreet', 'AddressAccuracy',
                    ]:
            try:
                df = df.drop(col, axis = 'columns')
            except:
                print(col, 'not present')

        return df

    def species_dummies(df):
        species = ['CULEX PIPIENS', 'CULEX PIPIENS/RESTUANS',
                'CULEX RESTUANS', 'CULEX SALINARIUS',
                'CULEX TERRITANS', 'CULEX TARSALIS',
                 'CULEX ERRATICUS']
        for s in species:
            df[s] = (df['Species'] == s).astype(int)

        return df

    def transform(df):
        df = drop_unused(df)
        df = location_add(df)
        df = change_date(df)
        df = species_dummies(df)
        return df

    return transform(train), transform(test)

def LocationProcess(train, test):
    parkDir = './AddData/Parks/'
    waterDir = './AddData/Water/'

    def buildWaterFinder():
        water = [f for f in listdir(waterDir)
                if isfile(join(waterDir,f))
                if f.count('.csv') ==0]

        waterShape = waterDir + water[0].split('.')[0]
        waterSR = shapefile.Reader(waterShape).shapeRecords()

        waterFinder = {}
        for i, s in enumerate(waterSR):
            waterFinder[i] = cKDTree(s.shape.points)

        return waterFinder

    def buildParkDicts():
        parks = [f for f in listdir(parkDir)
                if isfile(join(parkDir,f))
                if f.count('.csv') ==0]
        parkShape = parkDir + parks[0].split('.')[0]
        parkSR = shapefile.Reader(parkShape).shapeRecords()

        parkFinder = {}
        parkSize = {}
        for s in parkSR:
            parkSize[s.record[4]] = s.record[19]
            parkFinder[s.record[4]] = cKDTree(s.shape.points)

        return parkFinder, parkSize

    def calculate_distances(loc, finder, size = None):
        Dist = {}
        for k in finder:
            Dist[k] = finder[k].query(loc, 1)[0]

        if size:
            toRet = {}
            for k in Dist:
                Dist[k] = (Dist[k], size[k], size[k]/(Dist[k]**2))
        return Dist

    def dfFromDict(dct):
        toRet = pd.DataFrame(dct)
        toRet = toRet.transpose()
        toRet.index = [idx for idx in toRet.index]

        if type(toRet.iloc[0,0]) == tuple:
            for c in toRet:
                toRet['P ' + str(c) + ' A'] = [e[1] for e in toRet[c]]
                toRet['P ' + str(c) + ' E'] = [e[2] for e in toRet[c]]
                toRet['P ' + str(c)] = [e[0] for e in toRet[c]]
                toRet = toRet.drop(c, axis = 'columns')
        else:
            toRet.columns = ['W ' + str(c) for c in toRet.columns]

        return toRet

    def info(df, finder, size = None):
        uniqueLocs = df['Location'].unique()
        rows = {}
        for loc in uniqueLocs:
            rows[loc] = calculate_distances(loc, finder, size)

        return dfFromDict(rows)

    def transform(df):
        toRet = pd.concat( [info(df, waterFinder),
                    info(df, parkFinder, parkSize)],
                    axis = 'columns')

        return toRet

    parkFinder, parkSize = buildParkDicts()
    waterFinder = buildWaterFinder()

    # Returns DFs: index = locations
    return transform(train), transform(test)

def SVD(train, test):

    def find_cols(df, tpe):
        mask = [c for c in df.columns if c[0] == tpe ]

        return df.loc[:,mask]

    def yeildFitTSVD(df):
        comps = 4

        TSVD = TruncatedSVD(n_components = comps)
        TSVD.fit(df)

        return TSVD

    def transformTSVD(df, TSVD,tpe):
        toRet = TSVD.transform(df)
        toRet = pd.DataFrame(toRet, index = df.index)
        toRet.columns = [tpe + str(c) for c in toRet.columns]

        return toRet


    toRetTrain = []
    toRetTest = []
    for t in ['W', 'P']:
        sTrain = find_cols(train, t)
        sTest = find_cols(test,t)
        sTSVD = yeildFitTSVD(sTrain)

        toRetTrain.append(transformTSVD(sTrain, sTSVD, t))
        toRetTest.append(transformTSVD(sTest, sTSVD, t))

    toRetTrain = pd.concat(toRetTrain, axis = 'columns')
    toRetTest = pd.concat(toRetTest, axis = 'columns')

    return toRetTrain, toRetTest

def WeatherProcess(train, test):

    def yeildWeather(target):
        weather = pd.read_csv(target)
        weather['Date'] = pd.to_datetime(weather['Date'])

        toDrop = ['Depart', 'Depth','Water1',
                'SnowFall', 'CodeSum', 'Heat',
                'Cool', 'Sunrise']
        weather = weather.drop(toDrop, axis=1)

        toReplace = {'M':np.nan, '  T': 0.001, '-': '0000'}
        for k in toReplace:
            weather = weather.replace(k, toReplace[k])


        toFloats = ['Tavg', 'WetBulb', 'PrecipTotal','StnPressure',
                    'SeaLevel', 'ResultSpeed','AvgSpeed']
        for c in toFloats:
            weather[c] = weather[c].astype(float)

        weather['Sunset'] = [date
                            if date[-2:] != '60'
                            else str(int(date[0:2])+1)+'00'
                            for date in weather['Sunset']]

        weather['Sunset'] = pd.to_datetime(weather['Sunset'],
                                            format="%H%M")
        weather.dropna(inplace=True)

        return weather[weather['Station']== 1]

    def yeildAvgTemp(weather):
        weather['Wk'] = weather['Date'].dt.week
        weekTemp = pd.DataFrame(
                        weather.groupby('Wk')['Tavg'].mean())
        weekTemp['Week'] = weekTemp.index - 17
        weekTemp['Week^2'] = weekTemp['Week']**2

        lr = LinearRegression().fit(weekTemp.drop('Tavg', axis = 'columns'),
                                    weekTemp['Tavg'])
        toRet = {}
        for w in range(1,53):
            toRet[w] = lr.intercept_ + (lr.coef_[0]*(w-17)) + (lr.coef_[1] * ((w-17)**2))

        return toRet

    def calculate_agregate( weather_sub, avgTDict):
        toRet = pd.Series()

        allAgg = [np.max, np.min, np.mean]
        toAgg = {'DewPoint': allAgg,
                'StnPressure': allAgg,
                'AvgSpeed': allAgg,
                'Tmax':[np.max],
                'Tmin':[np.min],
                'Tavg':[np.mean],
                'PrecipTotal':[np.sum, np.mean]
                }
        for k in toAgg:
            for f in toAgg[k]:
                toRet.loc[k + str(f).split(' ')[1]] = f(weather_sub[k])

        finalEntry = weather_sub.iloc[len(weather_sub)-1]

        toRet['temp_expected'] = avgTDict[pd.to_datetime(finalEntry['Date']).week]
        toRet['temp_diff'] = toRet['Tavgmean'] - toRet['temp_expected']

        sunset = finalEntry['Sunset']
        toRet['sunset'] = sunset.hour + (sunset.minute / 60)

        return toRet

    def date_ranges(dates):
        uniqueYears = set([pd.to_datetime(d).year for d in dates])

        dates = sorted(dates)
        fyear = []
        for y in uniqueYears:
            for d in dates:
                if pd.to_datetime(d).year == y:
                    fyear.append(d)
                    break

        for d in fyear:
            dates = np.insert(dates, 0, d - pd.Timedelta(days = 8))

        dateRanges = []
        for i in range(len(dates)-1):
            if pd.to_datetime(dates[i]).year == pd.to_datetime(dates[i+1]).year:
                dateRanges.append( (dates[i], dates[i+1]) )

        return dateRanges

    def subset_weather(dateRange, weather):
        mask = (weather['Date']>dateRange[0]) & (weather['Date'] <= dateRange[1])
        return weather.loc[mask]

    def TWeatherDFMaker(dct):
        toRet = pd.DataFrame().from_dict(dct)
        toRet = toRet.transpose()
        toRet.index = [idx for idx in toRet.index]
        toRet['Location'] = toRet.index.map(lambda x: x[0])
        toRet['Date'] = toRet.index.map(lambda x: x[1])
        toRet.index = range(len(toRet))

        return toRet

    def trap_agregator(trap_df, weather, avgTDict):
        trapWeather = {}
        loc = trap_df['Location'].iloc[0]

        dates = trap_df['Date'].unique()
        dates = sorted(dates)

        dateRanges = date_ranges(dates)

        for dr in dateRanges:
            weather_sub = subset_weather(dr, weather)
            trapWeather[(loc, dr[1])] = calculate_agregate(weather_sub, avgTDict)
        toRet = pd.DataFrame().from_dict(trapWeather)

        return TWeatherDFMaker(trapWeather)

    def transform(df):
        observations = []

        locs = df['Location'].unique()
        for l in locs:
            observations.append(trap_agregator(df[df['Location'] == l],
            weather, avgTDict))
        toRet = pd.concat(observations, axis = 'rows')

        return toRet

    weatherTarget = './input/weather.csv'
    weather = yeildWeather(weatherTarget)
    avgTDict = yeildAvgTemp(weather)

    return transform(train), transform(test)

def ProcessPipeline(train, test):
    train, test = agg_on_species(train, test)
    print('0',test.shape)
    train, test = InitPrepross(train, test)
    print('1',test.shape)
    trainW, testW = WeatherProcess(train, test)

    trainL, testL = LocationProcess(train, test)
    trainL, testL = SVD(trainL, testL)

    train = train.merge(trainL, left_on = 'Location', right_index = True)
    test = test.merge(testL, left_on = 'Location', right_index = True)
    print('3',test.shape)
    train = train.merge(trainW, on = ['Location','Date'])
    test = test.merge(testW,on = ['Location','Date'], how = 'left')
    print('2',test.shape)

    return train, test
