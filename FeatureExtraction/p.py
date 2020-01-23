import math
import json
from datetime import datetime
import pandas as pd
import numpy as np
from multiprocessing import Process, Manager
import os
from itertools import repeat
import time
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder

def FindIdInColumn(colName, column, fieldName, returnList, appendColName=True):
    for i in range(0, len(column)):
        collectionJson = column[i]
        if type(collectionJson) != str or collectionJson == '':
            continue
        idIndex = 0
        idIndex = collectionJson.find(fieldName, idIndex, len(collectionJson))
        while idIndex != -1:
            idStr = ''
            j = idIndex + len(fieldName) + 2
            while j < len(collectionJson) and collectionJson[j] != ',':
                if not(collectionJson[j].isspace()):
                    idStr = idStr + collectionJson[j]
                j = j+1
            if appendColName:
                returnList.append((i, colName + '_' + idStr))
            else:
                returnList.append((i, idStr))
            idIndex = idIndex+2
            idIndex = collectionJson.find(fieldName, idIndex, len(collectionJson))
    return returnList

def day_of_week(strDate):
	date = datetime.strptime(strDate, '%m/%d/%y')
	year = date.year
	month = date.month
	day = date.day
	t = [0, 3, 2, 5, 0, 3, 5, 1, 4, 6, 2, 4]
	year -= month < 3
	return (year + int(year/4) - int(year/100) + int(year/400) + t[month-1] + day) % 7

def ProcessCategoryColumns(dataFrameTrain):
	manager = Manager()
	dictList = []
	p1List = manager.list()
	dictList.append(p1List)
	p2List = manager.list()
	dictList.append(p2List)
	p3List = manager.list()
	dictList.append(p3List)
	p4List = manager.list()
	dictList.append(p4List)
	p5List = manager.list()
	dictList.append(p5List)
	p6List = manager.list()
	dictList.append(p6List)
	p7List = manager.list()
	dictList.append(p7List)
	p8List = manager.list()
	processList = []
	p1 = Process(target=FindIdInColumn, args=('genres', dataFrameTrain['genres'],'\'id\'',p1List))
	processList.append(p1)
	p2 = Process(target=FindIdInColumn, args=('production_companies', dataFrameTrain['production_companies'],'\'id\'',p2List))
	processList.append(p2)
	p3 = Process(target=FindIdInColumn, args=('production_countries', dataFrameTrain['production_countries'],'\'iso_3166_1\'',p3List))
	processList.append(p3)
	p4 = Process(target=FindIdInColumn, args=('Keywords', dataFrameTrain['Keywords'],'\'id\'',p4List))
	processList.append(p4)
	p6 = Process(target=FindIdInColumn, args=('cast', dataFrameTrain['cast'],'\'id\'',p6List))
	processList.append(p6)
	p7 = Process(target=FindIdInColumn, args=('crew', dataFrameTrain['crew'],'\'id\'',p7List))
	processList.append(p7)
	p8 = Process(target=FindIdInColumn, args=('belongs_to_collection', dataFrameTrain['belongs_to_collection'],'\'id\'',p8List,False))
	processList.append(p8)
	print('Process start')
	for process in processList:
		process.start()
	for process in processList:
		process.join()
	print('process joined')
	dfRowCount = dataFrameTrain.shape[0]
	d = {}
	for pList in dictList:
		for t in pList:
			if t[1] in d.keys():
				d[t[1]][t[0]] = 1
			else:
				l = list(repeat(0, dfRowCount))
				l[t[0]] = 1
				d[t[1]] = l
	df1 = pd.DataFrame.from_dict(d)
	dataFrameTrain = dataFrameTrain.join(df1, how='left')
	for t in p8List:
		dataFrameTrain.at[t[0], 'belongs_to_collection'] = t[1]
	return dataFrameTrain

def OneHotEncodeColumns(dataFrameTrain):
	XreleaseDateCol = dataFrameTrain['release_date'].apply(day_of_week)
	oneHotEncoderReleaseDate = OneHotEncoder()
	XoneEncodedValuesReleasDate = oneHotEncoderReleaseDate.fit_transform(np.asarray(XreleaseDateCol).reshape(-1,1)).toarray()
	
	label_encoderBelongs = LabelEncoder()
	dataFrameTrain['belongs_to_collection'] = dataFrameTrain['belongs_to_collection'].fillna(0,inplace = True)
	dataFrameTrain['belongs_to_collection'] = label_encoderBelongs.fit_transform(dataFrameTrain['belongs_to_collection'])
	oneHotEncoderBelongs = OneHotEncoder()
	XoneEncodedValuesBelongs = oneHotEncoderBelongs.fit_transform(np.asarray(dataFrameTrain['belongs_to_collection']).reshape(-1,1)).toarray()
	
	oneHotEncoderLang = OneHotEncoder()
	label_encoderLang = LabelEncoder()
	dataFrameTrain['original_language'] =label_encoderLang.fit_transform(dataFrameTrain['original_language'])
	XoneHotEncodedLang = oneHotEncoderLang.fit_transform(np.asarray(dataFrameTrain['original_language']).reshape(-1,1)).toarray()
	
	return dataFrameTrain,oneHotEncoderReleaseDate,XoneEncodedValuesReleasDate,label_encoderBelongs,oneHotEncoderBelongs,XoneEncodedValuesBelongs,label_encoderLang,oneHotEncoderLang,XoneHotEncodedLang

def ExtractFeatures(dataFrameTrain):
	dataFrameTrain = ProcessCategoryColumns(dataFrameTrain)
	
	releaseDateCol = dataFrameTrain['release_date']
	for i in range(0, len(releaseDateCol)):
		strDate = releaseDateCol[i]
		date = datetime.strptime(strDate, '%m/%d/%y')
		dataFrameTrain.at[i, 'day'] = date.day
		dataFrameTrain.at[i, 'month'] = date.month
		dataFrameTrain.at[i, 'year'] = date.year
	
	dataFrameTrain,oneHotEncoderReleaseDate,XoneEncodedValuesReleasDate,label_encoderBelongs,oneHotEncoderBelongs,XoneEncodedValuesBelongs,label_encoderLang,oneHotEncoderLang,XoneHotEncodedLang = OneHotEncodeColumns(dataFrameTrain)
	
	budgetCollection = dataFrameTrain.budget
	for i in range(0, len(budgetCollection)):
		budget = budgetCollection[i]
		if(budget == 0):
			dataFrameTrain.loc[i, 'budget'] = np.nan

	
	dataFrameTrain = dataFrameTrain.drop(['original_language','id','genres','production_companies','production_countries','Keywords','belongs_to_collection','cast','crew','homepage','imdb_id','original_title','overview','poster_path','release_date','spoken_languages','status','tagline','title','revenue'],axis=1)
	XTrain = np.concatenate((dataFrameTrain.to_numpy(),XoneEncodedValuesBelongs, XoneEncodedValuesReleasDate,XoneHotEncodedLang),axis = 1)
	return oneHotEncoderReleaseDate,label_encoderBelongs,oneHotEncoderBelongs,label_encoderLang,oneHotEncoderLang,XTrain

if __name__ == '__main__':
	dir = os.path.dirname(__file__)
	filename = os.path.join(dir, 'train.csv')
	dataFrameTrain = pd.read_csv(filename)
	
	oneHotEncoderReleaseDate,label_encoderBelongs,oneHotEncoderBelongs,label_encoderLang,oneHotEncoderLang,XTrain = ExtractFeatures(dataFrameTrain)

	YTrain =  dataFrameTrain['revenue'].to_numpy()

	print(XTrain.shape)
	print(YTrain.shape)
	#dataFrameTrain.to_csv('E:\\finalMovieDatabse.csv')
	#df2 = pd.read_csv("E:\\finalMovieDatabse.csv")
	#print('df2.Shape - ' + str(df2.shape[0]) + ' , ' + str(df2.shape[1]))
