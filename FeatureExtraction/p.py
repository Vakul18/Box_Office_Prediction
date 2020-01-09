import math
import json
from datetime import datetime
import pandas as pd
import numpy as np
from multiprocessing import Process, Manager
import os
from itertools import repeat
import time


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


def CreateOrAddValueToCol(i, colName):
    dataFrameTrain.at[i, colName] = 1


def AddValueToCollectionColumn(i, value):
    dataFrameTrain.at[i, 'belongs_to_collection'] = value


if __name__ == '__main__':
	dir = os.path.dirname(__file__)
	filename = os.path.join(dir, 'train.csv')
	dataFrameTrain = pd.read_csv(filename)
	#dataFrameTrain = dataFrameTrain[:5]
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
	print('df1.Shape - ' + str(df1.shape[0]) + ' , ' + str(df1.shape[1]))
	dataFrameTrain = dataFrameTrain.join(df1, how='left')
	print('checkpoint 1')
	for t in p8List:
		AddValueToCollectionColumn(t[0], t[1])
	print('checkpoint 2')
	budgetCollection = dataFrameTrain.budget
	for i in range(0, len(budgetCollection)):
		budget = budgetCollection[i]
		if(budget == 0):
			dataFrameTrain.loc[i, 'budget'] = np.nan

	dict = {language: id for id, language in enumerate(
	    set(dataFrameTrain.original_language))}
	dataFrameTrain.original_language = [dict[language]
                                     for language in dataFrameTrain.original_language]

	releaseDateCol = dataFrameTrain['release_date']

	for i in range(0, len(releaseDateCol)):
		strDate = releaseDateCol[i]
		date = datetime.strptime(strDate, '%m/%d/%y')
		dataFrameTrain.at[i, 'day'] = date.day
		dataFrameTrain.at[i, 'month'] = date.month
		dataFrameTrain.at[i, 'year'] = date.year
	dataFrameTrain.drop(['genres','production_companies','production_countries','Keywords','cast','crew','homepage','imdb_id','original_title','overview','poster_path','release_date','spoken_languages','status','tagline','title'],axis=1)
	print('dataFrameTrain.Shape - ' +
	      str(dataFrameTrain.shape[0]) + ' , ' + str(dataFrameTrain	.shape[1]))
	dataFrameTrain.to_excel('E:\\finalMovieDatabse.xlsx')
	#df2 = pd.read_csv("E:\\finalMovieDatabse.csv")
	#print('df2.Shape - ' + str(df2.shape[0]) + ' , ' + str(df2.shape[1]))
