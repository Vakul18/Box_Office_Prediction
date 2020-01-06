import math
import json
from datetime import datetime
import pandas as pd
import numpy as np
from multiprocessing import Process
import threading
import os

dir = os.path.dirname(__file__)
filename = os.path.join(dir, 'train.csv')

dataFrameTrain = pd.read_csv(filename)

lock = threading.Lock()

def FindIdInColumn(column,callBack,fieldName):
    for i in range(0,len(column)):
        collectionJson = column[i]
        if type(collectionJson) !=str or collectionJson == '':
            continue
        idIndex = 0
        idIndex = collectionJson.find(fieldName,idIndex,len(collectionJson))
        while idIndex != -1:
            idStr = ''
            j = idIndex+5
            while j<len(collectionJson) and collectionJson[j]!=',':
                if not(collectionJson[j].isspace()) and collectionJson[j].isnumeric():
                    idStr = idStr + collectionJson[j]
                j=j+1
            callBack(i,idStr)
            idIndex = idIndex+2
            idIndex = collectionJson.find(fieldName,idIndex,len(collectionJson))


def CreateOrAddValueToCol(i,colName):
    lock.acquire()
    dataFrameTrain.at[i,colName] = 1
    lock.release()

def AddValueToCollectionColumn(i,value):
    lock.acquire()
    dataFrameTrain.at[i,'belongs_to_collection'] = value
    lock.release()





if __name__ == '__main__': 
	p1 = Process(target=FindIdInColumn,args=(dataFrameTrain['genres'],CreateOrAddValueToCol,'\'id\''))
	p2 = Process(target=FindIdInColumn,args=(dataFrameTrain['production_companies'],CreateOrAddValueToCol,'\'id\''))
	p3 = Process(target=FindIdInColumn,args=(dataFrameTrain['production_countries'],CreateOrAddValueToCol,'\'name\''))
	p4 = Process(target=FindIdInColumn,args=(dataFrameTrain['Keywords'],CreateOrAddValueToCol,'\'id\''))
	p5 = Process(target=FindIdInColumn,args=(dataFrameTrain['Keywords'],CreateOrAddValueToCol,'\'id\''))
	p6 = Process(target=FindIdInColumn,args=(dataFrameTrain['cast'],CreateOrAddValueToCol,'\'id\''))
	p7 = Process(target=FindIdInColumn,args=(dataFrameTrain['crew'],CreateOrAddValueToCol,'\'id\''))
	p8 = Process(target=FindIdInColumn,args=(dataFrameTrain['belongs_to_collection'],AddValueToCollectionColumn,'\'id\''))
	print('Process start')
	p1.start()
	p2.start()
	p3.start()
	p4.start()            

	p1.join()
	p2.join()
	p3.join()
	p4.join()
	print('second batch start')
	p5.start()
	p6.start()
	p7.start()
	p8.start()

	p5.join()
	p6.join()
	p7.join()
	p8.join()


	print('process joined')
	budgetCollection = dataFrameTrain.budget
	for i in range(0,len(budgetCollection)):
		budget = budgetCollection[i]
		if(budget==0):
			dataFrameTrain.loc[i,'budget'] = np.nan

	dict = {language: id for id, language in enumerate(set(dataFrameTrain.original_language))}
	dataFrameTrain.original_language = [dict[language] for language in dataFrameTrain.original_language]

	releaseDateCol = dataFrameTrain['release_date']

	for i in range(0,len(releaseDateCol)):
		strDate = releaseDateCol[i]
		date = datetime.strptime(strDate, '%m/%d/%y')
		dataFrameTrain.at[i,'day'] = date.day
		dataFrameTrain.at[i,'month'] = date.month
		dataFrameTrain.at[i,'year'] = date.year
dataFrameTrain.to_excel('finalMovieDatabse.xlsx')