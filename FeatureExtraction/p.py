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
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.impute import KNNImputer
from sklearn.decomposition import PCA



def GetValue(idIndex,fieldNameLength,collectionJson):
  j = idIndex + fieldNameLength + 2
  idStr = ''
  while j < len(collectionJson) and collectionJson[j] != ',':
    if not(collectionJson[j].isspace()):
        idStr = idStr + collectionJson[j]
    j = j+1
  return idStr


def FindIdForSpecificField(colName, column, fieldName, returnList,anchorFieldValue,appendColName=True):
  for i in range(0, len(column)):
    collectionJson = column[i]
    if type(collectionJson) != str or collectionJson == '':
        continue
    anchorValueIndex = 0  
    anchorValueIndex = collectionJson.find(anchorFieldValue, anchorValueIndex, len(collectionJson))
    idIndex =  collectionJson.rfind(fieldName,0,anchorValueIndex)
    idStr = GetValue(idIndex,len(fieldName),collectionJson)
    returnList.append((i, colName + '_' + idStr,returnList))



def FindIdInColumn(colName, column, fieldName, returnList, appendColName=True):
  fieldNameLength = len(fieldName)
  for i in range(0, len(column)):
      collectionJson = column[i]
      if type(collectionJson) != str or collectionJson == '':
          continue
      idIndex = 0
      idIndex = collectionJson.find(fieldName, idIndex, len(collectionJson))
      while idIndex != -1:
          idStr = GetValue(idIndex,fieldNameLength,collectionJson)
          if appendColName:
              returnList.append((i, colName + '_' + idStr,returnList))
          else:
              returnList.append((i, idStr,returnList))
          idIndex = idIndex+2
          idIndex = collectionJson.find(fieldName, idIndex, len(collectionJson))

def day_of_week(strDate):
	date = datetime.strptime(strDate, '%m/%d/%y')
	year = date.year
	month = date.month
	day = date.day
	t = [0, 3, 2, 5, 0, 3, 5, 1, 4, 6, 2, 4]
	year -= month < 3
	return (year + int(year/4) - int(year/100) + int(year/400) + t[month-1] + day) % 7

def UpdateDictionaryFromList(d,t,rowCount):
	if t[1] in d.keys():
				d[t[1]][t[0]] = 1
	else:
		l = list(repeat(0, rowCount))
		l[t[0]] = 1
		d[t[1]] = l

def ProcessCategoryColumns(dataFrame):
	manager = Manager()
	dictList = []
	p1List = manager.list()
	dictList.append(p1List)
	p2List = manager.list()
	#dictList.append(p2List)
	p3List = manager.list()
	dictList.append(p3List)
	p4List = manager.list()
	#dictList.append(p4List)
	p5List = manager.list()
	dictList.append(p5List)
	p6List = manager.list()
	#dictList.append(p6List)
	p7List = manager.list()
	#dictList.append(p7List)
	p8List = manager.list()
	processList = []
	p1 = Process(target=FindIdInColumn, args=('genres', dataFrame['genres'],'\'id\'',p1List))
	processList.append(p1)
	p2 = Process(target=FindIdInColumn, args=('production_companies', dataFrame['production_companies'],'\'id\'',p2List))
	processList.append(p2)
	p3 = Process(target=FindIdInColumn, args=('production_countries', dataFrame['production_countries'],'\'iso_3166_1\'',p3List))
	processList.append(p3)
	p4 = Process(target=FindIdInColumn, args=('Keywords', dataFrame['Keywords'],'\'id\'',p4List))
	processList.append(p4)
	p6 = Process(target=FindIdInColumn, args=('cast', dataFrame['cast'],'\'id\'',p6List))
	processList.append(p6)
	p7 = Process(target=FindIdForSpecificField, args=('crew', dataFrame['crew'],'\'id\'',p7List,'\'Director\'',True))
	processList.append(p7)
	p8 = Process(target=FindIdInColumn, args=('belongs_to_collection', dataFrame['belongs_to_collection'],'\'id\'',p8List,False))
	processList.append(p8)
	print('Process start')
	for process in processList:
		process.start()
	for process in processList:
		process.join()
	print('process joined')
	dfRowCount = dataFrame.shape[0]
	
	d = {}
	for pList in dictList:
		for t in pList:
			UpdateDictionaryFromList(d,t,dfRowCount)

	df1 = pd.DataFrame.from_dict(d)
	dataFrame = dataFrame.join(df1, how='left')
	
	for t in p8List:
		dataFrame.at[t[0], 'belongs_to_collection'] = t[1]
	
	
	dataFrame,pcaProd = ReduceColumns(p2List, dfRowCount, dataFrame,0.5,'Prod')
	
	dataFrame,pcaCast = ReduceColumns(p6List, dfRowCount, dataFrame,0.5,'Cast')

	dataFrame,pcaKeyword = ReduceColumns(p4List, dfRowCount, dataFrame,0.5,'Keyword')

	dataFrame,pcaCrew = ReduceColumns(p7List, dfRowCount, dataFrame,0.5,'Crew')

	return dataFrame

def ReduceColumns(processList, dfRowCount, dataFrame,variance,name):
    
	dictDf = {}
	for t in processList:
		UpdateDictionaryFromList(dictDf,t,dfRowCount)

	df = pd.DataFrame.from_dict(dictDf)
	X = df.to_numpy()
	print(name + 'X.shape = '+ str(X.shape))
	pca=  PCA(variance)
	XRed =  pca.fit_transform(X)
	print(name + 'XRed.shape = '+ str(XRed.shape))
	dataFrameRed = pd.DataFrame(XRed)
	dataFrameRed.columns = [str(col_name) + '_' + name for col_name in dataFrameRed.columns]
	dataFrame = dataFrame.join(dataFrameRed)
	return dataFrame,pca

def OneHotEncodeColumns(dataFrame,oneHotEncoderReleaseDate,label_encoderBelongs,oneHotEncoderBelongs,oneHotEncoderLang,label_encoderLang):
	XreleaseDateCol = dataFrame['release_date'].apply(day_of_week)
	XoneEncodedValuesReleasDate = oneHotEncoderReleaseDate.fit_transform(np.asarray(XreleaseDateCol).reshape(-1,1)).toarray()
	
	dataFrame['belongs_to_collection'] = dataFrame['belongs_to_collection'].fillna(0,inplace = True)
	dataFrame['belongs_to_collection'] = label_encoderBelongs.fit_transform(dataFrame['belongs_to_collection'])
	XoneEncodedValuesBelongs = oneHotEncoderBelongs.fit_transform(np.asarray(dataFrame['belongs_to_collection']).reshape(-1,1)).toarray()
	
	dataFrame['original_language'] =label_encoderLang.fit_transform(dataFrame['original_language'])
	XoneHotEncodedLang = oneHotEncoderLang.fit_transform(np.asarray(dataFrame['original_language']).reshape(-1,1)).toarray()
	
	return dataFrame,XoneEncodedValuesReleasDate,XoneEncodedValuesBelongs,XoneHotEncodedLang

def ExtractFeatures(dataFrame,oneHotEncoderReleaseDate,label_encoderBelongs,oneHotEncoderBelongs,oneHotEncoderLang,label_encoderLang):
	
	dataFrame = ProcessCategoryColumns(dataFrame)
	belongsCount = 0
	genreCount = 0
	production_companiesCount = 0
	KeywordsCount = 0
	crewCount = 0
	#castCount = 0
	production_countriesCount = 0
	for col in dataFrame.columns:
		
		if(isinstance(col,int)):
			continue

		if 'belongs_to_collection' in col:
			belongsCount = belongsCount + 1
		
		if 'genres' in col:
			genreCount = genreCount + 1
		
		if 'production_companies' in col:
			production_companiesCount = production_companiesCount + 1
		
		if 'Keywords' in col:
			KeywordsCount = KeywordsCount + 1
		
		if 'crew' in col:
			crewCount = crewCount + 1
		
		#if 'cast' in col:
		#	castCount = castCount + 1
		
		if 'production_countries' in col:
			production_countriesCount = production_countriesCount + 1

	print('belong = ' + str(belongsCount))
	print('genre = ' + str(genreCount))
	print('production_companiesCount = ' + str(production_companiesCount))
	print('KeywordsCount = ' + str(KeywordsCount))
	print('crewCount = ' + str(crewCount))
	#print('castCount = ' + str(castCount))
	print('production_countriesCount = ' + str(production_countriesCount))

	releaseDateCol = dataFrame['release_date']
	for i in range(0, len(releaseDateCol)):
		strDate = releaseDateCol[i]
		date = datetime.strptime(strDate, '%m/%d/%y')
		dataFrame.at[i, 'day'] = date.day
		dataFrame.at[i, 'month'] = date.month
		dataFrame.at[i, 'year'] = date.year


	dataFrame,XoneEncodedValuesReleasDate,XoneEncodedValuesBelongs,XoneHotEncodedLang = OneHotEncodeColumns(dataFrame,oneHotEncoderReleaseDate,label_encoderBelongs,oneHotEncoderBelongs,oneHotEncoderLang,label_encoderLang)
	
	budgetCollection = dataFrame.budget
	for i in range(0, len(budgetCollection)):
		budget = budgetCollection[i]
		if(budget == 0):
			dataFrame.loc[i, 'budget'] = np.nan


	scaler = StandardScaler()
	dataFrame[['day','month','year','budget','popularity','runtime']] = scaler.fit_transform(dataFrame[['day','month','year','budget','popularity','runtime']])
	
	dataFrame = dataFrame.drop(['original_language','id','genres','production_companies','production_countries','Keywords','belongs_to_collection','cast','crew','homepage','imdb_id','original_title','overview','poster_path','release_date','spoken_languages','status','tagline','title','revenue'],axis=1)
	XTrain = np.concatenate((dataFrame.to_numpy(),XoneEncodedValuesBelongs, XoneEncodedValuesReleasDate,XoneHotEncodedLang),axis = 1)
	return XTrain

if __name__ == '__main__':
	dir = os.path.dirname(__file__)
	filename = os.path.join(dir, 'train.csv')
	dataFrameTrain = pd.read_csv(filename)
	
	oneHotEncoderReleaseDate = OneHotEncoder()
	label_encoderBelongs = LabelEncoder()
	oneHotEncoderBelongs = OneHotEncoder()
	oneHotEncoderLang = OneHotEncoder()
	label_encoderLang = LabelEncoder()
	XTrain = ExtractFeatures(dataFrameTrain,oneHotEncoderReleaseDate,label_encoderBelongs,oneHotEncoderBelongs,oneHotEncoderLang,label_encoderLang)
	YTrain =  dataFrameTrain['revenue'].to_numpy()
	
	print(XTrain.shape)
	print(YTrain.shape)
	print('start imputing')
	imputer = KNNImputer(n_neighbors=5)
	XTrain = imputer.fit_transform(XTrain)
	print('finish imputing')
	df12 = pd.DataFrame (XTrain)
	#df12.to_excel('E:\\n.xlsx')
	degrees = [2,3,4,5,8]
	for i in range(0,len(degrees)):
		print('start regression for degree' + str(degrees[i]))
		polyFeature = PolynomialFeatures(degree=degrees[i])
		linearRegression = LinearRegression()
		#scoring = 'r2'
		pipeline = Pipeline([('polyFeature',polyFeature),('linearRegression',linearRegression)])
		score = cross_val_score(pipeline,XTrain,YTrain,n_jobs=4,cv=5)
		#print('score for degree = ' + str(degrees[i])+ " is " + str(score))
		print("Accuracy: %0.2f (+/- %0.2f) for degree = %d" % (score.mean(), score.std() * 2,degrees[i]))


	#print(XTrain)
	#dataFrameTrain.to_csv('E:\\finalMovieDatabse.csv')
	#df2 = pd.read_csv("E:\\finalMovieDatabse.csv")
	#print('df2.Shape - ' + str(df2.shape[0]) + ' , ' + str(df2.shape[1]))
