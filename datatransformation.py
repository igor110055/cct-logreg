import pyodbc
import pandas as pd
import urllib
import time
import numpy as np
import datetime


def importdb(Sql):
    cnxn = pyodbc.connect('DRIVER={SQL Server};SERVER=54.154.28.119,1433;DATABASE=4E;UID=daniele_sicoli;PWD=Landau#1')
    cursor = cnxn.cursor()

    cursor.execute(Sql)

    columns = [column[0] for column in cursor.description]
    rows = cursor.fetchall()
    pdf = pd.DataFrame.from_records(rows, columns=columns)


    cursor.close()
    cnxn.close()

    return pdf


def UploadDfToSqlTable(dfToUpload = None, columns = [], sqlTableInfo = {'db':'SIM', 'table':''}):
    if ((sqlTableInfo['table'] == '') | (dfToUpload is None)):
        return 1
    cnxn = pyodbc.connect(f'DRIVER={{SQL Server}};SERVER=54.154.28.119,1433;DATABASE={sqlTableInfo["db"]};UID=daniele_sicoli;PWD=Landau#1')
    cursor = cnxn.cursor()
 
    for index, row in dfToUpload.iterrows():
        print(f'insert into {sqlTableInfo["db"]}.{sqlTableInfo["table"]} ({",".join(columns)}) values({", ".join([str(row[k]) for k in range(len(row))])})')
        cursor.execute(f'insert into {sqlTableInfo["db"]}.{sqlTableInfo["table"]} ({",".join(columns)}) values({", ".join([str(row[k]) for k in range(len(row))])})')
        cnxn.commit()
        
    cursor.close()

def AddColumnsToSqlTable(columns = [], dataTypes = [], defaults = [None], sqlTableInfo = {'db':'SIM', 'table':''}):
    if sqlTableInfo['table'] == '':
        return 1
        
    cnxn = pyodbc.connect(f'DRIVER={{SQL Server}};SERVER=54.154.28.119,1433;DATABASE={sqlTableInfo["db"]};UID=daniele_sicoli;PWD=Landau#1')
    cursor = cnxn.cursor()
    
    for k in range(len(columns)):
        cursor.execute(f'alter table {sqlTableInfo["db"]}.{sqlTableInfo["table"]} add {columns[k]} {dataTypes[k]}')
        cursor.execute(f'UPDATE {sqlTableInfo["db"]}.{sqlTableInfo["table"]} SET {columns[k]} = {defaults[k]}')
        cnxn.commit()
    cursor.close()

def ExecuteDownloadFromDb(sql, verbose = False):
    startTime = time.time()
    data = importdb(sql)
    endTime = time.time()
    if verbose == True:
        print(f"data from {sql} took --- %s seconds --- to download" % (endTime - startTime))

    return data

# def GroupByApplyAndMaintainRows(df, groupBy, actOnAndFuncsToApplyDict):
#     dfAggr = df.groupby(groupBy).agg(actOnAndFuncsToApplyDict).reset_index()
#     rightOn = GetGroupByColumnNames(dfAggr.columns, groupBy)
#     dfAggr.columns = GetAggrColumnNames(dfAggr.columns, groupBy)
#     dfWithGroupedValues = pd.merge(df, dfAggr, left_on = groupBy, right_on = rightOn, how = 'left')

#     return dfWithGroupedValues

def AddLeadingZerosAndJoin(strings):
    stringsWithZeros = [item.zfill(2) for item in strings]
    joinedStringsWithZeros = '_'.join(stringsWithZeros)
    return joinedStringsWithZeros

class GroupByTableInfo:
    def __init__(self, groupBy, orderBy, applyOnVariable, rollWindow = 5):
        self.groupBy = groupBy
        self.orderBy = orderBy
        self.applyOnVariable = applyOnVariable
        self.rollWindow = rollWindow

def AddLeadingZerosAndJoin(strings):
    stringsWithZeros = [item.zfill(2) for item in strings]
    joinedStringsWithZeros = '_'.join(stringsWithZeros)
    return joinedStringsWithZeros

def GroupByApplyAndMaintainRows(df, groupByTableInfo, FunctionApplied, appendToNewColName = ''):
    try:
        functionName = FunctionApplied.__name__
        newVariable = groupByTableInfo.applyOnVariable+'_' + functionName + appendToNewColName

        groupByJoined = '_'.join(groupByTableInfo.groupBy)
        df[groupByJoined] = df[groupByTableInfo.groupBy].astype(str).agg(AddLeadingZerosAndJoin, axis=1)

        df = df.sort_values(groupByTableInfo.orderBy)

        df[newVariable] = df.groupby(groupByTableInfo.groupBy)[groupByTableInfo.applyOnVariable].apply(lambda x: FunctionApplied(x)).values

    except Exception as error:
        print('Caught this error: ' + repr(error))
    return df

def GroupByRollApplyAndMaintainRows(df, groupByTableInfo, FunctionApplied, appendToNewColName = ''):
    try:
        functionName = FunctionApplied.__name__
        newVariable = groupByTableInfo.applyOnVariable+ '_' + functionName + appendToNewColName

        groupByJoined = '_'.join(groupByTableInfo.groupBy)
        df[groupByJoined] = df[groupByTableInfo.groupBy].astype(str).agg(AddLeadingZerosAndJoin, axis=1)

        df = df.sort_values(groupByTableInfo.orderBy)

        df[newVariable] = df.groupby(groupByTableInfo.groupBy)[groupByTableInfo.applyOnVariable].rolling(window=groupByTableInfo.rollWindow, 
            center = False).apply(lambda x: FunctionApplied(x)).values

    except Exception as error:
        print('Caught this error: ' + repr(error))
    return df

def AddBandsTableToDf(bands, df, dfDatetimeColName = 'dataora'):
    bands = bands[['PRICING','data','DATA_GME','MESE','ora', 'GIORNO_TIPO','F123','POP']]
    bands = bands.sort_values(by = ['PRICING'], ascending = True)

    df.head(5)
    df[dfDatetimeColName] = pd.to_datetime(df[dfDatetimeColName])
    df = df.sort_values(by = [dfDatetimeColName], ascending = True)
    df = pd.merge_asof(df, bands, left_on=dfDatetimeColName, right_on='PRICING')

    return df

def GetGroupByColumnNames(tupleList, groupBy):
    n = len(groupBy)
    tupleElements = []
    k = 0
    for tpl in tupleList:
        if k < n:
            tupleElements.append(tpl[0])
            k = k + 1
        else:
            continue

    return tupleElements

def GetAggrColumnNames(tupleList, groupBy):
    n = len(groupBy)
    tupleElements = []
    k = 0
    for tpl in tupleList:
        if k < n:
            tupleElements.append(tpl[0])
            k = k + 1
        else:
            tupleElements.append(tpl[0]+'_'+tpl[1])

    return tupleElements

def BuildConcatFunc(groupBy):
    def Concat(row):
        return '_'.join([str(row[g]).zfill(2) for g in groupBy])
 
    return Concat

def CreateStandardizedColumns(df, colNames):
    #you need to import numpy to use this function
    newVarNames = []
    for colName in colNames:
        df[colName] = df[colName].fillna(0)
        newVarName = colName + '_standardized'
        df[colName+'_mean'] = np.average(df[colName])
        df[colName+'_std'] = np.std(df[colName])
        
        df[colName+'_standardized'] = (df[colName]-np.average(df[colName]))/np.std(df[colName])
        newVarNames.append(newVarName)

    return df, newVarNames

def CreateDestandardizedColumns(df, columnToDestandardize, avg, stdev):
    #you need to import numpy to use this function
    
    for colName in columnToDestandardize:
        df[str(colName) + '_destandardized'] = df[stdev]*df[colName] + df[avg]

    return df