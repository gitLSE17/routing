from sklearn.linear_model import LinearRegression as LR
from pyDOE import *
import pandas as pd
import numpy as np
import pickle
from gurobipy import * 
import math as mt
import random
import time

#load data
airspaceDF = pd.read_csv("./XX_airspaces.csv", sep=";", index_col=0)
sectorDF = pd.read_csv("./XX_elemsec.csv", sep=";", index_col=0)
terminalDF = pd.read_csv("./XX_terminal.csv", sep=";", index_col=0)
collapDF = pd.read_csv("./XX_collsec.csv", sep=";", index_col=1) #NOT COMPLETE YET
configDF = pd.read_csv("./XX_configurations.csv", sep=";", index_col=1) #NOT COMPLETE YET
flightDF = pd.read_csv("./XX_flights.csv", sep=";", index_col=0)
routesDF = pd.read_csv("./XX_routes.csv", sep=";")
rcostDF = pd.read_csv("./XX_routcost.csv", sep=";", index_col=0)

#----------------define sets, parameters and iterators----------------------
#iterators
num_U = 48
start_time = 0 #beginning in minutes
times = 30 #number of times small time period fits into u

#sets
set_U = [i for i in range(num_U)] #booking time periods
set_AS = airspaceDF.index.tolist() #airspaces 
set_AP = terminalDF.index.tolist() #terminal airspaces
set_A = [*set_AS, *set_AP]
set_Sa = {a: [sectorDF.iloc[sectorDF.index.get_loc(a),1+j] for j in range(sectorDF.loc[a,'#elemsec'])] for a in set_A} #airspace sectors fir airsace (num)
set_S = [set_Sa[a][i] for a in set_A for i in range(len(set_Sa[a]))] #elementary sectors 
set_P = collapDF.index.tolist()  #collapsed sectors
set_Pa = {a: collapDF.index[collapDF['Airspace']==a].tolist()  for a in set_A}
set_Sp = {p: [collapDF.iloc[collapDF.index.get_loc(p),4+j] for j in range(collapDF.loc[p,'#elemsec'])] for p in set_P} # lists of elementary sectors indexes for each collapsed sector p
#set_C =   configDF.index.tolist() #configurations 
set_Ca = {a: configDF.index[configDF['Airspace']==a].tolist() for a in set_AS} #configurations for each airspace
set_C = [set_Ca[a][i] for a in set_AS for i in range(len(set_Ca[a]))]
set_Pc = {c: [configDF.iloc[configDF.index.get_loc(c),2+j] for j in range(configDF.loc[c,'#collsec'])] for c in set_C} #lists of collapsed sectors for each configuration c
#set_F = flightDF.index.tolist() #flights 
set_FS = flightDF.index[flightDF['f_type'] =="S"].tolist()
set_FU = flightDF.index[flightDF['f_type'] =="N"].tolist()
set_F = [*set_FS, *set_FU]
set_R = rcostDF.index.tolist() 

#parameters
Q = {p: mt.ceil(collapDF.loc[p,'Cap']/2) for p in set_P} #capacity for each collapsed sector
prob_QE = {a: airspaceDF.loc[a,'ProbE'] for a in set_A} # probability for ATFM regulation due to staffing issue
#dur_QE = 12
dur_QE = {a: airspaceDF.loc[a,'DurE'] for a in set_A}
prob_QW = {a: airspaceDF.loc[a,'ProbW'] for a in set_A}
dur_QW = {a: airspaceDF.loc[a,'DurW'] for a in set_A}

#---------------FUNCTIONS-------------------
     
def createScen(num_scen, k):
    set_Fi = {i: [] for i in range(num_scen)} # flight set for each scenario
    QEi = {i: {} for i in range(num_scen)} # employee absence for each scenario
   # QWi = {i: [{p: Q[p] for p in set_P} for u in set_U] for i in range(num_scen)} # weather uncertainty for each scenario
    QWi = {i: {} for i in range(num_scen)} # weather uncertainty for each scenario
    U_unc = [i for i in range(10,42)]
    random.seed(k)
    np.random.seed(k)
    
    #create standard scenario 0
    set_FUi = random.sample(set_FU, 4000)
    set_Fi[0] = random.sample(set_FUi + set_FSi, len(set_FUi + set_FSi)) 
    
    for i in range(1,num_scen):
        set_FUi = random.sample(set_FU, min(max(round(np.random.normal(4000, 600,1)[0]),0),len(set_FU)))
        set_Fi[i] = random.sample(set_FUi + set_FSi, len(set_FUi + set_FSi)) 
                
        # weather uncertainty
        for a in set_A:
            if np.random.choice([0,1], p = [1-prob_QW[a], prob_QW[a]]) == 1:
                si = random.sample(set_Sa[a], 1)[0] 
                ui = random.sample(U_unc, 1)[0]  
                QWi_unc = np.random.choice([0.95, 0.9, 0.75], p = [0.4, 0.4, 0.2]) #
                QWi[i][a,si,ui] = QWi_unc
        # employee uncertainty
        for a in set_A:
            if np.random.choice([0,1], p = [1-prob_QE[a], prob_QE[a]]) == 1:
                ui = random.sample(U_unc, 1)[0] 
                QEi[i][a,ui] = 1
    return set_Fi, QWi, QEi         

#------------------------------- MASTER CODE ----------------------------------
set_FSi = random.sample(set_FS, 30000) 
set_FU.extend([j for j in set_FS if j not in set_FSi])
print(len(set_FS))
set_Fi, QWi, QEi = createScen(200,1)
print([len(set_Fi[i]) for i in range(30)])

pickle.dump(set_Fi, open( "set_Fi1_34.p", "wb" ) )
#pickle.dump(QEi, open( "QEi2.p", "wb" ) )
#pickle.dump(QWi, open( "QWi1.p", "wb" ) )
print('done')

