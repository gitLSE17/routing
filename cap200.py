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
set_A = airspaceDF.index.tolist() #airspaces 
set_AP = airspaceDF.index[airspaceDF['a_type'] =="TMA"].tolist() #terminal airspaces
set_Aup = airspaceDF.index[airspaceDF['a_type'] =="Upper"].tolist() #terminal airspaces
set_Alow = airspaceDF.index[airspaceDF['a_type'] =="Lower"].tolist() #terminal airspaces
set_AS = [*set_Aup, *set_Alow]
set_Sa = {a: [sectorDF.iloc[sectorDF.index.get_loc(a),1+j] for j in range(sectorDF.loc[a,'#elemsec'])] for a in set_A} #airspace sectors fir airsace (num)
set_S = [set_Sa[a][i] for a in set_A for i in range(len(set_Sa[a]))] #elementary sectors 
set_P = collapDF.index.tolist()  #collapsed sectors
set_Pa = {a: collapDF.index[collapDF['Airspace']==a].tolist()  for a in set_A}
set_Sp = {p: [collapDF.iloc[collapDF.index.get_loc(p),4+j] for j in range(collapDF.loc[p,'#elemsec'])] for p in set_P} # lists of elementary sectors indexes for each collapsed sector p
set_Ps = {s: [p for p in set_P if s in set_Sp[p]] for s in set_S}
set_C =   configDF.index.tolist() #configurations 
set_Ca = {a: configDF.index[configDF['Airspace']==a].tolist() for a in set_A} #configurations for each airspace
#set_C = [set_Ca[a][i] for a in set_AS for i in range(len(set_Ca[a]))]
set_Pc = {c: [configDF.iloc[configDF.index.get_loc(c),2+j] for j in range(configDF.loc[c,'#collsec'])] for c in set_C} #lists of collapsed sectors for each configuration c
set_F = flightDF.index.tolist() #flights 
set_FS = flightDF.index[flightDF['f_type'] =="S"].tolist()
set_FU = flightDF.index[flightDF['f_type'] =="N"].tolist()
set_F = [*set_FS, *set_FU]
set_R = rcostDF.index.tolist() 
airconfig = tuplelist([(c,u) for c in set_C for u in set_U]) #airspace-configuration combinations
resources = tuplelist([(s,u) for s in set_S for u in set_U]) #sector-time combinations

#parameters
hours = {c: configDF.loc[c,'#collsec'] for c in set_C}
cost = {a: airspaceDF.loc[a,'Cost']/2 for a in set_A} #cost per sector-hour for airspace and configuration
Q = {p:  mt.ceil(1.0*collapDF.loc[p,'Cap']/2) for p in set_P} #capacity for each collapsed sector
dur_QE = {a: airspaceDF.loc[a,'DurE'] for a in set_A}
dur_QW = {a: airspaceDF.loc[a,'DurW'] for a in set_A }

rout_time = {r: [] for r in set_R}
rout_sec = {r: [] for r in set_R}
dept = {r: [] for r in set_R}
dept[routesDF.iloc[0,1]] = routesDF.iloc[0,3] 
for j in range(490221):
    rout_time[routesDF.iloc[j,1]].extend([routesDF.iloc[j,4]/60])
    rout_sec[routesDF.iloc[j,1]].extend([routesDF.iloc[j,2]])
    if j > 0 and routesDF.iloc[j,1] != routesDF.iloc[j-1,1]:
        dept[routesDF.iloc[j,1]] = routesDF.iloc[j,3] 
rout_U = {r: [j for j in range(int(dept[r]//times), int((dept[r]+ sum(rout_time[r]))//times))] for r in set_R}
set_Rs = {s: [r for r in set_R if s in rout_sec[r]] for s in set_S}
set_Ra  = {a: list(set().union(*[set_Rs[s] for s in set_Sa[a]])) for a in set_A}

routres = tuplelist([(r,s,u) for r in set_R for s in rout_sec[r] for u in rout_U[r]])
b = {(r,s,u): 0 for (r,s,u) in routres} #route-resource incidence matrix
for r in set_R:
    for i in range(len(rout_sec[r])):
        s = rout_sec[r][i]
        u = round((dept[r] - start_time + sum(rout_time[r][j] for j in range(i))  )//times)
        b[r,s,u] = 1

r_ini = {f: str(f)+'s' for f in set_F}         
            
#-----------------functions --------------------------------------------------

#find best configuration set
def Sub(set_Fi, Q, QH, H, ai): 
    airconfigs = tuplelist([(c,u) for c in set_Ca[ai] for u in set_U])
    set_Ri = [r_ini[f] for f in set_Fi if r_ini[f] in set_Ra[ai]]
    flow = {(p,u): 0 for p in set_Pa[ai] for u in set_U}
    for r in set_Ri:
        p0 = []
        for s in [j for j in rout_sec[r] if j in set_Sa[ai]]:
            for p in set_Ps[s]:
                if p not in p0:
                    p0.extend([p])
                    for u in [j for j in rout_U[r] if j in set_U]:
                        flow[p,u] += b[r,s,u] 
    kp = {(p,u): max(0,flow[p,u]-Q[u][p]) for p in set_Pa[ai] for u in set_U}
    kc = {(c,u): sum(kp[p,u] for p in set_Pc[c]) for (c,u) in airconfigs}
    
    m = Model("benchmark")
    Z = m.addVars(airconfigs, name = "Z", vtype = GRB.BINARY) 
    m.setObjective(quicksum(kc[c,u]*Z[c,u] for (c,u) in airconfigs), GRB.MINIMIZE) #beta[set_Ac[c]]
    m.addConstrs((sum(Z[c,u] for c in set_Ca[ai]) == 1 for u in set_U), name = "oneconfig")
    m.addConstr((quicksum(Z[c,u] * (hours[c] + QH[ai,u]) for c in set_Ca[ai] for u in set_U) <= H), name = "budget")
    m.optimize()
    ob = m.getObjective()
    obj = ob.getValue()
    z = m.getAttr('x', Z)
    
    Dj = beta[ai] * obj 
    return Dj

def MeasureAndAdd(X_new, Y_opt, k, ai):  
    for j in X_new:
        C_new = cost[ai] * j
        D_new = Sub(set_Fi[0], Qi[0], QHi[0], j, ai) # determine best configuration 
        print('Measure' + str(D_new + C_new))
        
    #acceptance criteria
        if D_new + C_new - Y_opt <= lambda1:
            X.append(j)
            C.append(C_new)
            D_all.append([D_new])
            D.append(D_new)
    return X, D, C, D_all
 
def ReAssess(j, num_scen, ai):    
    if j == []:
        print([len(D_all[i]) for i in range(len(D))])
        j = np.random.choice([i for i in range(len(D)) if len(D_all[i]) < s_max])  
     
    D_new = []
    for i in range(len(D_all[j]), len(D_all[j]) + num_scen):
        Di = Sub(set_Fi[i], Qi[i], QHi[i], X[j], ai) # determine best configuration  
        D_new.append(Di)
    D_all[j].extend(D_new)
    D[j] = sum(D_all[j])/ len(D_all[j]) 
    print('Reassess' + str(D[j] + C[j]))
    return X, D, C, D_all    
    
#------------------------------- MASTER CODE ----------------------------------

# Initialization
beta0 = 200
X_ALL, C_ALL, Y_ALL, D_ALL, x_opt, y_opt, X_opt_all, Y_opt_all =  {}, {}, {}, {}, {}, {}, {a: [] for a in set_A}, {a: [] for a in set_A} 
lambda1, s_max = 2000, 200
#betain = [  85.35967506,   26.17054151,  220.32046907,  133.11007319,   85.2522958, 111.69380394,   91.95738291,    1.0, 32.66371047, 1677.55262985, 810.94708679,   10.51393241,   99.0450031,    87.7338237,   734.35150912]
beta = {a: beta0 for a in set_Aup}
X_min = {a: airspaceDF.loc[a,'MinCap'] for a in set_AS} #minimum capacity for each airspace
X_max = {a: airspaceDF.loc[a,'MaxCap'] for a in set_AS} #maximum capacity for each airspace

set_Fi = pickle.load(open( "set_Fi1.p", "rb" ) )
QW = pickle.load(open( "QWi1.p", "rb" ) )
QE = pickle.load(open( "QEi1.p", "rb" ) )
Qi = {i: [{p: Q[p] for p in set_P} for u in set_U] for i in range(len(QW))} 
QHi = {i: {(a,u): 0 for a in set_A for u in set_U} for i in range(len(QW))} 
for i in range(1, len(QW)):
    for a,si,ui in QW[i].keys():
        for u in range(ui, min(42, ui + dur_QW[a])):
            for p in [j for j in set_P if si in set_Sp[j]]:
                Qi[i][u][p] = mt.floor(Q[p] * QW[i][a,si,ui])
    for (a,ui) in QE[i].keys():
        for u in range(ui, min(42, ui + dur_QE[a])):
            QHi[i][a,u] = 1
                   
#1 Start AOS procedure
for ai in set_Aup:
    print(ai)
    frange = X_max[ai] - X_min[ai] + 1 
    if frange == 1:
        x_opt[ai] = X_max[ai]
    else:
        X, C, D, D_all,  Y_opt, m = [], [], [], [],  1000000, 1
        max_iter = frange * 50
     
##1A Evaluate new candidate
        for k in range(1, max_iter):
            if k == mt.floor(m**1.5): 
                X_new = mt.floor(X_min[ai] + lhs(1) * (X_max[ai] +1 - X_min[ai]))
                if X_new not in X:
                    X, D, C, D_all = MeasureAndAdd([X_new], Y_opt, k, ai)
        
                #additional resampling
                num_scenARE = mt.ceil(m**0.5) 
                for j in D_all:
                    if len(j) < num_scenARE:
                        X, D, C, D_all = ReAssess(D_all.index(j), num_scenARE-len(j), ai)
        
                #rejection criteria 
                j = 0
                while j < len(X):
                    if len(D) > 1 and D[j] + C[j] - Y_opt > lambda1/(m**0.5):
                        del X[j], D[j], C[j], D_all[j] 
                    else:
                        j += 1     
                m += 1  
                
#1B Reassess exisiting pool
            else:
                X, D, C, D_all = ReAssess([], 1, ai)
            
#2 Stopping criteria and print optimum budget
            Y = [sum(x) for x in zip(D, C)]
            Y_opt, X_opt = min(Y), X[Y.index(min(Y))]
            print(str(k) + ": " + str(Y_opt) + "  " + str(X[Y.index(Y_opt)]))
            print([len(x) for x in D_all])
            X_opt_all[ai].append(X_opt)
            Y_opt_all[ai].append(Y_opt)
            if all(len(x) > 195 for x in D_all):
                print("break4 in iteration: " + str(k))
                break
            elif (m > frange-1) and len(D) == 1: 
                print("break3 in iteration: " + str(k))
                break  
        D_ALL[ai] = D
        C_ALL[ai] = C
        X_ALL[ai] = X
        Y_ALL[ai] = [sum(x) for x in zip(D, C)]
        x_opt[ai] = X_opt
        y_opt[ai] = Y_opt

print("cost " + str(beta0) )
print("X_opt_all: " + str(X_opt_all)) 
print("Y_opt_all: " + str(Y_opt_all)) 

print("D: " + str(D_ALL)) 
print("X: " + str(X_ALL)) 
print("C: " + str(C_ALL)) 
print("Y: " + str(Y_ALL)) 
print("X_opt: " + str(x_opt)) 
print("Y_opt: " + str(y_opt)) 

