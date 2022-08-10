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
routesDF2 = pd.read_csv("./XX_routes2.csv", sep=";")
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
set_Pc = {c: [configDF.iloc[configDF.index.get_loc(c),2+j] for j in range(configDF.loc[c,'#collsec'])] for c in set_C} #lists of collapsed sectors for each configuration c
set_F = flightDF.index.tolist() #flights 
set_Rall = rcostDF.index[rcostDF['r_type'] <= 1].tolist() 
set_Rf = pickle.load(open( "set_Rf1.p", "rb" ) )  
Ffixed = [f for f in set_F if len(set_Rf[f]) == 1] 

set_R = []
for f in set_F:
    set_R.extend(set_Rf[f])
airconfig = tuplelist([(c,u) for c in set_C for u in set_U]) #airspace-configuration combinations
resources = tuplelist([(s,u) for s in set_S for u in set_U]) #sector-time combinations

#parameters
hours = {c: configDF.loc[c,'#collsec'] for c in set_C}
cost = {a: airspaceDF.loc[a,'Cost']/2 for a in set_A} #cost per sector-hour for airspace and configuration
Q = {p: mt.ceil(collapDF.loc[p,'Cap']/2) for p in set_P} #capacity for each collapsed sector
dur_QE = {a: airspaceDF.loc[a,'DurE'] for a in set_A}
dur_QW = {a: airspaceDF.loc[a,'DurW'] for a in set_A }
delay = {r: 0 for r in set_Rall} #delay for each route
d = {r: rcostDF.loc[r, 'd'] for r in set_Rall} #displacement cost
e1 = {r: rcostDF.loc[r, 'e1'] for r in set_Rall} #emission cost
co2 = {r: rcostDF.loc[r, 'CO2'] for r in set_Rall} #emission 
nox = {r: rcostDF.loc[r, 'NOx'] for r in set_Rall} #emission 
NM = {r: rcostDF.loc[r, 'routNM'] for r in set_Rall} #emission cost
bm = {f: flightDF.loc[f,'bm'] for f in set_F}

rout_time, rout_sec, dept = {}, {}, {}
dept[routesDF2.iloc[0,3]] = routesDF2.iloc[0,5] 
rout_time[routesDF2.iloc[0,3]] = [routesDF2.iloc[0,6]/60]
rout_sec[routesDF2.iloc[0,3]] = [routesDF2.iloc[0,4]]
for j in range(1, 369935):
    if routesDF2.iloc[j,3] != routesDF2.iloc[j-1,3]:
        dept[routesDF2.iloc[j,3]] = routesDF2.iloc[j,5]
        rout_time[routesDF2.iloc[j,3]] = []
        rout_sec[routesDF2.iloc[j,3]] = []
    rout_time[routesDF2.iloc[j,3]].extend([routesDF2.iloc[j,6]/60])
    rout_sec[routesDF2.iloc[j,3]].extend([routesDF2.iloc[j,4]]) 
dept[routesDF.iloc[0,3]] = routesDF.iloc[0,5] 
rout_time[routesDF.iloc[0,3]] = [routesDF.iloc[0,6]/60]
rout_sec[routesDF.iloc[0,3]] = [routesDF.iloc[0,4]]
for j in range(1, 886255):
    if routesDF.iloc[j,3] != routesDF.iloc[j-1,3]:
        dept[routesDF.iloc[j,3]] = routesDF.iloc[j,5]
        rout_time[routesDF.iloc[j,3]] = []
        rout_sec[routesDF.iloc[j,3]] = []
    rout_time[routesDF.iloc[j,3]].extend([routesDF.iloc[j,6]/60])
    rout_sec[routesDF.iloc[j,3]].extend([routesDF.iloc[j,4]])    
rout_U = {r: [j for j in range(int(dept[r]//times), int((dept[r]+ sum(rout_time[r]))//times))] for r in set_Rall}

# FURTHER DELAY OPTIONS AND DUMMY ROUTES      
delays = [15, 30, 60]
set_delay = {i: [str(f)+"-d" + str(i) for f in set_F] for i in delays}
Rdelay = {f: [str(f)+"-d" + str(i)] for i in delays}
Rrerout = {f: [r for r in set_Rf[f] if r not in Rdelay[f]]}
for f in set_F:
    for i in delays:
        delay[str(f)+"-d" + str(i)] = i
        d[str(f)+"-d" + str(i)] = flightDF.loc[f, 'delay'+str(i)] 
        e1[str(f)+"-d" + str(i)] = flightDF.loc[f, 'delay'+str(i)] 
        co2[str(f)+"-d" + str(i)] = 0 
        nox[str(f)+"-d" + str(i)] = 0 
        dept[str(f)+"-d" + str(i)] = dept[set_Rf[f][0]]
        rout_sec[str(f)+"-d" + str(i)] = rout_sec[set_Rf[f][0]]
        rout_time[str(f)+"-d" + str(i)] = rout_time[set_Rf[f][0]]  
        rout_U[str(f)+"-d" + str(i)] = [j for j in range(int((dept[str(f)+"-d" + str(i)] + i)//times), int((dept[str(f)+"-d" + str(i)] + i + sum(rout_time[str(f)+"-d" + str(i)]))//times))]
    rout_sec[str(f)+"-d60"] = []

routres = tuplelist([(r,s,u) for r in set_R for s in rout_sec[r] for u in rout_U[r]])
b = {(r,s,u): 0 for (r,s,u) in routres} #route-resource incidence matrix
for r in set_R:
    for i in range(len(rout_sec[r])):
        s = rout_sec[r][i]
        u = round((dept[r] - start_time + sum(rout_time[r][j] for j in range(i))  )//times)
        b[r,s,u] = 1

r_ini = {}
for f in set_F:  
    dnew = {r: d[r] for r in set_Rf[f]}     
    key1 = min(dnew, key= lambda key: dnew[key])
    r_ini[f] = key1   
    
#-----------------functions for MMKP--------------------------------------------------

#find best configuration set
def Sub(set_Fi, r_opt, Qi, QH, H):
    airconfigs = tuplelist([(c,u) for a in set_A1 for c in set_Ca[a] for u in set_U])
    set_Ri = [r_opt[f] for f in r_opt.keys()]
    for f in [j for j in set_Fi if j not in r_opt.keys()]:
        set_Ri.extend([r_ini[f]])  
    flow = {(p,u): 0 for p in set_P for u in set_U}
    for r in set_Ri:
        p0 = []
        for s in rout_sec[r]:
            for p in set_Ps[s]:
                if p not in p0:
                    p0.extend([p])
                    for u in [j for j in rout_U[r] if j in set_U]:
                        flow[p,u] += b[r,s,u] 
    kp = {(p,u): max(0,flow[p,u]-Qi[u][p]) for p in set_P for u in set_U}
    kc = {(c,u): sum(kp[p,u] for p in set_Pc[c]) for (c,u) in airconfigs}

    c_opt = {}
    m = Model("benchmark")
 #   Y = m.addVars(set_A, name = "Y", lb = 0, vtype = GRB.INTEGER)
    Z = m.addVars(airconfig, name = "Z", vtype = GRB.BINARY) 
    m.setObjective(quicksum(kc[c,u]*Z[c,u] for (c,u) in airconfigs), GRB.MINIMIZE)  #employment
    m.addConstrs((sum(Z[c,u] for c in set_Ca[a]) == 1 for a in set_A1 for u in set_U), name = "oneconfig")
 #   m.addConstrs((sum(Y[a] for a in set_Ai[i]) <= H["cb"+ str(i)] for i in set_I), name = "totalcb")
 #   m.addConstrs((quicksum(Z[c,u] * (hours[c] + QH[a,u]) for c in set_Ca[a] for u in set_U) <= H[a] + Y[a] for a in set_Acb), name = "budget")
    m.addConstrs((quicksum(Z[c,u] * (hours[c] + QH[a,u]) for c in set_Ca[a] for u in set_U) <= H[a] for a in set_A1), name = "budget")
    m.optimize()
    z = m.getAttr('x', Z)
    for a in set_A1:
        for u in set_U:
            for c in set_Ca[a]:
                if z[c,u] > 0:
                    c_opt[a,u] = c
    
    cols = tuplelist([(p,u) for a in set_A1 for u in set_U for p in set_Pc[c_opt[a,u]]]) 
    flow, routcols, bp0 = {(p,u): 0 for (p,u) in cols}, {r: [] for r in set_R} , {}
    r_all = [r_opt[f] for f in r_opt.keys()]    
    for r in set_R:
        p0 = []
        for s in rout_sec[r]:
            for p in set_Ps[s]:
                if p not in p0:
                    p0.extend([p])
                    for u in [j for j in rout_U[r] if j in set_U]:
                        bp0[r,p,u] = b[r,s,u] 
                        if (p,u) in cols:
                            routcols[r].append((p,u))
                            if r in r_all:
                                flow[p,u] += b[r,s,u] 
    routcol = {r: tuplelist(routcols[r]) for r in set_R}
    return c_opt, cols, bp0, routcol, flow

def Simulation(f, flow, kp, Qi, bp0, routcol):
    r_feas = []
    for r in set_Rf[f]: 
        if r in set_delay[60]: 
            r_feas.extend([r])
        elif all(flow[p,u] + bp0[r,p,u] <= Qi[u][p] + kp[p,u] for (p,u) in routcol[r]):
            r_feas.extend([r])
    d_feas = {r: d[r] for r in r_feas}
    unc = np.random.normal(1,bm[f]*0.1,1)[0]
    for r in [j for j in r_feas if j in Rdelay]:
        d_feas[r] = d[r] * unc 
    ri = min(d_feas, key= lambda key: d_feas[key])
    return ri              
    
#------------------------------- MASTER CODE ----------------------------------

#0 Load uncertainties and capacity budget
set_A1 = [*set_Aup, *set_Alow, *set_AP]
set_Fi = pickle.load(open( "set_Fi2.p", "rb" ) )
QW = pickle.load(open( "QWi2.p", "rb" ) )
QE = pickle.load(open( "QEi2.p", "rb" ) )
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

x = {a: airspaceDF.loc[a,'Baseline'] for a in set_A}
#x = {a: airspaceDF.loc[a,'cap200'] for a in set_A}
#x = {a: airspaceDF.loc[a,'cap300'] for a in set_A}

#capacity sharing
#set_I = list(set(airspaceDF['Country'].tolist())) #ANSPs for cross-ACC
#set_Ai = {i: airspaceDF.index[airspaceDF['Country'] ==i].tolist() for i in set_I} #ANSPs
#a1 = [*set_Alow, *set_AP]
#set_I = list(set(airspaceDF['Alliance'].tolist())) #Alliances for cross-border
#set_Ai = {i: airspaceDF.index[airspaceDF['Alliance'] ==i].tolist() for i in set_I}
#a1 = airspaceDF.index[airspaceDF['Alliance'] == 'na'].tolist()
#set_I.remove('na')
#for i in set_I:
#    x['cb' + str(i)] = 0
#    for a in set_Ai[i]:
#        k = round(0.1*x[a])
#        x[a] += -k
#        x['cb' + str(i)] += k
   
#1 Start simulation to determine routing
D, E1, E2, CO2, NOX, n30, n20, n10, nrr, ddelay, drr, e1rr, excess, Hact = [], [], [], [], [], [], [], [], [], [], [], [], [], {a: [] for a in set_Aup}

sc = [i for i in range(10)]
np.random.seed(123)

for i in sc:
    r_opt, m, k = {}, 6, 0
    Fdept = {f: dept[r_ini[f]] for f in set_Fi[i]}
    ci, cols, bp0, routcol, flow = Sub(set_Fi[i], r_opt, Qi[k], QHi[k], x)
    for f in sorted(Fdept, key=Fdept.get):
        print(len(r_opt))
        #make weather uncertainty appear dynamically over time (rest known at beginning of day)
        if dept[r_ini[f]]//times == m: 
            m += 6
            k = np.random.choice([j for j in range(100)])
            ci, cols, bp0, routcol, flow = Sub(set_Fi[i], r_opt, Qi[k], QHi[k], x) 
        #determine  trajectory
        kp = {(p,u): max(0,flow[p,u]-Qi[k][u][p]) for (p,u) in cols}
        if all(flow[p,u] + bp0[r_ini[f],p,u] <= kp[p,u] + Qi[k][u][p] for (p,u) in routcol[r_ini[f]]):
            r_opt[f] = r_ini[f]
        else: 
            r_opt[f] = Simulation(f, flow, kp, Qi[k], bp0, routcol) 
        for (p,u) in routcol[r_opt[f]]:
                flow[p,u] += bp0[r_opt[f],p,u]
    #report results
    Dsum = sum(d[r_opt[f]] for f in set_Fi[i])
    Drr = sum(d[r_opt[f]] for f in set_Fi[i] if r_opt[f] not in set_delay[15] if r_opt[f] not in set_delay[30] if r_opt[f] not in set_delay[60])
    Ddummy = sum(d[r_opt[f]] for f in set_Fi[i] if r_opt[f] in set_delay[60])
    D.append(Dsum) 
    E1.append(sum(e1[r_opt[f]] for f in set_Fi[i]) )
    CO2.append(sum(co2[r_opt[f]] for f in set_Fi[i]) )
    NOX.append(sum(nox[r_opt[f]] for f in set_Fi[i]) )
    e1rr.append(sum(e1[r_opt[f]] - d[r_opt[f]] for f in set_Fi[i]))
    n30.append(len([f for f in set_Fi[i] if r_opt[f] in set_delay[60]]))
    n20.append(len([f for f in set_Fi[i] if r_opt[f] in set_delay[30]]))
    n10.append(len([f for f in set_Fi[i] if r_opt[f] in set_delay[15]]))
    nrr.append(len([f for f in set_Fi[i] if r_opt[f] not in set_delay[15] if r_opt[f] not in set_delay[30] if r_opt[f] not in set_delay[60] if d[r_opt[f]] != 0]))
    ddelay.append(Dsum - Drr - Ddummy) 
    drr.append(Drr)
    for a in set_Aup:
        Hact[a].append(sum(hours[ci[a,u]] for u in set_U))   

print('test simulation')
print("Hact: " + str(Hact))
print('Kosten' + str(D))
print('ddelay'+ str(ddelay))
print('drr'+ str(drr))
print('n60'+ str(n30))
print('n30'+ str(n20))
print('n15'+ str(n10))
print('nrr'+ str(nrr))
print('Emissionen1'+ str(E1))
print('CO2'+ str(CO2))
print('NOX'+ str(NOX))
print('e1rr'+ str(e1rr))