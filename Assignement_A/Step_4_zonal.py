
from Data import Generators, Demands, Wind_Farms, Transmission, Zones_2, Zones_3
from Step_1 import Single_hour_plot, Commodities
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import gurobipy as gp
import copy
import os
GRB = gp.GRB

############################
""" Optimization problem """
############################

#Parameters of our problem
# Number of hours (but the data are already considered with 24 hours)
n_hour = 24
# Index of the electrolyzer, change first numbers to change the place
index_elec = {0:0, 1:1}
# Hydrogen demand per electrolyser (in tons)
Hydro_demand = 20

# Taking the hour supply and load information and optimizing on 24 hours in zonal mode
def Zonal_optimization(Generators, Wind_Farms, Demands, Zones) :
    #Number of units to take into account (based on data)
    # Number of conventional generation units
    n_gen = len(Generators)
    # Number of wind farms generation units
    n_wf = len(Wind_Farms)
    # Number of demand units
    n_dem = len(Demands)
    #number of nodes of the network
    n_zones=len(Zones)
    
    
    #Optimization part
    # Create the model
    model = gp.Model()
    #Initialize the decision variables, each one of them is now time dependant
    # Capacities provided by conventional generation units
    var_conv_gen = model.addVars(n_gen, n_hour, lb=-gp.GRB.INFINITY, vtype=GRB.CONTINUOUS, name='conv_gen')
    # Capacities provided by wind farms
    var_wf_gen = model.addVars(n_wf, n_hour, lb=-gp.GRB.INFINITY, vtype=GRB.CONTINUOUS, name='wf_gen')
    # Demand capacities fullfilled
    var_dem = model.addVars(n_dem, n_hour, lb=-gp.GRB.INFINITY, vtype=GRB.CONTINUOUS, name='dem')
    # Capacities provided to the electrolyzer
    var_elec = model.addVars(len(index_elec), n_hour, lb=-gp.GRB.INFINITY, vtype=GRB.CONTINUOUS, name='elec')
    # Power trade between zones
    var_trade = model.addVars(n_zones, n_zones, n_hour, lb=-gp.GRB.INFINITY, vtype=GRB.CONTINUOUS, name='trade')
    
    # Add the objective function to the model, sum on every t, separation of the conventional generation units and the wind farms
    model.setObjective(gp.quicksum(gp.quicksum(Demands['Offer price'][d][t]*var_dem[d,t] for d in range(n_dem))-gp.quicksum(Generators['Bid price'][g]*var_conv_gen[g,t] for g in range(n_gen))-gp.quicksum(Wind_Farms['Bid price'][wf][t]*var_wf_gen[wf,t] for wf in range(n_wf)) for t in range(n_hour)), GRB.MAXIMIZE)
    
    #Add constraints to the model
    # Constraints that must be fullfilled for every time step
    for t in range(n_hour) :
        # Constraints of power balance in every zone
        for n in range(1,n_zones+1):
            # Power balance 
            constr_name = f"Power balance hour {t+1} in zone {n}"
            D = Zones[n]["D"] #take the demand information corresponding to each node
            G = Zones[n]["G"] #take the generator information corresponding to each node
            W = Zones[n]["W"] #take the wind farm information corresponding to each node
            L = Zones[n]["L"] #take the line information corresponding to each node
            model.addConstr(gp.quicksum(var_dem[d-1,t] for d in D) 
                - gp.quicksum(var_conv_gen[g-1,t] for g in G) 
                - gp.quicksum(var_wf_gen[wf-1,t] for wf in W) 
                + gp.quicksum(var_trade[n-1,m-1,t] for m in [L[j][0] for j in range(len(L))]) == 0, name=constr_name)
            # No transmission inside a zone
            model.addConstr(var_trade[n-1,n-1,t] == 0)
            # Transmission in one direction is equal the transmission in the other direction
            for m in [L[j][0] for j in range(len(L))] :
                model.addConstr(var_trade[n-1,m-1,t] == -var_trade[m-1,n-1,t])
                
            # Transmission line limits
            for l in L :
                m, atc_neg, atc_pos = l
                model.addConstr(var_trade[n-1,m-1,t] >= -atc_neg)
                model.addConstr(var_trade[n-1,m-1,t] <= atc_pos)
        
        # Constraints on generators
        for g in range(n_gen) :
            # The provided capacity of a conventional unit should not be more than the maximum capacity
            model.addConstr(var_conv_gen[g,t] <= Generators['Capacity'][g])
            model.addConstr(var_conv_gen[g,t] >= 0)
            # Ramp up and ramp down constraints of conventional units depends on the time step, for t=1 take into account initial power
            if t != 0 :
                model.addConstr(var_conv_gen[g,t] - var_conv_gen[g,t-1] <= Generators['Ramp up'][g])
                model.addConstr(var_conv_gen[g,t-1] - var_conv_gen[g,t] <= Generators['Ramp down'][g])
            else :
                model.addConstr(var_conv_gen[g,t] - Generators['Initial power'][g] <= Generators['Ramp up'][g])
                model.addConstr(Generators['Initial power'][g] - var_conv_gen[g,t] <= Generators['Ramp down'][g])
        
        # Constraints on demand
        for d in range(n_dem) :
            # The capacity provided to a demand unit should not be higher than the maximum capacity demand
            model.addConstr(var_dem[d,t] <= Demands['Load'][d][t])
            model.addConstr(var_dem[d,t] >= 0)
            
        # Constraints on wind farms and electrolyzer
        for wf in range(n_wf) :
            # If the wind farm has an electrolyzer, then the capacity provided to the grid is limited by the maximum capacity (which is the wind profile here) minus the capacity provided to the electrolyzer
            if wf in index_elec :
                model.addConstr(var_wf_gen[wf,t] <= Wind_Farms['Capacity'][wf][t] - var_elec[index_elec[wf],t])
            else :
                model.addConstr(var_wf_gen[wf,t] <= Wind_Farms['Capacity'][wf][t])
            model.addConstr(var_wf_gen[wf,t] >= 0)
        for e in range(len(index_elec)) :
            # The capacity of each electrolyzer is limited to 100 MW which corresponds to half the maximum capacity of a wind farm (without wind profil)
            model.addConstr(var_elec[e,t] <= 100)
            model.addConstr(var_elec[e,t] >= 0)
            
    # Constraints on hydrogen production which should be higher than the demand, for each electrolyzer
    for e in range(len(index_elec)) :
        model.addConstr(gp.quicksum(var_elec[e,t] for t in range(n_hour))*0.018 >= Hydro_demand)
     
    #Solve the problem
    model.optimize()
    
    
    #Get the optimal values
    if model.status == GRB.OPTIMAL:
        # Create a list to store the optimal values of the variables
        optimal_conv_gen = [[round(var_conv_gen[g,t].X,2) for g in range(n_gen)] for t in range(n_hour)]
        optimal_wf_gen = [[round(var_wf_gen[wf,t].X,2) for wf in range(n_wf)] for t in range(n_hour)]
        optimal_dem = [[round(var_dem[d,t].X,2) for d in range(n_dem)] for t in range(n_hour)]
        optimal_elec = [[round(var_elec[e,t].X,2) for e in range(2)] for t in range(n_hour)]
        equilibrium_prices = np.zeros((n_zones,n_hour),dtype=np.float64)
        
        # Retrieve the values of the variable var_trade
        optimal_trans = np.zeros((n_zones,n_zones,n_hour))
        for n in range(n_zones) :
            for m in range(n_zones) :
                for t in range(n_hour) :
                    optimal_trans[n,m,t] = round(var_trade[n,m,t].X,2)
        
        # Value of the optimal objective
        optimal_obj = model.ObjVal
        
        """ KKTs output """
        # Write KKT's condition in a text file
        # Define the folder path and the file name
        folder_path = "results/zonal"
        file_name = "KKTs.txt"
        
        # Create the folder if it doesn't exist
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        
        # Combine the folder path and file name
        file_path = os.path.join(folder_path, file_name)
        
        # Open the file in write mode
        with open(file_path, 'w') as file:
            file.write("KKTs for the multiple hour optimization :")
            file.write("\n\n")
            # To help identify to which zones correspond the price
            ind = n_zones
            temps = -1
            for c in model.getConstrs():
                file.write(f"{c.ConstrName}: {c.Pi} : {c.Sense} \n")
                if 'Power balance hour' in c.constrName :
                    remind = ind % n_zones
                    if remind == 0:
                        temps += 1
                    equilibrium_prices[remind,temps] = c.Pi
                    ind += 1
        
    else:
        print("Optimization did not converge to an optimal solution.")
    
    return(optimal_conv_gen, optimal_wf_gen, optimal_dem, optimal_elec, optimal_trans, equilibrium_prices)


##########################################################
""" Written output of transmission decision and prices """
##########################################################

def Zonal_transmission_prices(Zones, optimal_conv_gen, optimal_wf_gen, optimal_dem, optimal_elec, optimal_trans, equilibrium_prices) :
    # Number of zones
    n_zones = len(Zones)
    
    #Write output in a text file
    # Define the folder path and the file name
    folder_path = "results/zonal"
    file_name = "Transmission and prices.txt"
    
    # Create the folder if it doesn't exist
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    
    # Combine the folder path and file name
    file_path = os.path.join(folder_path, file_name)
    
    # Open the file in write mode
    with open(file_path, 'w') as file:
        file.write("Transmission and equilibrium prices for the zonal optimization :")
        file.write("\n\n")
        
        for t in range(n_hour) :
            file.write(f"\n---------- Hour {t+1} ----------")
            file.write("\n\n")
            
            for n in range(n_zones) :
                file.write(f"---------- Zone {n+1} ----------\n")
                file.write(f"Equilibrium price : {equilibrium_prices[n,t]}\n")
                
                D = Zones[n+1]["D"] #take the demand information corresponding to each node
                G = Zones[n+1]["G"] #take the generator information corresponding to each node
                W = Zones[n+1]["W"] #take the wind farm information corresponding to each node
                Gen = sum([optimal_conv_gen[t][g-1] for g in G]) + sum([optimal_wf_gen[t][w-1] for w in W])
                Dem = sum([optimal_dem[t][d-1] for d in D])
                
                file.write(f"Power generation : {Gen}\n")
                file.write(f"Power demand : {Dem}\n")
                
                for m in range(n_zones) :
                    if m != n :
                        file.write(f"Transmission with zone {m+1} : {optimal_trans[n,m,t]}\n")

                        

Zones = Zones_3
optimal_conv_gen, optimal_wf_gen, optimal_dem, optimal_elec, optimal_trans, equilibrium_prices = Zonal_optimization(Generators, Wind_Farms, Demands, Zones)
Zonal_transmission_prices(Zones, optimal_conv_gen, optimal_wf_gen, optimal_dem, optimal_elec, optimal_trans, equilibrium_prices)















