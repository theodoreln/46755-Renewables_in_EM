from Data import Generators, Generators_reserve, Demands, Wind_Farms
from Step_1 import Single_hour_plot, Commodities, Single_hour_optimization
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

# Reserve optimization problem on 24 hours
def Reserve_optimization(Generators_reserve, Demands) :
    # Global variables
    # global optimal_conv_gen, optimal_wf_gen, optimal_dem, optimal_elec
    #Number of units to take into account (based on data)
    # Number of conventional generation units
    n_gen = len(Generators_reserve)
    # Number of demand units
    n_dem = len(Demands)
    # Total of demands per hour
    Demands_sum = []
    for t in range(n_hour) :
        sum_hour = 0
        for i in range(len(Demands)) :
            sum_hour += Demands['Load'][i][t]
        Demands_sum.append(sum_hour)
    
    #Optimization part
    # Create the model
    model = gp.Model()
    #Initialize the decision variables, each one of them is now time dependant
    # Capacities provided by conventional generation units
    var_up = model.addVars(n_gen, n_hour, lb=-gp.GRB.INFINITY, vtype=GRB.CONTINUOUS, name='conv_gen')
    # Capacities provided by wind farms
    var_down = model.addVars(n_gen, n_hour, lb=-gp.GRB.INFINITY, vtype=GRB.CONTINUOUS, name='wf_gen')
    
    # Add the objective function to the model, sum on every t, separation of the conventional generation units and the wind farms
    model.setObjective(gp.quicksum(gp.quicksum(Generators_reserve['Up reserve price'][g]*var_up[g,t] for g in range(n_gen)) + gp.quicksum(Generators_reserve['Down reserve price'][g]*var_down[g,t] for g in range(n_gen)) for t in range(n_hour)), GRB.MINIMIZE)
    
    #Add constraints to the model
    # Quantity reserved = Quantity needed for every time step t
    for t in range(n_hour):
        constr_name = f"Reserve up need hour {t+1}"
        model.addConstr(gp.quicksum(var_up[g,t] for g in range(n_gen)) == Demands_sum[t]*0.15, name = constr_name)
        constr_name = f"Reserve down need hour {t+1}"
        model.addConstr(gp.quicksum(var_down[g,t] for g in range(n_gen)) == Demands_sum[t]*0.10, name = constr_name)
    
    # Constraint that must be fullfilled for every time step
    for t in range(n_hour) :
        # Maximum and minimum for reserve + sum should not be higher than max capacity
        for g in range(n_gen) :
            model.addConstr(var_up[g,t] >= 0)
            model.addConstr(var_up[g,t] <= Generators_reserve['Maximum up reserve'][g])
            model.addConstr(var_down[g,t] >= 0)
            model.addConstr(var_down[g,t] <= Generators_reserve['Maximum down reserve'][g])
            model.addConstr(var_up[g,t] + var_down[g,t] <= Generators_reserve['Capacity'][g])
            
    
    #Solve the problem
    model.optimize()
    
    #Get the optimal values
    if model.status == GRB.OPTIMAL:
        # Create a list to store the optimal values of the variables
        optimal_up = [[round(var_up[g,t].X,2) for g in range(n_gen)] for t in range(n_hour)]
        optimal_down = [[round(var_down[g,t].X,2) for g in range(n_gen)] for t in range(n_hour)]
        up_prices = []
        down_prices = []
        # Value of the optimal objective
        optimal_obj = model.ObjVal
        
        """ KKTs output """
        # Write KKT's condition in a text file
        # Define the folder path and the file name
        folder_path = "results/reserve"
        file_name = "KKTs.txt"
        
        # Create the folder if it doesn't exist
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        
        # Combine the folder path and file name
        file_path = os.path.join(folder_path, file_name)
        
        # Open the file in write mode
        with open(file_path, 'w') as file:
            file.write("KKTs for the regulation capacity reservation :")
            file.write("\n\n")
            # Write lines to the file
            for c in model.getConstrs():
                file.write(f"{c.ConstrName}: {c.Pi} : {c.Sense} \n")
                if 'Reserve up' in c.constrName :
                    up_prices.append(c.Pi)
                if 'Reserve down' in c.constrName :
                    down_prices.append(c.Pi)
        
    else:
        print("Optimization did not converge to an optimal solution.")
    
    return(optimal_up, optimal_down, up_prices, down_prices)

optimal_up, optimal_down, up_prices, down_prices = Reserve_optimization(Generators_reserve, Demands)


# Optimization problem for the day-ahead problem
def DA_optimization(Generators, Wind_Farms, Demands, optimal_up, optimal_down) :
    # Global variables
    # global optimal_conv_gen, optimal_wf_gen, optimal_dem, optimal_elec
    #Number of units to take into account (based on data)
    # Number of conventional generation units
    n_gen = len(Generators)
    # Number of wind farms generation units
    n_wf = len(Wind_Farms)
    # Number of demand units
    n_dem = len(Demands)
    
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
    
    # Add the objective function to the model, sum on every t, separation of the conventional generation units and the wind farms
    model.setObjective(gp.quicksum(gp.quicksum(Demands['Offer price'][d][t]*var_dem[d,t] for d in range(n_dem))-gp.quicksum(Generators['Bid price'][g]*var_conv_gen[g,t] for g in range(n_gen))-gp.quicksum(Wind_Farms['Bid price'][wf][t]*var_wf_gen[wf,t] for wf in range(n_wf)) for t in range(n_hour)), GRB.MAXIMIZE)
    
    #Add constraints to the model
    # Quantity supplied = Quantity used for every time step t
    for t in range(n_hour):
        constr_name = f"Power balance hour {t+1}"
        model.addConstr(gp.quicksum(var_dem[d,t] for d in range(n_dem)) - gp.quicksum(var_conv_gen[g,t] for g in range(n_gen)) - gp.quicksum(var_wf_gen[wf,t] for wf in range(n_wf)) == 0, name = constr_name)
    # Constraint that must be fullfilled for every time ste
    for t in range(n_hour) :
        for g in range(n_gen) :
            # The provided capacity of a conventional unit should not be more than the maximum capacity minus the reserved capacity
            model.addConstr(var_conv_gen[g,t] <= Generators['Capacity'][g] - optimal_up[t][g])
            model.addConstr(var_conv_gen[g,t] >= optimal_down[t][g])
            # Ramp up and ramp down constraints of conventional units depends on the time step, for t=1 take into account initial power
            if t != 0 :
                model.addConstr(var_conv_gen[g,t] - var_conv_gen[g,t-1] <= Generators['Ramp up'][g])
                model.addConstr(var_conv_gen[g,t-1] - var_conv_gen[g,t] <= Generators['Ramp down'][g])
            else :
                model.addConstr(var_conv_gen[g,t] - Generators['Initial power'][g] <= Generators['Ramp up'][g])
                model.addConstr(Generators['Initial power'][g] - var_conv_gen[g,t] <= Generators['Ramp down'][g])
        for d in range(n_dem) :
            # The capacity provided to a demand unit should not be higher than the maximum capacity demand
            model.addConstr(var_dem[d,t] <= Demands['Load'][d][t])
            model.addConstr(var_dem[d,t] >= 0)
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
        equilibrium_prices = []
        # Value of the optimal objective
        optimal_obj = model.ObjVal
        
        """ KKTs output """
        # Write KKT's condition in a text file
        # Define the folder path and the file name
        folder_path = "results/multiple_hour"
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
            # Write lines to the file
            for c in model.getConstrs():
                file.write(f"{c.ConstrName}: {c.Pi} : {c.Sense} \n")
                if 'Power balance hour' in c.constrName :
                    equilibrium_prices.append(c.Pi)
        
    else:
        print("Optimization did not converge to an optimal solution.")
    
    return(optimal_conv_gen, optimal_wf_gen, optimal_dem, optimal_elec, equilibrium_prices)


optimal_conv_gen, optimal_wf_gen, optimal_dem, optimal_elec, equilibrium_prices = DA_optimization(Generators, Wind_Farms, Demands, optimal_up, optimal_down)





