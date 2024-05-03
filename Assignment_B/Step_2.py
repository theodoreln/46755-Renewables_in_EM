########################
""" Relevant modules """
########################

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import pandas as pd
import copy
import random
import gurobipy as gp
import os
GRB = gp.GRB
from Data import sto_anc_scenarios

###########################
""" Scenarios selection """
###########################

#Selecting 50 random scenarios for 'in sample' scenarios case, and creating a dataframe with them
random.seed(123)
sto_anc_in_sample = random.sample(sto_anc_scenarios,50)


# List of remaining scenarios
sto_anc_out_of_sample = [scenario for scenario in sto_anc_scenarios if scenario not in sto_anc_in_sample]

########################################
""" Offering strategy under P90 rule """
########################################

def P90_ALSO_X(in_sample, epsilon) :
    #Number of units to take into account (based on data)
    # Number of scenarios
    n_scen = len(in_sample)
    # Number of minute
    n_min = 60
    
    #Optimization part
    # Create the model
    model = gp.Model()
    #Initialize the decision variables
    # Quantity offered by the ancillary services provider 
    var_qu_off = model.addVar(lb=0, ub=gp.GRB.INFINITY, vtype=GRB.CONTINUOUS, name='quantity_offered')
    # Binary variable, one per minute, per scenario
    var_bin = model.addVars(n_scen, n_min, vtype=GRB.BINARY, name='binary_variable')
    
    # Add the objective function to the model, sum on every t, separation of the conventional generation units and the wind farms
    model.setObjective(var_qu_off, GRB.MAXIMIZE)
    
    #Add constraints to the model
    for s in range(n_scen) :
        for m in range(n_min) :
            model.addConstr(var_qu_off - in_sample[s][m] <= var_bin[s,m]*100000)
    model.addConstr(gp.quicksum(gp.quicksum(var_bin[s,m] for m in range(n_min)) for s in range(n_scen)) <= epsilon*n_scen*n_min)
    
    #Solve the problem
    model.optimize()
    
    #Get the optimal values
    if model.status == GRB.OPTIMAL:
        
        # Create a list to store the optimal values of the variables
        optimal_qu_off = round(var_qu_off.X,2)
    
    else:
        print("Optimization did not converge to an optimal solution.")
    
    return(optimal_qu_off)


def P90_CVAR(in_sample,epsilon) :
    #Number of units to take into account (based on data)
    # Number of scenarios
    n_scen = len(in_sample)
    # Number of minute
    n_min = 60
    
    #Optimization part
    # Create the model
    model = gp.Model()
    #Initialize the decision variables
    # Quantity offered by the ancillary services provider 
    var_qu_off = model.addVar(lb=0, ub=gp.GRB.INFINITY, vtype=GRB.CONTINUOUS, name='quantity_offered')
    # Beta, negative
    var_beta = model.addVar(lb=-gp.GRB.INFINITY, ub=0, vtype=GRB.CONTINUOUS, name='beta')
    # Zeta variable
    var_zeta = model.addVars(n_scen, n_min, lb=-gp.GRB.INFINITY, ub=gp.GRB.INFINITY, vtype=GRB.CONTINUOUS, name='zeta')
    
    # Add the objective function to the model, sum on every t, separation of the conventional generation units and the wind farms
    model.setObjective(var_qu_off, GRB.MAXIMIZE)
    
    #Add constraints to the model
    for s in range(n_scen) :
        for m in range(n_min) :
            model.addConstr(var_qu_off - in_sample[s][m] <= var_zeta[s,m])
            model.addConstr(var_beta <= var_zeta[s,m])
    model.addConstr((1/(n_scen*n_min))*gp.quicksum(gp.quicksum(var_zeta[s,m] for m in range(n_min)) for s in range(n_scen)) <= (1-epsilon)*var_beta)
    
    #Solve the problem
    model.optimize()
    
    #Get the optimal values
    if model.status == GRB.OPTIMAL:
        
        # Create a list to store the optimal values of the variables
        optimal_qu_off = round(var_qu_off.X,2)
    
    else:
        print("Optimization did not converge to an optimal solution.")
    
    return(optimal_qu_off)


optimal_qu_off_ALSO_X = P90_ALSO_X(sto_anc_in_sample, 0.1)
optimal_qu_off_CVAR = P90_CVAR(sto_anc_in_sample, 0.1)


#######################################################
""" Verifying P90 rule with out_of_sample scenarios """
#######################################################

def P90_verify(optimal_qu_off, out_of_sample, epsilon) :
    # To count the number of time it's not fullfilling the condition
    def_count = 0
    # Number of scenarios
    n_scen = len(out_of_sample)
    # Number of minute
    n_min = 60
    
    # Stock when there is a power shortage, for a given minute, for a given scenarios
    shortage = []
    
    for s in range(n_scen) :
        short = [0]*n_min
        for m in range(n_min) :
            if optimal_qu_off > out_of_sample[s][m] :
                def_count += 1
                short[m] = round(optimal_qu_off - out_of_sample[s][m],2)
        shortage.append(short.copy())
                
    total_pos = epsilon*n_scen*n_min
    if def_count > total_pos :
        print(f"P90 rule is not fullfilled for {optimal_qu_off} kW offered")
    else :
        print(f"P90 is fullfilled for {optimal_qu_off} kW offered")
    
    return(shortage)
    
shortage = P90_verify(optimal_qu_off_ALSO_X, sto_anc_out_of_sample, 0.1)


###########################################################
""" Plotting the CFD graph for a selection of scenarios """
###########################################################

def Plotting_CFD(sample, qu_off) :
    # Number of scenarios
    n_scen = len(sample)
    # Number of minute
    n_min = 60
    
    # Limits of power 
    p_min = 0
    p_max = 500 
    p_step = 0.01
    n_pvalue = int((p_max-p_min)/p_step+1)
    
    # Transformation of the scenarios in probabilities
    data = np.zeros((60,n_pvalue))
    for m in range(n_min) :
        for p in range(n_pvalue) :
            power = p*p_step
            prob = 0
            for s in range(n_scen) :
                if power <= sample[s][m] :
                    prob += 1
            prob = prob/n_scen
            data[m,p] = prob

    # Create a custom colormap from purple to yellow
    cmap_colors = [(1, 1, 0), (0.5, 0, 0.5)]  # Purple to yellow
    custom_cmap = LinearSegmentedColormap.from_list("custom_cmap", cmap_colors)
    
    # Plotting
    plt.figure(figsize=(10, 6))
    plt.imshow(data.T, aspect='auto', cmap=custom_cmap, extent=[0, n_min, 0, 500], origin='lower')
    plt.colorbar()
    
    # Add a red horizontal bar at quantity offered
    plt.axhline(y=qu_off, color='red', linestyle='-', linewidth=2, label='Quantity offered')
    plt.legend()
    plt.xlabel('Minute in Hour')
    plt.ylabel('Power (kW)')
    plt.title('Probability distribution of the FCR-D Up service')
    plt.show()


Plotting_CFD(sto_anc_out_of_sample, optimal_qu_off_ALSO_X)




























