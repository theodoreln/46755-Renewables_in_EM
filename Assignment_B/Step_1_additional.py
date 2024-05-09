########################
""" Relevant modules """
########################


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import copy
import random
import gurobipy as gp
import os
GRB = gp.GRB
from Data import in_sample

##################################################
""" Offering strategy under a one-price scheme """
##################################################

def Offering_two_zeta(in_sample, coeff_1, coeff_2) :
    #Number of units to take into account (based on data)
    # Number of scenarios
    n_scen = len(in_sample)
    # Number of hour
    n_hour = 24
    
    #Optimization part
    # Create the model
    model = gp.Model()
    #Initialize the decision variables
    # Quantity offered by the wind farm in hour t
    var_qu_off = model.addVars(n_hour, lb=-gp.GRB.INFINITY, ub=gp.GRB.INFINITY, vtype=GRB.CONTINUOUS, name='quantity_offered')
    # Difference between the forecast value and the offered value in hour t for scenario w
    var_qu_diff = model.addVars(n_hour, n_scen, lb=-gp.GRB.INFINITY, ub=gp.GRB.INFINITY, vtype=GRB.CONTINUOUS, name='quantity_difference')
    
    # Add the objective function to the model, sum on every t, separation of the conventional generation units and the wind farms
    model.setObjective(gp.quicksum(gp.quicksum((1/n_scen)*(in_sample['DA_price'][w][t]*var_qu_off[t] + in_sample['Binary_var'][w][t]*coeff_1*in_sample['DA_price'][w][t]*var_qu_diff[t,w]
                                                           + (1-in_sample['Binary_var'][w][t])*coeff_2*in_sample['DA_price'][w][t]*var_qu_diff[t,w]) for w in range(n_scen)) for t in range(n_hour)), GRB.MAXIMIZE)
    
    #Add constraints to the model
    for t in range(n_hour):
        # Quantity offered limited to the maximum power of the wind turbine
        model.addConstr(var_qu_off[t] <= 200)
        model.addConstr(var_qu_off[t] >= 0)
        # Definition of the difference quantity variable
        for w in range(n_scen):
            model.addConstr(var_qu_diff[t,w] == in_sample['DA_forecast'][w][t] - var_qu_off[t])
    
    #Solve the problem
    model.optimize()
    
    #Get the optimal values
    if model.status == GRB.OPTIMAL:
        
        # Create a list to store the optimal values of the variables
        optimal_qu_off = [round(var_qu_off[t].X,2) for t in range(n_hour)]
        # Value of the optimal objective
        optimal_obj = model.ObjVal
    
    else:
        print("Optimization did not converge to an optimal solution.")
    
    return(optimal_qu_off, optimal_obj)


######################################################
""" Function to iterate on the value of expected Z """
######################################################

def Iterate_Z(in_sample) :
    # Creation of the variables
    exp_z = [0]*251
    decision_0 = [0]*251
    decision_1 = [0]*251
    decision_2 = [0]*251
    
    sample = in_sample.copy(deep=True)
    
    # We only look at hour 1
    
    for i in range(250) :
        sample['Binary_var'][i][0] = 0
    
    for i in range(251) :
        print(i)
        
        if i != 0 :
            sample['Binary_var'][i-1][0] = 1
            
        optimal_qu_off_0, _ = Offering_two_zeta(sample, 0.9, 1.2)
        optimal_qu_off_1, _ = Offering_two_zeta(sample, 0.9, 1.1)
        optimal_qu_off_2, _ = Offering_two_zeta(sample, 0.9, 1.3)
        
        exp_z[i] = i/250
        decision_0[i] = optimal_qu_off_0[0]
        decision_1[i] = optimal_qu_off_1[0]
        decision_2[i] = optimal_qu_off_2[0]
        
    plt.figure(figsize = (10, 7))
    plt.rcParams["font.size"] = 12
        
    plt.plot(exp_z, decision_0, 'b', label="1.2 lambda")
    plt.plot(exp_z, decision_1, 'r', label="1.1 lambda")
    plt.plot(exp_z, decision_2, 'g', label="1.3 lambda")
    plt.xlabel('Expected value of Z')
    plt.ylabel('Offer decision [$/MWh]')
    plt.legend()
    plt.show()
        
Iterate_Z(in_sample)































