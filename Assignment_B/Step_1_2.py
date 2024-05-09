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

def Offering_one_price(in_sample) :
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
    model.setObjective(gp.quicksum(gp.quicksum((1/n_scen)*(in_sample['DA_price'][w][t]*var_qu_off[t] + in_sample['Binary_var'][w][t]*0.9*in_sample['DA_price'][w][t]*var_qu_diff[t,w]
                                                           + (1-in_sample['Binary_var'][w][t])*1.2*in_sample['DA_price'][w][t]*var_qu_diff[t,w]) for w in range(n_scen)) for t in range(n_hour)), GRB.MAXIMIZE)
    
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

optimal_qu_off_one, optimal_obj_one = Offering_one_price(in_sample)

##################################################
""" Offering strategy under a two-price scheme """
##################################################

def Offering_two_price(in_sample) :
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
    # Auxillary variable for power excess
    var_aux_exc = model.addVars(n_hour, n_scen, lb=-gp.GRB.INFINITY, ub=gp.GRB.INFINITY, vtype=GRB.CONTINUOUS, name='auxillary_excess')
    # Auxillary variable for power deficit
    var_aux_def = model.addVars(n_hour, n_scen, lb=-gp.GRB.INFINITY, ub=gp.GRB.INFINITY, vtype=GRB.CONTINUOUS, name='auxillary_deficit')
    
    # Add the objective function to the model, sum on every t, separation of the conventional generation units and the wind farms
    model.setObjective(gp.quicksum(gp.quicksum((1/n_scen)*(in_sample['DA_price'][w][t]*var_qu_off[t] 
                                                           + in_sample['Binary_var'][w][t]*(0.9*in_sample['DA_price'][w][t]*var_aux_exc[t,w] - in_sample['DA_price'][w][t]*var_aux_def[t,w])
                                                           + (1-in_sample['Binary_var'][w][t])*(in_sample['DA_price'][w][t]*var_aux_exc[t,w] -1.2*in_sample['DA_price'][w][t]*var_aux_def[t,w])) 
                                               for w in range(n_scen)) for t in range(n_hour)), GRB.MAXIMIZE)
    
    #Add constraints to the model
    for t in range(n_hour):
        # Quantity offered limited to the maximum power of the wind turbine
        model.addConstr(var_qu_off[t] <= 200)
        model.addConstr(var_qu_off[t] >= 0)
        # Definition of the difference quantity variable
        for w in range(n_scen):
            model.addConstr(var_qu_diff[t,w] == in_sample['DA_forecast'][w][t] - var_qu_off[t])
            model.addConstr(var_qu_diff[t,w] == var_aux_exc[t,w] - var_aux_def[t,w])
            model.addConstr(var_aux_exc[t,w] >= 0)
            model.addConstr(var_aux_def[t,w] >= 0)
    
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

optimal_qu_off_two, optimal_obj_two = Offering_two_price(in_sample)


##############################################
""" Compute the profits for each scenarios """
##############################################

def Profits_scenarios(in_sample, optimal_qu_off, price_scheme, n_scen) :
    #Number of units to take into account (based on data)
    # Number of hour
    n_hour = 24
    
    # Creating the list of profits
    profit = [0]*n_scen
    
    # Compute the profit for each scenarios
    for w in range(n_scen) :
        profit_w = 0
        for t in range(n_hour) :
            optimal_qu_diff = in_sample['DA_forecast'][w][t] - optimal_qu_off[t]
            # Profit depends on the price scheme
            if price_scheme == 1 :
                profit_w += in_sample['DA_price'][w][t]*optimal_qu_off[t] + in_sample['Binary_var'][w][t]*0.9*in_sample['DA_price'][w][t]*optimal_qu_diff + (1-in_sample['Binary_var'][w][t])*1.2*in_sample['DA_price'][w][t]*optimal_qu_diff
            elif price_scheme == 2 :
                profit_w +=in_sample['DA_price'][w][t]*optimal_qu_off[t] 
                if in_sample['Binary_var'][w][t] == 1 :
                    if optimal_qu_diff >= 0 :
                        profit_w += 0.9*in_sample['DA_price'][w][t]*optimal_qu_diff
                    else :
                        profit_w += in_sample['DA_price'][w][t]*optimal_qu_diff
                else :
                    if optimal_qu_diff >= 0 :
                        profit_w += in_sample['DA_price'][w][t]*optimal_qu_diff
                    else :
                        profit_w += 1.2*in_sample['DA_price'][w][t]*optimal_qu_diff
                    
        # Save the profit for scenario w
        profit[w] = profit_w
    
    return(profit)


def Show_distribution(profit, nb_bins) :
    
    # Create histogram
    plt.hist(profit, bins=nb_bins, color='blue', edgecolor='black')
    
    # Add labels and title
    plt.xlabel('Profit [$]')
    plt.ylabel('Number of occurence')
    #plt.title('Histogram of profits')
    
    # Display the plot
    plt.show()

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
            
        optimal_qu_off_0, _ = Offering_one_price(sample, 0.9, 1.2)
        optimal_qu_off_1, _ = Offering_one_price(sample, 0.9, 1.1)
        optimal_qu_off_2, _ = Offering_one_price(sample, 0.9, 1.3)
        
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

if __name__ == "__main__":
    # Number of scenarios
    n_scen = len(in_sample)
    profit_one = Profits_scenarios(in_sample, optimal_qu_off_one, 1, n_scen)
    profit_two = Profits_scenarios(in_sample, optimal_qu_off_two, 2, n_scen)
    Show_distribution(profit_one, 80)
    Show_distribution(profit_two, 80)

    #Iterate_Z(in_sample)















