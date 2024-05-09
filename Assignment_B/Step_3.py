########################
""" Relevant modules """
########################


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import copy
import random
import gurobipy as gp
GRB = gp.GRB
from Data import in_sample


##################################################
""" Offering strategy under a one-price scheme """
##################################################

def Offering_one_price_risk(in_sample, beta) :
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
    # Auxilliary variables for the risk selection
    var_zeta = model.addVar(lb=-gp.GRB.INFINITY, ub=gp.GRB.INFINITY, vtype=GRB.CONTINUOUS, name='zeta')
    var_eta = model.addVars(n_scen, lb=-gp.GRB.INFINITY, ub=gp.GRB.INFINITY, vtype=GRB.CONTINUOUS, name='eta')
    
    # Add the objective function to the model, sum on every t, separation of the conventional generation units and the wind farms
    model.setObjective((1-beta)*(gp.quicksum(gp.quicksum((1/n_scen)*(in_sample['DA_price'][w][t]*var_qu_off[t] + in_sample['Binary_var'][w][t]*0.9*in_sample['DA_price'][w][t]*var_qu_diff[t,w]
                                                           + (1-in_sample['Binary_var'][w][t])*1.2*in_sample['DA_price'][w][t]*var_qu_diff[t,w]) for w in range(n_scen)) for t in range(n_hour)))
                       + beta*(var_zeta - 1/(1-0.90)*gp.quicksum((1/n_scen)*var_eta[w] for w in range(n_scen))), GRB.MAXIMIZE)
    
    #Add constraints to the model
    for t in range(n_hour):
        # Quantity offered limited to the maximum power of the wind turbine
        model.addConstr(var_qu_off[t] <= 200)
        model.addConstr(var_qu_off[t] >= 0)
        # Definition of the difference quantity variable
        for w in range(n_scen):
            model.addConstr(var_qu_diff[t,w] == in_sample['DA_forecast'][w][t] - var_qu_off[t])
            
    # Constraints for the risk selection 
    for w in range(n_scen):
        model.addConstr(var_eta[w] >= 0)
        model.addConstr(- gp.quicksum(in_sample['DA_price'][w][t]*var_qu_off[t] + in_sample['Binary_var'][w][t]*0.9*in_sample['DA_price'][w][t]*var_qu_diff[t,w]
                                      + (1-in_sample['Binary_var'][w][t])*1.2*in_sample['DA_price'][w][t]*var_qu_diff[t,w] for t in range(n_hour)) + var_zeta - var_eta[w] <= 0)
    
    #Solve the problem
    model.optimize()
    
    #Get the optimal values
    if model.status == GRB.OPTIMAL:
        
        # Create a list to store the optimal values of the variables
        optimal_qu_off = [round(var_qu_off[t].X,2) for t in range(n_hour)]

        # Compute CVAR
        cvar = var_zeta.X - (1/(1-0.9))*(1/n_scen)*sum([var_eta[w].X for w in range(n_scen)])

        # Value of the optimal objective
        optimal_obj = (model.ObjVal - beta*cvar)/(1-beta)
    
    else:
        print("Optimization did not converge to an optimal solution.")
    
    return(optimal_qu_off, optimal_obj, cvar)

##################################################
""" Offering strategy under a two-price scheme """
##################################################

def Offering_two_price_risk(in_sample, beta) :
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
    # Auxilliary variables for the risk selection
    var_zeta = model.addVar(lb=-gp.GRB.INFINITY, ub=gp.GRB.INFINITY, vtype=GRB.CONTINUOUS, name='zeta')
    var_eta = model.addVars(n_scen, lb=-gp.GRB.INFINITY, ub=gp.GRB.INFINITY, vtype=GRB.CONTINUOUS, name='eta')
    
    # Add the objective function to the model, sum on every t, separation of the conventional generation units and the wind farms
    model.setObjective((1-beta)*(gp.quicksum(gp.quicksum((1/n_scen)*(in_sample['DA_price'][w][t]*var_qu_off[t] 
                                                           + in_sample['Binary_var'][w][t]*(0.9*in_sample['DA_price'][w][t]*var_aux_exc[t,w] - in_sample['DA_price'][w][t]*var_aux_def[t,w])
                                                           + (1-in_sample['Binary_var'][w][t])*(in_sample['DA_price'][w][t]*var_aux_exc[t,w] -1.2*in_sample['DA_price'][w][t]*var_aux_def[t,w])) 
                                               for w in range(n_scen)) for t in range(n_hour)))
                       + beta*(var_zeta - 1/(1-0.90)*gp.quicksum((1/n_scen)*var_eta[w] for w in range(n_scen))), GRB.MAXIMIZE)
    
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
    
    # Constraints for the risk selection 
    for w in range(n_scen):
        model.addConstr(var_eta[w] >= 0)
        model.addConstr(- gp.quicksum(in_sample['DA_price'][w][t]*var_qu_off[t] + in_sample['Binary_var'][w][t]*(0.9*in_sample['DA_price'][w][t]*var_aux_exc[t,w] - in_sample['DA_price'][w][t]*var_aux_def[t,w])
                                      + (1-in_sample['Binary_var'][w][t])*(in_sample['DA_price'][w][t]*var_aux_exc[t,w] -1.2*in_sample['DA_price'][w][t]*var_aux_def[t,w]) for t in range(n_hour)) + var_zeta - var_eta[w] <= 0)
    
    #Solve the problem
    model.optimize()
    
    #Get the optimal values
    if model.status == GRB.OPTIMAL:
        
        # Create a list to store the optimal values of the variables
        optimal_qu_off = [round(var_qu_off[t].X,2) for t in range(n_hour)]
        # Compute CVAR
        cvar = var_zeta.X - (1/(1-0.9))*(1/n_scen)*sum([var_eta[w].X for w in range(n_scen)])
        # Value of the optimal objective
        optimal_obj = (model.ObjVal - beta*cvar)/(1-beta)   

    else:
        print("Optimization did not converge to an optimal solution.")
    
    return(optimal_qu_off, optimal_obj, cvar)


######################################
""" Iterating on the value of beta """
######################################

def beta_iteration(in_sample, price_scheme, beta_max, beta_step) :
    # Initialize the lists
    expected_value = []
    cvar = []
    labels = []
    beta = 0
    count = 0
    while beta < beta_max :
        # Condition on which price_scheme
        if price_scheme == 1 :
            _, optimal_obj, cvar_value = Offering_one_price_risk(in_sample, beta)
        elif price_scheme == 2 :
            _, optimal_obj, cvar_value = Offering_two_price_risk(in_sample, beta)
        
        expected_value.append(optimal_obj)
        cvar.append(cvar_value)
        labels.append(round(beta,2))
        beta += beta_step
    
    
    first_label_position = (cvar[0], expected_value[0])  # First label
    middle_label_position = (cvar[len(cvar)//2], expected_value[len(expected_value)//2])  # Middle label
    last_label_position = (cvar[-1], expected_value[-1])  # Last label  
    plt.scatter(cvar, expected_value, color='red', marker='o', edgecolors='black', label='Data Points', zorder=2)

    # Add labels to the dots with beta symbol
    plt.annotate(f'β = {labels[0]}', first_label_position, fontsize=8, ha='right', color='black', xytext=(6, -12), textcoords='offset points')
    plt.annotate(f'β = {labels[len(cvar)//2]}', middle_label_position, fontsize=8, ha='right', color='black', xytext=(-6, 0), textcoords='offset points')
    plt.annotate(f'β = {labels[-1]}', last_label_position, fontsize=8, ha='right', color='black', xytext=(-10, -4), textcoords='offset points')
    
    # Add labels and title
    plt.xlabel('CVaR ($)')
    plt.ylabel('Expected Profit ($)')

    # Connect the dots with a dashed red line
    plt.plot(cvar, expected_value, color='red', linestyle=':', linewidth=0.8, label='Trend Line', zorder=1)

    # Add gridlines
    plt.grid(True, color='#D9E6FF')

    # Customize font style
    plt.rcParams.update({'font.family': 'Arial', 'font.size': 10})

    # Change background color to a lighter shade of blue
    plt.gca().set_facecolor('white')  # Lighter shade of blue

    plt.show()
    
    return(expected_value, cvar)
    
if __name__ == "__main__":
    expected_value, cvar = beta_iteration(in_sample, 1, 1, 0.05)
    expected_value, cvar = beta_iteration(in_sample, 2, 1, 0.05)


























