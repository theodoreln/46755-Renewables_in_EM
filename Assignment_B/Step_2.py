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
from matplotlib.colors import ListedColormap

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
    
    # Set Gurobi parameter to suppress output
    model.setParam('OutputFlag', 0) 
    
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
    
    # Set Gurobi parameter to suppress output
    model.setParam('OutputFlag', 0)
    
    #Solve the problem
    model.optimize()
    
    #Get the optimal values
    if model.status == GRB.OPTIMAL:
        
        # Create a list to store the optimal values of the variables
        optimal_qu_off = round(var_qu_off.X,2)
    
    else:
        print("Optimization did not converge to an optimal solution.")
    
    return(optimal_qu_off)


# optimal_qu_off_ALSO_X = P90_ALSO_X(sto_anc_in_sample, 0.1)
# optimal_qu_off_CVAR = P90_CVAR(sto_anc_in_sample, 0.1)


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
    # Stock weither (1) or not (0) the quantity verify the P90 rule
    decision = 0
    
    for s in range(n_scen) :
        short = [0]*n_min
        for m in range(n_min) :
            if optimal_qu_off > out_of_sample[s][m] :
                def_count += 1
                short[m] = round(optimal_qu_off - out_of_sample[s][m],2)
        shortage.append(short.copy())
                
    total_pos = epsilon*n_scen*n_min
    if def_count > total_pos :
        decision = 0
        # print(f"P90 rule is not fullfilled for {optimal_qu_off} kW offered")
    else :
        decision = 1
        # print(f"P90 is fullfilled for {optimal_qu_off} kW offered")
    
    return(shortage, def_count, decision)
    
# shortage, def_count, decision = P90_verify(optimal_qu_off_ALSO_X, sto_anc_out_of_sample, 0.1)
# shortage, def_count, decision = P90_verify(optimal_qu_off_CVAR, sto_anc_out_of_sample, 0.1)

def Shortage_plot(in_sample, out_of_sample, epsilon) :
    # Number of minute
    n_min = 60
    # List of minutes
    minutes = list(range(1, n_min+1))
    
    # Create a list to store the average value of shortage for a given minute
    avg_short_in_ALSO_X = [0]*n_min
    avg_short_in_CVAR = [0]*n_min
    avg_short_out_ALSO_X = [0]*n_min
    avg_short_out_CVAR = [0]*n_min
    
    # In sample offering decisions
    optimal_qu_off_ALSO_X = P90_ALSO_X(in_sample, epsilon)
    optimal_qu_off_CVAR = P90_CVAR(in_sample, epsilon)
    
    # Shortages computation
    shortage_in_ALSO_X, def_count_in_ALSO_X, decision_in_ALSO_X = P90_verify(optimal_qu_off_ALSO_X, in_sample, epsilon)
    shortage_in_CVAR, def_count_in_CVAR, decision_in_CVAR = P90_verify(optimal_qu_off_CVAR, in_sample, epsilon)
    shortage_out_ALSO_X, def_count_out_ALSO_X, decision_out_ALSO_X = P90_verify(optimal_qu_off_ALSO_X, out_of_sample, epsilon)
    shortage_out_CVAR, def_count_out_CVAR, decision_out_CVAR = P90_verify(optimal_qu_off_CVAR, out_of_sample, epsilon)
    
    # Average shortage per minute
    for m in range(n_min) :
        for s in range(len(in_sample)) :
            avg_short_in_ALSO_X[m] += shortage_in_ALSO_X[s][m]
            avg_short_in_CVAR[m] += shortage_in_CVAR[s][m]
        avg_short_in_ALSO_X[m] = avg_short_in_ALSO_X[m]/len(in_sample)
        avg_short_in_CVAR[m] = avg_short_in_CVAR[m]/len(in_sample)
    for m in range(n_min) :
        for s in range(len(out_of_sample)) :
            avg_short_out_ALSO_X[m] += shortage_out_ALSO_X[s][m]
            avg_short_out_CVAR[m] += shortage_out_CVAR[s][m]
        avg_short_out_ALSO_X[m] = avg_short_out_ALSO_X[m]/len(out_of_sample)
        avg_short_out_CVAR[m] = avg_short_out_CVAR[m]/len(out_of_sample)
    # Standard deviation
    shortage_in_ALSO_X_array = np.array(shortage_in_ALSO_X)
    shortage_in_CVAR_array = np.array(shortage_in_CVAR)
    shortage_out_ALSO_X_array = np.array(shortage_out_ALSO_X)
    shortage_out_CVAR_array = np.array(shortage_out_CVAR)
    sd_in_ALSO_X = np.std(shortage_in_ALSO_X_array, axis=0)
    sd_in_CVAR = np.std(shortage_in_CVAR_array, axis=0)
    sd_out_ALSO_X = np.std(shortage_out_ALSO_X_array, axis=0)
    sd_out_CVAR = np.std(shortage_out_CVAR_array, axis=0)
    
    # Plotting of lines of average shortage
    plt.figure(figsize = (10, 7))
    plt.rcParams["font.size"] = 12
        
    plt.plot(minutes, avg_short_in_ALSO_X, 'cornflowerblue', label="ALSO_X in sample")
    plt.plot(minutes, avg_short_out_ALSO_X, 'navy', label="ALSO_X out of sample")
    plt.plot(minutes, avg_short_in_CVAR, 'darkorange', label="CVaR in sample")
    plt.plot(minutes, avg_short_out_CVAR, 'red', label="CVaR out of sample")
    # plt.fill_between(minutes, avg_short_in_ALSO_X - sd_in_ALSO_X, avg_short_in_ALSO_X + sd_in_ALSO_X, color='cornflowerblue', alpha=0.2)
    # plt.fill_between(minutes, avg_short_out_ALSO_X - sd_out_ALSO_X, avg_short_out_ALSO_X + sd_out_ALSO_X, color='navy', alpha=0.2)
    # plt.fill_between(minutes, avg_short_in_CVAR - sd_in_CVAR, avg_short_in_CVAR + sd_in_CVAR, color='darkorange', alpha=0.2)
    # plt.fill_between(minutes, avg_short_out_CVAR - sd_out_CVAR, avg_short_out_CVAR + sd_out_CVAR, color='red', alpha=0.2)
    plt.xlabel('Minute in Hour')
    plt.ylabel('Average power shortage [kW]')
    plt.legend(loc=1)
    plt.show()
    
    # Plotting bar of number of shortage 
    # Define the situations and their corresponding values
    situations = ['ALSO_X in sample', 'CVAR in sample', 'ALSO_X out of sample', 'CVAR out of sample']
    values = [def_count_in_ALSO_X, def_count_in_CVAR, def_count_out_ALSO_X, def_count_out_CVAR]  # Example values for each situation
    
    # Define colors for each bar
    colors1 = ['cornflowerblue', 'darkorange']  # Colors for the first subplot
    colors2 = ['navy', 'red']    # Colors for the second subplot
    
    # Define specific values for the red horizontal bars
    red_line = [epsilon*len(in_sample)*n_min, epsilon*len(out_of_sample)*n_min]
    
    # Create a figure and two subplots side by side
    fig, axs = plt.subplots(1, 2, figsize=(10, 7))
    plt.rcParams["font.size"] = 12
    
    # First subplot with the first two bars
    axs[0].bar(situations[:2], values[:2], color=colors1)
    axs[0].set_title('In sample')
    axs[0].set_ylabel('Number of power shortage')
    
    # Add red horizontal bar with specific value
    axs[0].axhline(y=red_line[0], color='red', linestyle='-', linewidth=1, label='P90 rule')
    axs[0].legend(loc=1)
    
    # Second subplot with the last two bars
    axs[1].bar(situations[2:], values[2:], color=colors2)
    axs[1].set_title('Out of sample')
    axs[1].set_ylabel('Number of power shortage')
    
    # Add red horizontal bar with specific value
    axs[1].axhline(y=red_line[1], color='red', linestyle='-', linewidth=1, label='P90 rule')
    axs[1].legend(loc=1)
    
    # Show the graph
    plt.tight_layout()
    plt.show()
    

Shortage_plot(sto_anc_in_sample, sto_anc_out_of_sample, 0.1)


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


# Plotting_CFD(sto_anc_out_of_sample, optimal_qu_off_ALSO_X)
# Plotting_CFD(sto_anc_out_of_sample, optimal_qu_off_CVAR)


##################################################################
""" Plotting the effect of epsilon choosing on the missing kWh """
##################################################################

def Effect_epsilon(sto_anc_scenarios) :
    # Let's cancel the seed to be able to have something really random
    random.seed(None)
    
    # We are going to iterate on multiple selection of random in sample to reduce the influence of scenarios selection
    n_sel = 30
    
    # List of values for data generation
    epsilon_values = np.arange(0, 0.6, 0.05)
    Qu_off = np.zeros((n_sel,len(epsilon_values)))
    Lost_table = np.zeros((n_sel,150,len(epsilon_values)))
    
    for sel in range(n_sel) :
        print(sel)
        # Selecting scenarios
        in_sample = random.sample(sto_anc_scenarios,50)
        out_of_sample = [scenario for scenario in sto_anc_scenarios if scenario not in in_sample]
        
        # Compute the shortage values
        for i in range(len(epsilon_values)) :
            eps = epsilon_values[i]
            Qu_off[sel,i] = P90_ALSO_X(in_sample, eps)
            shortage, def_count, decision = P90_verify(Qu_off[sel,i], out_of_sample, eps)
            shortage_array = np.array(shortage)
            shortage_sum = np.sum(shortage_array, axis=1)
            shortage_kWh = shortage_sum * (1/60)
            Lost_table[sel,:,i] = shortage_kWh
        
    # Compute mean values and standard deviation
    Mean_lost = np.mean(Lost_table, axis=(0,1))
    SD_lost = np.std(Lost_table, axis=(0,1))
    Q10_lost = np.percentile(Lost_table, 10, axis=(0,1))
    Q90_lost = np.percentile(Lost_table, 90, axis=(0,1))
    Mean_qu = np.mean(Qu_off, axis=0)
    Q10_qu = np.percentile(Qu_off, 10, axis=0)
    Q90_qu = np.percentile(Qu_off, 90, axis=0)
    SD_qu = np.std(Qu_off, axis=0)
    
    # Plotting of lines of average reserve shortfall
    plt.figure(figsize = (10, 7))
    plt.rcParams["font.size"] = 12
    plt.plot(epsilon_values, Mean_lost, label='Average reserve shorfall')
    plt.fill_between(epsilon_values, Q10_lost, Q90_lost, label='Quantiles 10% and 90% of the reserve shortfall', alpha=0.2)
    plt.legend(loc=2)
    plt.xlabel('Epsilon')
    plt.ylabel('Expected reserve shortfall [kWh]')
    plt.show()
    
    # Plotting of lines of optimal reserve bid
    plt.figure(figsize = (10, 7))
    plt.rcParams["font.size"] = 12
    plt.plot(epsilon_values, Mean_qu, label='Average optimal reserve bid')
    plt.fill_between(epsilon_values, Q10_qu, Q90_qu,label='Quantiles 10% and 90% of the optimal reserve bid', alpha=0.2)
    plt.legend(loc=2)
    plt.xlabel('Epsilon')
    plt.ylabel('Power bid [kW]')
    plt.show()
    
    # Take out the standard deviations
    print(f"Standard deviation lost {SD_lost}")
    print(f"Standard deviation qu {SD_qu}")
    
Effect_epsilon(sto_anc_scenarios)


##########################################################################
""" Heat Map for making Energinet rule and number of in_sample varying """
##########################################################################

def Heat_map(sto_anc_scenarios) :
    # Let's cancel the seed to be able to have something really random
    random.seed(None)
    
    # We are going to iterate on multiple selection of random in sample to reduce the influence of scenarios selection
    n_sel = 20
    
    #Generate the data for the heat map
    # Lists for data generation
    sample_values = np.arange(10, 210, 20)
    epsilon_values = np.arange(0, 0.6, 0.1)
    
    # Table with values of offered quantity
    Off_qu = np.zeros((n_sel, len(epsilon_values), len(sample_values)))
    # Table with the values of verification
    Verif = np.zeros((n_sel, len(epsilon_values), len(sample_values)))
    Numb_shortage = np.zeros((n_sel, len(epsilon_values), len(sample_values)))
    
    for sel in range(n_sel) :
        print(sel)
        # Get the data for the plotting
        for i in range(len(sample_values)):
            for j in range(len(epsilon_values)) :
                print(f"\n Sample : {sample_values[i]} and Epsilon : {epsilon_values[j]}")
                sample = sample_values[i]
                epsilon = epsilon_values[j]
                # Selecting scenarios
                in_sample = random.sample(sto_anc_scenarios,sample)
                out_of_sample = [scenario for scenario in sto_anc_scenarios if scenario not in in_sample]
                # Computing the offered value
                Off_qu[sel,j,i] = P90_ALSO_X(in_sample, epsilon)
                # Verifying Energinet rule
                _, Numb_shortage[sel,j,i], Verif[sel,j,i] = P90_verify(Off_qu[sel,j,i], out_of_sample, epsilon)
                Numb_shortage[sel,j,i] = Numb_shortage[sel,j,i]/len(out_of_sample)
            
    # Computing the average of the values
    Avg_Off_qu = np.round(np.mean(Off_qu, axis=0), decimals=2)
    Avg_Numb_shortage = np.mean(Numb_shortage, axis=0)
    Avg_Verif = np.mean(Verif, axis=0)
    Avg_Verif_round = np.round(Avg_Verif)
    
    #Plotting the thing
    
    # Create a grid of squares
    x, y = np.meshgrid(sample_values, epsilon_values)
    
    # Plot the grid with colors representing the binary values of the third property
    plt.figure(figsize=(10, 5))
    plt.rcParams["font.size"] = 12
    
    # With percentage of shortage
    plt.pcolormesh(x, y, Avg_Numb_shortage, cmap='viridis', shading='auto')
    # Add color scale on the right side
    plt.colorbar(label='Percentage of power shortage [%]')
    # Add text annotations to each square
    for i in range(len(sample_values)):
        for j in range(len(epsilon_values)):
            plt.text(sample_values[i], epsilon_values[j], f'{Avg_Off_qu[j,i]}',
                     ha='center', va='center', color='white', fontsize=10)
    plt.xlabel('Number of in sample scenarios selected')
    plt.ylabel('Epsilon')
    # Remove horizontal grid lines
    plt.grid(axis='y', linestyle='')
    plt.show()
    
    
    #Plotting the thing
    # Define colors for binary values (0: red, 1: green)
    colors = ['red', 'green']
    cmap = ListedColormap(colors)
    
    # Create a grid of squares
    x, y = np.meshgrid(sample_values, epsilon_values)
    
    # Plot the grid with colors representing the binary values of the third property
    plt.figure(figsize=(10, 6))
    plt.rcParams["font.size"] = 12
    
    # With if it respect or not
    plt.pcolormesh(x, y, Avg_Verif_round, cmap=cmap, shading='auto')
    # Add text annotations to each square
    for i in range(len(sample_values)):
        for j in range(len(epsilon_values)):
            plt.text(sample_values[i], epsilon_values[j], f'{Avg_Off_qu[j,i]}',
                     ha='center', va='center', color='white', fontsize=10)
    plt.xlabel('Number of in sample scenarios selected')
    plt.ylabel('Epsilon')
    # Remove horizontal grid lines
    plt.grid(axis='y', linestyle='')
    plt.show()
    
    return(Avg_Off_qu, Avg_Numb_shortage, Avg_Verif_round)


# Avg_Off_qu, Avg_Numb_shortage, Avg_Verif_round = Heat_map(sto_anc_scenarios)



















