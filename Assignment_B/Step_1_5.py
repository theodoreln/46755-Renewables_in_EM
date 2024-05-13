
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
from Data import scenarios
from Step_1_3 import Offering_one_price_risk, Offering_two_price_risk
from Step_1_2 import Profits_scenarios, Show_distribution, Box_plot

def Out_of_sample(in_sample, out_of_sample):
    
    beta=0.2
    n_out_scen=len(out_of_sample)
    n_scen=len(in_sample)
    
    ######################################
    """ Out of sample simulation"""
    ######################################  

    optimal_qu_off_one, optimal_obj, cvar_value = Offering_one_price_risk(in_sample, beta)
    optimal_qu_off_two, optimal_obj, cvar_value = Offering_two_price_risk(in_sample, beta)

    out_profit_one = Profits_scenarios(out_of_sample, optimal_qu_off_one, 1, n_out_scen)
    out_profit_two = Profits_scenarios(out_of_sample, optimal_qu_off_two, 2, n_out_scen)
    # Show_distribution(out_profit_one, 80)
    # Show_distribution(out_profit_two, 80)
        
    out_of_sample_profit_one=sum(out_profit_one)/n_out_scen
    out_of_sample_profit_two=sum(out_profit_two)/n_out_scen


    ######################################
    """ In sample simulation"""
    ###################################### 
    optimal_qu_off_one, optimal_obj, cvar_value = Offering_one_price_risk(in_sample, beta)
    optimal_qu_off_two, optimal_obj, cvar_value = Offering_two_price_risk(in_sample, beta)

    in_profit_one = Profits_scenarios(in_sample, optimal_qu_off_one, 1, n_scen)
    in_profit_two = Profits_scenarios(in_sample, optimal_qu_off_two, 2, n_scen)
    # Show_distribution(in_profit_one, 80)
    # Show_distribution(in_profit_two, 80)
            
    in_sample_profit_one=sum(in_profit_one)/n_scen
    in_sample_profit_two=sum(in_profit_two)/n_scen
      
    # print("Out of sample profit one:", out_of_sample_profit_one)
    # print("Out of sample profit two:", out_of_sample_profit_two)  
    # print("In sample profit one:", in_sample_profit_one)
    # print("In sample profit two:", in_sample_profit_two)
    
    #diff_one=100*abs(out_of_sample_profit_one-in_sample_profit_one)/((out_of_sample_profit_one+in_sample_profit_one)/2)
    #diff_two=100*abs(out_of_sample_profit_two-in_sample_profit_two)/((out_of_sample_profit_two+in_sample_profit_two)/2)
    
    diff_one=abs(out_of_sample_profit_one-in_sample_profit_one)
    diff_two=abs(out_of_sample_profit_two-in_sample_profit_two)
    
    # print("Diff One:", diff_one)
    # print("Diff Two:", diff_two)
    
    return(in_profit_one, out_profit_one, in_profit_two, out_profit_two)

# =========================================================================================
# This part is for the sensitivity analysis on taking different scenarios as seen scenarios
# =========================================================================================

# profits = []
# names = []
# colors = []
# num_in_sample=250

# random.seed(123)
# in_sample = pd.DataFrame (random.sample(scenarios,num_in_sample), columns=['DA_forecast','DA_price','Binary_var'])
# remaining_scenarios = [scenario for scenario in scenarios if scenario not in in_sample.values.tolist()]
# out_of_sample = pd.DataFrame(remaining_scenarios, columns=['DA_forecast', 'DA_price', 'Binary_var'])
    
# _, _, in_profit, out_profit =Out_of_sample(in_sample, out_of_sample)
# profits.append(in_profit)
# profits.append(out_profit)

# # random.seed(456)
# scenarios_2 = [scenario for scenario in scenarios if scenario not in in_sample.values.tolist()]
# in_sample_2 = pd.DataFrame (random.sample(scenarios_2,num_in_sample), columns=['DA_forecast','DA_price','Binary_var'])
# remaining_scenarios = [scenario for scenario in scenarios if scenario not in in_sample_2.values.tolist()]
# out_of_sample = pd.DataFrame(remaining_scenarios, columns=['DA_forecast', 'DA_price', 'Binary_var'])
    
# _, _, in_profit, out_profit =Out_of_sample(in_sample_2, out_of_sample)
# profits.append(in_profit)
# profits.append(out_profit) 

# scenarios_3 = [scenario for scenario in scenarios if scenario not in in_sample.values.tolist() or in_sample_2.values.tolist()]
# in_sample_3 = pd.DataFrame (random.sample(scenarios_3,num_in_sample), columns=['DA_forecast','DA_price','Binary_var'])
# remaining_scenarios = [scenario for scenario in scenarios if scenario not in in_sample_3.values.tolist()]
# out_of_sample = pd.DataFrame(remaining_scenarios, columns=['DA_forecast', 'DA_price', 'Binary_var'])
    
# _, _, in_profit, out_profit =Out_of_sample(in_sample_3, out_of_sample)
# profits.append(in_profit)
# profits.append(out_profit)
  
# scenarios_4 = [scenario for scenario in scenarios if scenario not in in_sample.values.tolist() or in_sample_2.values.tolist() or in_sample_3.values.tolist()]
# in_sample_4 = pd.DataFrame (random.sample(scenarios_4,num_in_sample), columns=['DA_forecast','DA_price','Binary_var'])
# remaining_scenarios = [scenario for scenario in scenarios if scenario not in in_sample_4.values.tolist()]
# out_of_sample = pd.DataFrame(remaining_scenarios, columns=['DA_forecast', 'DA_price', 'Binary_var'])

# _, _, in_profit, out_profit =Out_of_sample(in_sample_4, out_of_sample)
# profits.append(in_profit)
# profits.append(out_profit)

# names = ['In 1', 'Out 1', 'In 2', 'Out 2', 'In 3', 'Out 3', 'In 4', 'Out 4']
# colors = ['cornflowerblue', 'blue', 'orange', 'darkorange', 'limegreen', 'green', 'plum', 'magenta']

# Box_plot(profits, names, colors)

# =============================================================================
# Impact of the choice of the 250 in-sample scenarios on the expected profit
# =============================================================================

def Effect_choice_in_sample_expected(sample, scheme) :
    
    # Initialization
    beta = 0.2
    random.seed(None)
    n_sel = 200
    
    # Store the values
    Quantity_off = np.zeros((n_sel, 24))
    Expected_profit_in = np.zeros((n_sel))
    Expected_profit_out = np.zeros((n_sel))
    Expected_profit_diff = np.zeros((n_sel))
    
    for i in range(n_sel) :
        print(i)
        in_sample = pd.DataFrame (random.sample(scenarios,250), columns=['DA_forecast','DA_price','Binary_var'])
        remaining_scenarios = [scenario for scenario in scenarios if scenario not in in_sample.values.tolist()]
        out_of_sample = pd.DataFrame(remaining_scenarios, columns=['DA_forecast', 'DA_price', 'Binary_var'])
        if scheme == 1 :
            optimal_qu_off_one, optimal_obj, cvar_value = Offering_one_price_risk(in_sample, beta)
            Quantity_off[i,:] = optimal_qu_off_one
            Expected_profit_in[i] = optimal_obj
            profit_one = Profits_scenarios(out_of_sample, optimal_qu_off_one, 1, 950)
            Expected_profit_out[i] = np.mean(np.array(profit_one), axis=0)
            Expected_profit_diff[i] = Expected_profit_in[i] - Expected_profit_out[i]
        elif scheme == 2 :
            optimal_qu_off_two, optimal_obj, cvar_value = Offering_two_price_risk(in_sample, beta)
            Quantity_off[i,:] = optimal_qu_off_two
            Expected_profit_in[i] = optimal_obj
            profit_two = Profits_scenarios(out_of_sample, optimal_qu_off_two, 2, 950)
            Expected_profit_out[i] = np.mean(np.array(profit_two), axis=0)
            Expected_profit_diff[i] = Expected_profit_in[i] - Expected_profit_out[i]
    
    Mean_off = np.mean(Quantity_off, axis=0)
    Q25_off = np.percentile(Quantity_off, 25, axis=0)
    Q75_off = np.percentile(Quantity_off, 75, axis=0)
    Mean_profit_in = np.mean(Expected_profit_in, axis=0)
    Mean_profit_out = np.mean(Expected_profit_out, axis=0)
    Mean_profit_diff = np.mean(Expected_profit_diff, axis=0)
    SD_profit_diff = np.std(Expected_profit_diff, axis=0)
    
    Hour = np.arange(1,25,1)
    
    # Quantity offered plot
    plt.figure(figsize = (10, 7))
    plt.rcParams["font.size"] = 12
    plt.plot(Hour, Mean_off, label='Average quantity offered')
    plt.fill_between(Hour, Q25_off, Q75_off, label='Quantiles 25% and 75% of the quantity offered', alpha=0.2)
    plt.legend(loc=2)
    plt.xlabel('Hours')
    plt.ylabel('Quantity offered [MW]')
    plt.show()
    
    # Distribution of in and out
    
    
    print(f"Mean profit in : {Mean_profit_in}")
    print(f"Mean profit out : {Mean_profit_out}")
    print(f"Mean profit diff : {Mean_profit_diff}")
    print(f"Standard deviation profit diff : {SD_profit_diff}")
    
    return(Mean_profit_in, Mean_profit_out, Mean_profit_diff, SD_profit_diff)

# Mean_profit_in_two, Mean_profit_out_two, Mean_profit_diff_two, SD_profit_diff_two = Effect_choice_in_sample_expected(scenarios, 2)
# Mean_profit_in_one, Mean_profit_out_one, Mean_profit_diff_one, SD_profit_diff_one = Effect_choice_in_sample_expected(scenarios, 1)


# =============================================================================
# Impact of the proportion of in-sample scenarios on the expected profit
# =============================================================================

def Effect_choice_proportion_expected(scenarios, scheme) :
    
    # Initialization
    beta = 0.2
    random.seed(None)
    n_sel = 100
    proportion_values = np.arange(0.05,1,0.05)
    nb_prop = len(proportion_values)
    
    # Store the values
    Quantity_off = np.zeros((nb_prop, n_sel, 24))
    Expected_profit_in = np.zeros((nb_prop, n_sel))
    Expected_profit_out = np.zeros((nb_prop, n_sel))
    Expected_profit_diff = np.zeros((nb_prop, n_sel))
    
    for p in range(nb_prop) :
        print(p)
        for i in range(n_sel) :
            in_sample = pd.DataFrame (random.sample(scenarios,int(1200*float(proportion_values[p]))), columns=['DA_forecast','DA_price','Binary_var'])
            remaining_scenarios = [scenario for scenario in scenarios if scenario not in in_sample.values.tolist()]
            out_of_sample = pd.DataFrame(remaining_scenarios, columns=['DA_forecast', 'DA_price', 'Binary_var'])
            if scheme == 1 :
                optimal_qu_off_one, optimal_obj, cvar_value = Offering_one_price_risk(in_sample, beta)
                Quantity_off[p,i,:] = optimal_qu_off_one
                Expected_profit_in[p,i] = optimal_obj
                profit_one = Profits_scenarios(out_of_sample, optimal_qu_off_one, 1, 1200-int(1200*float(proportion_values[p])))
                Expected_profit_out[p,i] = np.mean(np.array(profit_one), axis=0)
                Expected_profit_diff[p,i] = Expected_profit_in[p,i] - Expected_profit_out[p,i]
            elif scheme == 2 :
                optimal_qu_off_two, optimal_obj, cvar_value = Offering_two_price_risk(in_sample, beta)
                Quantity_off[p,i,:] = optimal_qu_off_two
                Expected_profit_in[p,i] = optimal_obj
                profit_two = Profits_scenarios(out_of_sample, optimal_qu_off_two, 2, 1200-int(1200*float(proportion_values[p])))
                Expected_profit_out[p,i] = np.mean(np.array(profit_two), axis=0)
                Expected_profit_diff[p,i] = Expected_profit_in[p,i] - Expected_profit_out[p,i]
    
    Mean_in = np.mean(Expected_profit_in, axis=1)
    Mean_out = np.mean(Expected_profit_out, axis=1)
    Q25_in = np.percentile(Expected_profit_in, 25, axis=1)
    Q75_in = np.percentile(Expected_profit_in, 75, axis=1)
    Q25_out = np.percentile(Expected_profit_out, 25, axis=1)
    Q75_out = np.percentile(Expected_profit_out, 75, axis=1)
    
    # In and out profits
    plt.figure(figsize = (10, 7))
    plt.rcParams["font.size"] = 12
    plt.plot(proportion_values, Mean_in, 'b', label='Average in sample profit')
    plt.fill_between(proportion_values, Q25_in, Q75_in, color='blue', label='Quantiles 25% and 75% of the in sample profit', alpha=0.2)
    plt.plot(proportion_values, Mean_out, 'r', label='Average out of sample profit')
    plt.fill_between(proportion_values, Q25_out, Q75_out, color='red', label='Quantiles 25% and 75% of the out of sample profit', alpha=0.2)
    plt.legend(loc=9)
    plt.xlabel('Proportion of in sample')
    plt.ylabel('Expected profit [$]')
    plt.show()
    
# Effect_choice_proportion_expected(scenarios, 1)
# Effect_choice_proportion_expected(scenarios, 2)















