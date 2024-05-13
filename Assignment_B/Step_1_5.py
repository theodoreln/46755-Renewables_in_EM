
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















