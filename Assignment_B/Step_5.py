
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
from Step_3 import Offering_one_price_risk, Offering_two_price_risk
from Step_1_2 import Profits_scenarios, Show_distribution

def Out_of_sample(in_sample, out_of_sample):
    
    beta=0.2
    n_out_scen=len(out_of_sample)
    n_scen=len(in_sample)
    
    ######################################
    """ Out of sample simulation"""
    ######################################  
    
    optimal_qu_off_one, optimal_obj, cvar_value = Offering_one_price_risk(in_sample, beta)
    optimal_qu_off_two, optimal_obj, cvar_value = Offering_two_price_risk(in_sample, beta)
    
    profit_one = Profits_scenarios(out_of_sample, optimal_qu_off_one, 1, n_out_scen)
    profit_two = Profits_scenarios(out_of_sample, optimal_qu_off_two, 2, n_out_scen)
        
    out_of_sample_profit_one=sum(profit_one)/n_out_scen
    out_of_sample_profit_two=sum(profit_two)/n_out_scen
    
    ######################################
    """ In sample simulation"""
    ###################################### 
    optimal_qu_off_one, optimal_obj, cvar_value = Offering_one_price_risk(in_sample, beta)
    optimal_qu_off_two, optimal_obj, cvar_value = Offering_two_price_risk(in_sample, beta)
    
    profit_one = Profits_scenarios(in_sample, optimal_qu_off_one, 1, n_scen)
    profit_two = Profits_scenarios(in_sample, optimal_qu_off_two, 2, n_scen)
            
    in_sample_profit_one=sum(profit_one)/n_scen
    in_sample_profit_two=sum(profit_two)/n_scen
      
    print("Out of sample profit one:", out_of_sample_profit_one)
    print("Out of sample profit two:", out_of_sample_profit_two)  
    print("In sample profit one:", in_sample_profit_one)
    print("In sample profit two:", in_sample_profit_two)
    
    #diff_one=100*abs(out_of_sample_profit_one-in_sample_profit_one)/((out_of_sample_profit_one+in_sample_profit_one)/2)
    #diff_two=100*abs(out_of_sample_profit_two-in_sample_profit_two)/((out_of_sample_profit_two+in_sample_profit_two)/2)
    
    diff_one=abs(out_of_sample_profit_one-in_sample_profit_one)
    diff_two=abs(out_of_sample_profit_two-in_sample_profit_two)
    
    print("Diff One:", diff_one)
    print("Diff Two:", diff_two)
    
    return(diff_one, diff_two)

# =========================================================================================
# This part is for the sensitivity analysis on taking different scenarios as seen scenarios
# =========================================================================================

# random.seed(123)
# in_sample = pd.DataFrame (random.sample(scenarios,num_in_sample), columns=['DA_forecast','DA_price','Binary_var'])
# remaining_scenarios = [scenario for scenario in scenarios if scenario not in in_sample.values.tolist()]
# out_of_sample = pd.DataFrame(remaining_scenarios, columns=['DA_forecast', 'DA_price', 'Binary_var'])
    
# diff_one, diff_two=Out_of_sample(in_sample, out_of_sample)
# diff_samples.append(diff_one)
# diff_samples.append(diff_two)

# scenarios_2 = [scenario for scenario in scenarios if scenario not in in_sample.values.tolist()]
# in_sample_2 = pd.DataFrame (random.sample(scenarios_2,num_in_sample), columns=['DA_forecast','DA_price','Binary_var'])
# remaining_scenarios = [scenario for scenario in scenarios if scenario not in in_sample_2.values.tolist()]
# out_of_sample = pd.DataFrame(remaining_scenarios, columns=['DA_forecast', 'DA_price', 'Binary_var'])
    
# diff_one, diff_two=Out_of_sample(in_sample_2, out_of_sample)
# diff_samples.append(diff_one)
# diff_samples.append(diff_two)  

# scenarios_3 = [scenario for scenario in scenarios if scenario not in in_sample.values.tolist() or in_sample_2.values.tolist()]
# in_sample_3 = pd.DataFrame (random.sample(scenarios_3,num_in_sample), columns=['DA_forecast','DA_price','Binary_var'])
# remaining_scenarios = [scenario for scenario in scenarios if scenario not in in_sample_3.values.tolist()]
# out_of_sample = pd.DataFrame(remaining_scenarios, columns=['DA_forecast', 'DA_price', 'Binary_var'])
    
# diff_one, diff_two=Out_of_sample(in_sample_3, out_of_sample)
# diff_samples.append(diff_one)
# diff_samples.append(diff_two)  
  
# scenarios_4 = [scenario for scenario in scenarios if scenario not in in_sample.values.tolist() or in_sample_2.values.tolist() or in_sample_3.values.tolist()]
# in_sample_4 = pd.DataFrame (random.sample(scenarios_4,num_in_sample), columns=['DA_forecast','DA_price','Binary_var'])
# remaining_scenarios = [scenario for scenario in scenarios if scenario not in in_sample_4.values.tolist()]
# out_of_sample = pd.DataFrame(remaining_scenarios, columns=['DA_forecast', 'DA_price', 'Binary_var'])

# diff_one, diff_two=Out_of_sample(in_sample_4, out_of_sample)
# diff_samples.append(diff_one)
# diff_samples.append(diff_two)


# =============================================================================
# Plotting the difference of in-sample and out-of sample profit as a function 
# number of in-sample scenarios
# =============================================================================

acceptable_samples = []
diff_samples = []
num_samples_list=[]
diff_one_list=[]
diff_two_list=[]
num_in_sample=180

for i in range(1,45):
    
    #Creating the in-sample scenarios
    random.seed(123)
    in_sample = pd.DataFrame (random.sample(scenarios,num_in_sample), columns=['DA_forecast','DA_price','Binary_var'])
    
    #Creating the out-of-sample scenarios
    remaining_scenarios = [scenario for scenario in scenarios if scenario not in in_sample.values.tolist()]
    out_of_sample = pd.DataFrame(remaining_scenarios, columns=['DA_forecast', 'DA_price', 'Binary_var'])

    #Calculating the profit differnce
    diff_one, diff_two=Out_of_sample(in_sample, out_of_sample)
    

    num_samples_list.append(num_in_sample)
    diff_one_list.append(diff_one)
    diff_two_list.append(diff_two)

    num_in_sample += 20

#Plotting the difference in profit as a function of number of in-sample scenarios
plt.plot(num_samples_list, diff_one_list, label='One price scheme')
plt.plot(num_samples_list, diff_two_list, label='Two price scheme')

plt.xlabel('Number of In-Sample Scenarios')
plt.ylabel('absolute Difference')

plt.legend()

plt.show()


    




