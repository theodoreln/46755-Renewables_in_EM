# -*- coding: utf-8 -*-
"""
Created on Mon May  6 14:31:50 2024

@author: leoni
"""

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

#in_sample=new_in_sample3

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
    Show_distribution(profit_one, 120)
    Show_distribution(profit_two, 120)
        
    out_of_sample_profit_one=sum(profit_one)/n_out_scen
    out_of_sample_profit_two=sum(profit_two)/n_out_scen
    
    
    ######################################
    """ In sample simulation"""
    ###################################### 
    optimal_qu_off_one, optimal_obj, cvar_value = Offering_one_price_risk(in_sample, beta)
    optimal_qu_off_two, optimal_obj, cvar_value = Offering_two_price_risk(in_sample, beta)
    
    profit_one = Profits_scenarios(in_sample, optimal_qu_off_one, 1, n_scen)
    profit_two = Profits_scenarios(in_sample, optimal_qu_off_two, 2, n_scen)
    Show_distribution(profit_one, 80)
    Show_distribution(profit_two, 80)
            
    in_sample_profit_one=sum(profit_one)/n_scen
    in_sample_profit_two=sum(profit_two)/n_scen
      
    
    print("Out of sample profit one:", out_of_sample_profit_one)
    print("Out of sample profit two:", out_of_sample_profit_two)  
    print("In sample profit one:", in_sample_profit_one)
    print("In sample profit two:", in_sample_profit_two)
    
    
    diff_one=100*abs(out_of_sample_profit_one-in_sample_profit_one)/((out_of_sample_profit_one+in_sample_profit_one)/2)
    diff_two=100*abs(out_of_sample_profit_two-in_sample_profit_two)/((out_of_sample_profit_two+in_sample_profit_two)/2)
    
    
    print("Diff One:", diff_one)
    print("Diff Two:", diff_two)
    
    return(diff_one, diff_two)


acceptable_samples = []
diff_samples = []
num_samples_list=[]
diff_one_list=[]
diff_two_list=[]
num_in_sample=250

    
random.seed(123)
in_sample = pd.DataFrame (random.sample(scenarios,num_in_sample), columns=['DA_forecast','DA_price','Binary_var'])
remaining_scenarios = [scenario for scenario in scenarios if scenario not in in_sample.values.tolist()]
out_of_sample = pd.DataFrame(remaining_scenarios, columns=['DA_forecast', 'DA_price', 'Binary_var'])
    
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

for i in range(1,15):
    random.seed(123)
    in_sample = pd.DataFrame (random.sample(scenarios,num_in_sample), columns=['DA_forecast','DA_price','Binary_var'])
    remaining_scenarios = [scenario for scenario in scenarios if scenario not in in_sample.values.tolist()]
    out_of_sample = pd.DataFrame(remaining_scenarios, columns=['DA_forecast', 'DA_price', 'Binary_var'])
    
    diff_one, diff_two=Out_of_sample(in_sample, out_of_sample)
    
    if diff_one <= 1.5 and diff_two <= 1.5:

        acceptable_samples.append(num_in_sample)
        acceptable_samples.append(diff_one)
        acceptable_samples.append(diff_two)
        
    num_samples_list.append(num_in_sample)
    diff_one_list.append(diff_one)
    diff_two_list.append(diff_two)

    num_in_sample += 20

#print("Anzahl der In-Sample-Szenarien mit akzeptabler Abweichung:", acceptable_samples)
plt.plot(num_samples_list, diff_one_list, label='Diff One')
plt.plot(num_samples_list, diff_two_list, label='Diff Two')

# Beschriftung der Achsen und Titel
plt.xlabel('Number of In-Sample Scenarios')
plt.ylabel('Difference')
plt.title('Difference One and Two over Number of In-Sample Scenarios')

# Legende anzeigen
plt.legend()

# Diagramm anzeigen
plt.show()

#Ersten 250
# Out of sample profit one: 83166.90067885822
# Out of sample profit two: 74590.2619073072
# In sample profit one: 85670.56763947399
# In sample profit two: 76517.57025923725
# Diff One: 2.9657717395942744
# Diff Two: 2.55090464113847

# Zweiten 250
# Out of sample profit one: 83618.28143828613
# Out of sample profit two: 74912.46634055913
# In sample profit one: 83955.32075364719
# In sample profit two: 75400.1034467704
# Diff One: 0.40225824467869364
# Diff Two: 0.6488307756313523

# Dritten 250
# Out of sample profit one: 83336.80244484523
# Out of sample profit two: 74846.51276280828
# In sample profit one: 85024.94092872359
# In sample profit two: 75687.45985406841
# Diff One: 2.005370638307821
# Diff Two: 1.1172854560882612

# Vierten 250
# Out of sample profit one: 84374.70103793302
# Out of sample profit two: 75725.31209912612
# In sample profit one: 81080.92627498964
# In sample profit two: 72179.86111514477
# Diff One: 3.981459943594335
# Diff Two: 4.794221739418193







