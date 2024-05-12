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
from Data import in_sample, out_of_sample
from Step_1_3 import Offering_one_price_risk, Offering_two_price_risk
from Step_1_2 import Profits_scenarios, Show_distribution, Box_plot


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
Show_distribution(out_profit_one, 80)
Show_distribution(out_profit_two, 80)
    
out_of_sample_profit_one=sum(out_profit_one)/n_out_scen
out_of_sample_profit_two=sum(out_profit_two)/n_out_scen


######################################
""" In sample simulation"""
###################################### 
optimal_qu_off_one, optimal_obj, cvar_value = Offering_one_price_risk(in_sample, beta)
optimal_qu_off_two, optimal_obj, cvar_value = Offering_two_price_risk(in_sample, beta)

in_profit_one = Profits_scenarios(in_sample, optimal_qu_off_one, 1, n_scen)
in_profit_two = Profits_scenarios(in_sample, optimal_qu_off_two, 2, n_scen)
Show_distribution(in_profit_one, 80)
Show_distribution(in_profit_two, 80)
        
in_sample_profit_one=sum(in_profit_one)/n_scen
in_sample_profit_two=sum(in_profit_two)/n_scen

profits = [in_profit_one, out_profit_one, in_profit_two, out_profit_two]
names = ['One-price in profit', 'One-price out profit', 'Two-price in profit', 'Two-price out profit']
colors = ['cornflowerblue', 'blue', 'orange', 'darkorange']
Box_plot(profits, names, colors)
  
print("Out of sample profit one:", out_of_sample_profit_one)
print("Out of sample profit two:", out_of_sample_profit_two)  
print("In sample profit one:", in_sample_profit_one)
print("In sample profit two:", in_sample_profit_two)

