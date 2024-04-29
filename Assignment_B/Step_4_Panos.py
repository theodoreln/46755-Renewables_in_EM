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
from Step_3 import Offering_one_price_risk, Offering_two_price_risk

def Show_distribution(profit, nb_bins) :
    
    # Create histogram
    plt.hist(profit, bins=nb_bins, color='blue', edgecolor='black')
    
    # Add labels and title
    plt.xlabel('Profit')
    plt.ylabel('Frequency')
    plt.title('Histogram of profits')
    
    # Display the plot
    plt.show()    


optimal_qu_off, optimal_obj, cvar = Offering_one_price_risk(in_sample, 0.5)
#optimal_qu_off, optimal_obj, cvar = Offering_two_price_risk(in_sample, 0.1)

n_scen=len(in_sample)
n_out_scen=len(out_of_sample)
n_hour=24
DA_profit = [0]*n_out_scen
BM_result = [0]*n_out_scen
total_revenue_out=np.zeros(n_out_scen)
price_scheme = 1

# Calculate revenue from the Day-Ahead market
for w in range(n_out_scen) :
    profit_w = 0
    for t in range(n_hour) :
        profit_w +=out_of_sample['DA_price'][w][t]*optimal_qu_off[t]  

    # Save the DA profit for scenario w
    DA_profit[w] = profit_w

# Calculate how much WF should pay or be paid in BM
for w in range(n_out_scen) :
    imb_pay = 0
    for t in range(n_hour) :
        optimal_qu_diff = out_of_sample['DA_forecast'][w][t] - optimal_qu_off[t]
        # BM result depends on the price scheme
        if price_scheme == 1 :
            imb_pay +=  out_of_sample['Binary_var'][w][t]*0.9*out_of_sample['DA_price'][w][t]*optimal_qu_diff + (1-out_of_sample['Binary_var'][w][t])*1.2*out_of_sample['DA_price'][w][t]*optimal_qu_diff
        elif price_scheme == 2 : 
            if out_of_sample['Binary_var'][w][t] == 1 :
                if optimal_qu_diff >= 0 :
                    imb_pay += 0.9*out_of_sample['DA_price'][w][t]*optimal_qu_diff
                else :
                    imb_pay += out_of_sample['DA_price'][w][t]*optimal_qu_diff
            else :
                if optimal_qu_diff >= 0 :
                    imb_pay += out_of_sample['DA_price'][w][t]*optimal_qu_diff
                else :
                    imb_pay += 1.2*out_of_sample['DA_price'][w][t]*optimal_qu_diff
                
    # Save the profit for scenario w
    BM_result[w] = imb_pay

BM_avg = sum(BM_result)/len(BM_result)

print(BM_avg)

for w in range(n_out_scen):
    
    total_revenue_out[w]=BM_avg+DA_profit[w]


Show_distribution(total_revenue_out, 80)












