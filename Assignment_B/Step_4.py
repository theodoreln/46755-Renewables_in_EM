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
import Step_3 as s3
import Step_1_2 as s12




######################################
""" Out of sample simulation"""
######################################  


def out_of_sample_simulation (price_scheme, beta):

    
    if price_scheme == 1 : 
        optimal_qu_off, optimal_obj, cvar_value = s3.Offering_one_price_risk(in_sample, beta)
    elif price_scheme == 2 :
        optimal_qu_off, optimal_obj, cvar_value = s3.Offering_two_price_risk(in_sample, beta)
        
    
    
    n_out_scen=len(out_of_sample)
    n_hour=24
    
    delta=np.zeros((n_out_scen, n_hour))
    delta_up=np.zeros((n_out_scen, n_hour))
    delta_down=np.zeros((n_out_scen, n_hour))
    y=np.zeros((n_out_scen, n_hour))
    DA_profit=np.zeros(n_out_scen)
    total_revenue_out=np.zeros(n_out_scen)
    
    # Calculate revenue from the Day-Ahead market
    
    for w in range(n_out_scen) :
        profit_w = 0
        for t in range(n_hour) :
            profit_w +=out_of_sample['DA_price'][w][t]*optimal_qu_off[t]  
    
        # Save the DA profit for scenario w
        DA_profit[w] = profit_w
        
    
    if price_scheme==1:
        for w in range(n_out_scen):
            for t in range(n_hour):
                y[w][t] = (out_of_sample['Binary_var'][w][t] * 0.9 * out_of_sample['DA_price'][w][t] * (out_of_sample['DA_forecast'][w][t] - optimal_qu_off[t]) +
                       (1 - out_of_sample['Binary_var'][w][t]) * 1.2 * out_of_sample['DA_price'][w][t] * (out_of_sample['DA_forecast'][w][t] - optimal_qu_off[t]))
                
                
    if price_scheme==2:
    
        #delta[w][t] = [out_of_sample['DA_forecast'][w][t] - optimal_qu_off[t] for w in range(n_out_scen) for t in range(n_hour)]
     
        for w in range(n_out_scen):
            for t in range(n_hour):
                
                delta[w][t] = out_of_sample['DA_forecast'][w][t] - optimal_qu_off[t]
             
                
                if delta[w][t]<0:
                    delta_up[w][t]=0
                    delta_down[w][t]=-delta[w][t]
                if delta[w][t]>0:
                    delta_up[w][t]=delta[w][t]
                    delta_down[w][t]=0
                if delta[w][t]==0:
                    delta_up[w][t]=0
                    delta_down[w][t]=0
                    
        
                y[w][t] = ((out_of_sample['Binary_var'][w][t] * (0.9 * out_of_sample['DA_price'][w][t] * delta_up[w][t] - out_of_sample['DA_price'][w][t] * delta_down[w][t])
                             + (1 - out_of_sample['Binary_var'][w][t]) * (out_of_sample['DA_price'][w][t] * delta_up[w][t] - 1.2 * out_of_sample['DA_price'][w][t] * delta_down[w][t])))
    
    #calculate balancing profit
    for w in range(n_out_scen):
        profit_balanced=0
        for t in range(n_hour):
            profit_balanced+=y[w][t]
    
        # Save the balancing profit for scenario w
        total_revenue_out[w]=profit_balanced+DA_profit[w]
    
    #display distribution
    s12.Show_distribution(total_revenue_out, 80)
    
    
    #calcualte average profit
    out_of_sample_profit=sum(total_revenue_out)/n_out_scen
    
    
    return(out_of_sample_profit)



out_of_sample_profit=out_of_sample_simulation(1, 0.5)


