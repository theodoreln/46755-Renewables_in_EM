
from Data import Generators, Demands, Wind_Farms
from Step_1 import Single_hour_plot, Commodities
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import gurobipy as gp
import copy
import os
GRB = gp.GRB


############################
""" Optimization problem """
############################

#Parameters of our problem
# Number of hours (but the data are already considered with 24 hours)
n_hour = 24
# Index of the electrolyzer, change first numbers to change the place
index_elec = {0:0, 1:1}
# Hydrogen demand per electrolyser (in tons)
Hydro_demand = 20


# Taking the hour supply and load information and optimizing on 24 hours
def Multiple_hour_optimization(Generators, Wind_Farms, Demands) :
    # Global variables
    # global optimal_conv_gen, optimal_wf_gen, optimal_dem, optimal_elec
    #Number of units to take into account (based on data)
    # Number of conventional generation units
    n_gen = len(Generators)
    # Number of wind farms generation units
    n_wf = len(Wind_Farms)
    # Number of demand units
    n_dem = len(Demands)
    
    #Optimization part
    # Create the model
    model = gp.Model()
    #Initialize the decision variables, each one of them is now time dependant
    # Capacities provided by conventional generation units
    var_conv_gen = model.addVars(n_gen, n_hour, vtype=GRB.CONTINUOUS, name='conv_gen')
    # Capacities provided by wind farms
    var_wf_gen = model.addVars(n_wf, n_hour, vtype=GRB.CONTINUOUS, name='wf_gen')
    # Demand capacities fullfilled
    var_dem = model.addVars(n_dem, n_hour, vtype=GRB.CONTINUOUS, name='dem')
    # Capacities provided to the electrolyzer
    var_elec = model.addVars(len(index_elec), n_hour, vtype=GRB.CONTINUOUS, name='elec')
    
    # Add the objective function to the model, sum on every t, separation of the conventional generation units and the wind farms
    model.setObjective(gp.quicksum(gp.quicksum(Demands['Offer price'][d][t]*var_dem[d,t] for d in range(n_dem))-gp.quicksum(Generators['Bid price'][g]*var_conv_gen[g,t] for g in range(n_gen))-gp.quicksum(Wind_Farms['Bid price'][wf][t]*var_wf_gen[wf,t] for wf in range(n_wf)) for t in range(n_hour)), GRB.MAXIMIZE)
    
    #Add constraints to the model
    # Quantity supplied = Quantity used for every time step t
    for t in range(n_hour):
        constr_name = f"Power balance hour {t+1}"
        model.addConstr(gp.quicksum(var_dem[d,t] for d in range(n_dem)) - gp.quicksum(var_conv_gen[g,t] for g in range(n_gen)) - gp.quicksum(var_wf_gen[wf,t] for wf in range(n_wf)) == 0, name = constr_name)
    # Constraint that must be fullfilled for every time ste
    for t in range(n_hour) :
        for g in range(n_gen) :
            # The provided capacity of a conventional unit should not be more than the maximum capacity
            model.addConstr(var_conv_gen[g,t] <= Generators['Capacity'][g])
            model.addConstr(var_conv_gen[g,t] >= 0)
            # Ramp up and ramp down constraints of conventional units depends on the time step, for t=1 take into account initial power
            if t != 0 :
                model.addConstr(var_conv_gen[g,t] - var_conv_gen[g,t-1] <= Generators['Ramp up'][g])
                model.addConstr(var_conv_gen[g,t-1] - var_conv_gen[g,t] <= Generators['Ramp down'][g])
            else :
                model.addConstr(var_conv_gen[g,t] - Generators['Initial power'][g] <= Generators['Ramp up'][g])
                model.addConstr(Generators['Initial power'][g] - var_conv_gen[g,t] <= Generators['Ramp down'][g])
        for d in range(n_dem) :
            # The capacity provided to a demand unit should not be higher than the maximum capacity demand
            model.addConstr(var_dem[d,t] <= Demands['Load'][d][t])
            model.addConstr(var_dem[d,t] >= 0)
        for wf in range(n_wf) :
            # If the wind farm has an electrolyzer, then the capacity provided to the grid is limited by the maximum capacity (which is the wind profile here) minus the capacity provided to the electrolyzer
            if wf in index_elec :
                model.addConstr(var_wf_gen[wf,t] <= Wind_Farms['Capacity'][wf][t] - var_elec[index_elec[wf],t])
            else :
                model.addConstr(var_wf_gen[wf,t] <= Wind_Farms['Capacity'][wf][t])
            model.addConstr(var_wf_gen[wf,t] >= 0)
        for e in range(len(index_elec)) :
            # The capacity of each electrolyzer is limited to 100 MW which corresponds to half the maximum capacity of a wind farm (without wind profil)
            model.addConstr(var_elec[e,t] <= 100)
            model.addConstr(var_elec[e,t] >= 0)
    # Constraints on hydrogen production which should be higher than the demand, for each electrolyzer
    for e in range(len(index_elec)) :
        model.addConstr(gp.quicksum(var_elec[e,t] for t in range(n_hour))*0.018 >= Hydro_demand)
    
    #Solve the problem
    model.optimize()
    
    #Get the optimal values
    if model.status == GRB.OPTIMAL:
        # Create a list to store the optimal values of the variables
        optimal_conv_gen = [[round(var_conv_gen[g,t].X,2) for g in range(n_gen)] for t in range(n_hour)]
        optimal_wf_gen = [[round(var_wf_gen[wf,t].X,2) for wf in range(n_wf)] for t in range(n_hour)]
        optimal_dem = [[round(var_dem[d,t].X,2) for d in range(n_dem)] for t in range(n_hour)]
        optimal_elec = [[round(var_elec[e,t].X,2) for e in range(2)] for t in range(n_hour)]
        equilibrium_prices = []
        # Value of the optimal objective
        optimal_obj = model.ObjVal
        
        """ KKTs output """
        # Write KKT's condition in a text file
        # Define the folder path and the file name
        folder_path = "results/multiple_hour"
        file_name = "KKTs.txt"
        
        # Create the folder if it doesn't exist
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        
        # Combine the folder path and file name
        file_path = os.path.join(folder_path, file_name)
        
        # Open the file in write mode
        with open(file_path, 'w') as file:
            file.write("KKTs for the multiple hour optimization :")
            file.write("\n\n")
            # Write lines to the file
            for c in model.getConstrs():
                file.write(f"{c.ConstrName}: {c.Pi} : {c.Sense} \n")
                if 'Power balance hour' in c.constrName :
                    equilibrium_prices.append(c.Pi)
        
    else:
        print("Optimization did not converge to an optimal solution.")
    
    return(optimal_conv_gen, optimal_wf_gen, optimal_dem, optimal_elec, equilibrium_prices)


#####################################
""" Vizualize result for one hour """
#####################################

# Function to select the value for only one hour out of the all day
def Select_one_hour(Generators, Demands, optimal_conv_gen, optimal_wf_gen, optimal_dem, optimal_elec, equilibrium_prices, select_hour) :
    Generators_hour = Generators.copy(deep=True)
    Wind_Farms_hour = Wind_Farms.copy(deep=True)
    for wf in range(len(Wind_Farms)) :
        Wind_Farms_hour.loc[wf, 'Capacity'] = round(Wind_Farms['Capacity'][wf][select_hour-1],2)
        Wind_Farms_hour.loc[wf, 'Bid price'] = Wind_Farms['Bid price'][wf][select_hour-1]
    Demands_hour = Demands.copy(deep=True)
    for d in range(len(Demands)) :
        Demands_hour.loc[d, 'Load'] = Demands['Load'][d][select_hour-1]
        Demands_hour.loc[d, 'Offer price'] = Demands['Offer price'][d][select_hour-1]
    optimal_conv_gen_hour = optimal_conv_gen[select_hour-1]
    optimal_wf_gen_hour = optimal_wf_gen[select_hour-1]
    optimal_dem_hour = optimal_dem[select_hour-1]
    optimal_elec_hour = optimal_elec[select_hour-1]
    equilibrium_price = equilibrium_prices[select_hour-1]
    return(Generators_hour, Wind_Farms_hour, Demands_hour, optimal_conv_gen_hour, optimal_wf_gen_hour, optimal_dem_hour, optimal_elec_hour, equilibrium_price)

# Function to put back the capacities and the optimal values in the right order so we can easily have the price 
def Right_order(Generators_hour, Wind_Farms_hour, Demands_hour, optimal_conv_gen_hour, optimal_wf_gen_hour, optimal_dem_hour) :
    Generators_hour['Optimal'] = optimal_conv_gen_hour
    Wind_Farms_hour['Optimal'] = optimal_wf_gen_hour
    Demands_hour['Optimal'] = optimal_dem_hour
    Supply_hour = pd.concat([Generators_hour, Wind_Farms_hour], axis=0).reset_index(drop=True)
    Supply_hour = Supply_hour.sort_values(by=['Bid price', 'Optimal'], ascending=[True, False]).reset_index(drop=True)
    Demands_hour = Demands_hour.sort_values(by='Offer price', ascending=False).reset_index(drop=True)
    optimal_sup_hour = Supply_hour['Optimal'].to_list()
    optimal_dem_hour = Demands_hour['Optimal'].to_list()
    return(Supply_hour, Demands_hour, optimal_sup_hour, optimal_dem_hour)

# Trying for only one hour    
# optimal_conv_gen, optimal_wf_gen, optimal_dem, optimal_elec = Multiple_hour_optimization(Generators, Wind_Farms, Demands)
# Generators_hour, Wind_Farms_hour, Demands_hour, optimal_conv_gen_hour, optimal_wf_gen_hour, optimal_dem_hour, optimal_elec_hour = Select_one_hour(Generators, Demands, optimal_conv_gen, optimal_wf_gen, optimal_dem, optimal_elec, 6)
# Supply_hour, Demands_hour, optimal_sup_hour, optimal_dem_hour = Right_order(Generators_hour, Wind_Farms_hour, Demands_hour, optimal_conv_gen_hour, optimal_wf_gen_hour, optimal_dem_hour)
# clearing_price = Single_hour_price(Supply_hour, Demands_hour, optimal_sup_hour, optimal_dem_hour)
# Single_hour_plot(Supply_hour, Demands_hour, clearing_price, optimal_sup_hour, optimal_dem_hour, "Copper_plate_hour_"+str(1))

#####################################################
""" Visualize wind farm and electrolyzer capacity"""
#####################################################

# You can launch this file and will have two dataframe Electrolizer_1 adnd Electrolizer_2 with different columns :
    # 'Hour' with the hour of the day
    # 'Wind farm capacity' with the total wind electrical capacity
    # 'Electrolyzer capacity' the capacity the electrolyzer is consuming at that time
    # 'Grid provided capacity' the power sent to the grid at that hour

# You can find the total demand per hour in the Dataframe Demand_total with columns :
    # 'Hour' with the hour of the day
    # 'Demand' with the total demand at that hour
    

def plot_electrolyzer(Electrolizer_1, Electrolizer_2, Demand_total):
    fig, ax1 = plt.subplots(figsize=(20, 12))
    plt.rcParams["font.size"] = 16
    
    # generate color palette
    colors_wf = sns.color_palette('flare', 24)
    colors_elec = sns.color_palette('Blues', 24)
    
    # choose colors out of palette
    wf_1 = colors_wf[10]
    wf_2 = colors_wf[20]
    
    elec_1=colors_elec[10]
    elec_2=colors_elec[20]
    
    hours = np.arange(1, 25)
    bar_width = 0.35
    
    # create bar diagram
    ax1.bar(hours - bar_width/2, Electrolizer_1['Wind farm capacity'],bar_width, fill = True, color = wf_1,label='Wind farm 1')
    ax1.bar(hours - bar_width/2, Electrolizer_1['Electrolyzer capacity'], bar_width, fill = True, color = elec_1, bottom=0, label='Electrolyzer 1')
    
    ax1.bar(hours + bar_width/2, Electrolizer_2['Wind farm capacity'], bar_width, fill = True, color = wf_2 ,label='Wind farm 2')
    ax1.bar(hours + bar_width/2, Electrolizer_2['Electrolyzer capacity'], bar_width, fill = True, color = elec_2, bottom=0, label='Electrolyzer 2')
    
    # create second axis with line diagram
    ax2 = ax1.twinx()
    ax2.set_ylim(0, 2700)
    ax2.plot(hours, Demand_total['Demand'], color='darkgreen', linestyle='-', marker='o', label='Total demand')
    ax2.set_ylabel('total demand (MW)')
    
    # Legend for second x axis
    ax2.legend(loc='upper right')
    
    # Set x-axis limits and ticks
    ax1.set_ylim(0, 200)
    ax1.set_xlim(0.5, 24.5)
    ax1.set_xticks(np.arange(1, 25, step=1))
    
    # x and y axis label
    ax1.set_xlabel('hours')
    ax1.set_ylabel('capacity (MW)')
    ax1.legend(loc='upper left')
    
    plt.show()


#######################
""" Global function """
#######################

def Copper_plate_multi_hour(Generators, Wind_Farms, Demands) :
    optimal_conv_gen, optimal_wf_gen, optimal_dem, optimal_elec, equilibrium_prices = Multiple_hour_optimization(Generators, Wind_Farms, Demands)
    Electrolizer_1 = pd.DataFrame(columns=['Hour', 'Wind farm capacity', 'Electrolyzer capacity', 'Grid provided capacity'])
    Electrolizer_2 = pd.DataFrame(columns=['Hour', 'Wind farm capacity', 'Electrolyzer capacity', 'Grid provided capacity'])
    Demand_total = pd.DataFrame(columns=['Hour', 'Demand'])
    all_dataframes = []
    for i in range(1,25) :
        Generators_hour, Wind_Farms_hour, Demands_hour, optimal_conv_gen_hour, optimal_wf_gen_hour, optimal_dem_hour, optimal_elec_hour, equilibrium_price = Select_one_hour(Generators, Demands, optimal_conv_gen, optimal_wf_gen, optimal_dem, optimal_elec, equilibrium_prices, i)
        Supply_hour, Demands_hour, optimal_sup_hour, optimal_dem_hour = Right_order(Generators_hour, Wind_Farms_hour, Demands_hour, optimal_conv_gen_hour, optimal_wf_gen_hour, optimal_dem_hour)
        Single_hour_plot(Supply_hour, Demands_hour, equilibrium_price, optimal_sup_hour, optimal_dem_hour, "Copper_plate_hour_"+str(i))
        
        # Add a new row to the dataframe
        Electrolizer_1.loc[len(Electrolizer_1.index)] = [i, Wind_Farms_hour['Capacity'][index_elec[0]], optimal_elec_hour[0], Wind_Farms_hour['Optimal'][index_elec[0]]]  
        Electrolizer_2.loc[len(Electrolizer_2.index)] = [i, Wind_Farms_hour['Capacity'][index_elec[1]], optimal_elec_hour[1], Wind_Farms_hour['Optimal'][index_elec[1]]]  
        Demand_total.loc[len(Demand_total.index)] = [i, Demands_hour['Optimal'].sum()]
        
        # Output of all the dataframe
        all_dataframes.append(Supply_hour)
    
    # Output of the excel
    final_dataframe = pd.concat(all_dataframes, ignore_index=True)
    final_dataframe.to_excel('output_file.xlsx', index=False)
        
    return(Electrolizer_1, Electrolizer_2, Demand_total)
        
Electrolizer_1, Electrolizer_2, Demand_total = Copper_plate_multi_hour(Generators, Wind_Farms, Demands)
plot_electrolyzer(Electrolizer_1, Electrolizer_2, Demand_total)


    


















