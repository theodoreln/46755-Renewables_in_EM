from Data import Generators, Generators_reserve, Demands, Wind_Farms
from Step_1 import Single_hour_plot, Commodities, Single_hour_optimization
from Step_2 import Multiple_hour_optimization, Select_one_hour, Right_order, Copper_plate_multi_hour
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

# Reserve optimization problem on 24 hours
def Reserve_optimization(Generators_reserve, Demands) :
    # Global variables
    # global optimal_conv_gen, optimal_wf_gen, optimal_dem, optimal_elec
    #Number of units to take into account (based on data)
    # Number of conventional generation units
    n_gen = len(Generators_reserve)
    # Number of demand units
    n_dem = len(Demands)
    # Total of demands per hour
    Demands_sum = []
    for t in range(n_hour) :
        sum_hour = 0
        for i in range(len(Demands)) :
            sum_hour += Demands['Load'][i][t]
        Demands_sum.append(sum_hour)
    
    #Optimization part
    # Create the model
    model = gp.Model()
    #Initialize the decision variables, each one of them is now time dependant
    # Capacities provided by conventional generation units
    var_up = model.addVars(n_gen, n_hour, lb=-gp.GRB.INFINITY, vtype=GRB.CONTINUOUS, name='conv_gen')
    # Capacities provided by wind farms
    var_down = model.addVars(n_gen, n_hour, lb=-gp.GRB.INFINITY, vtype=GRB.CONTINUOUS, name='wf_gen')
    
    # Add the objective function to the model, sum on every t, separation of the conventional generation units and the wind farms
    model.setObjective(gp.quicksum(gp.quicksum(Generators_reserve['Up reserve price'][g]*var_up[g,t] for g in range(n_gen)) + gp.quicksum(Generators_reserve['Down reserve price'][g]*var_down[g,t] for g in range(n_gen)) for t in range(n_hour)), GRB.MINIMIZE)
    
    #Add constraints to the model
    # Quantity reserved = Quantity needed for every time step t
    for t in range(n_hour):
        constr_name = f"Reserve up need hour {t+1}"
        model.addConstr(gp.quicksum(var_up[g,t] for g in range(n_gen)) == Demands_sum[t]*0.15, name = constr_name)
        constr_name = f"Reserve down need hour {t+1}"
        model.addConstr(gp.quicksum(var_down[g,t] for g in range(n_gen)) == Demands_sum[t]*0.10, name = constr_name)
    
    # Constraint that must be fullfilled for every time step
    for t in range(n_hour) :
        # Maximum and minimum for reserve + sum should not be higher than max capacity
        for g in range(n_gen) :
            model.addConstr(var_up[g,t] >= 0)
            model.addConstr(var_up[g,t] <= Generators_reserve['Maximum up reserve'][g])
            model.addConstr(var_down[g,t] >= 0)
            model.addConstr(var_down[g,t] <= Generators_reserve['Maximum down reserve'][g])
            model.addConstr(var_up[g,t] + var_down[g,t] <= Generators_reserve['Capacity'][g])
            
    
    #Solve the problem
    model.optimize()
    
    #Get the optimal values
    if model.status == GRB.OPTIMAL:
        # Create a list to store the optimal values of the variables
        optimal_up = [[round(var_up[g,t].X,2) for g in range(n_gen)] for t in range(n_hour)]
        optimal_down = [[round(var_down[g,t].X,2) for g in range(n_gen)] for t in range(n_hour)]
        up_prices = []
        down_prices = []
        # Value of the optimal objective
        optimal_obj = model.ObjVal
        
        """ KKTs output """
        # Write KKT's condition in a text file
        # Define the folder path and the file name
        folder_path = "results/reserve"
        file_name = "KKTs.txt"
        
        # Create the folder if it doesn't exist
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        
        # Combine the folder path and file name
        file_path = os.path.join(folder_path, file_name)
        
        # Open the file in write mode
        with open(file_path, 'w') as file:
            file.write("KKTs for the regulation capacity reservation :")
            file.write("\n\n")
            # Write lines to the file
            for c in model.getConstrs():
                file.write(f"{c.ConstrName}: {c.Pi} : {c.Sense} \n")
                if 'Reserve up' in c.constrName :
                    up_prices.append(c.Pi)
                if 'Reserve down' in c.constrName :
                    down_prices.append(c.Pi)
        
    else:
        print("Optimization did not converge to an optimal solution.")
    
    return(optimal_up, optimal_down, up_prices, down_prices)


# Optimization problem for the day-ahead problem
def DA_optimization(Generators, Wind_Farms, Demands, optimal_up, optimal_down) :
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
    var_conv_gen = model.addVars(n_gen, n_hour, lb=-gp.GRB.INFINITY, vtype=GRB.CONTINUOUS, name='conv_gen')
    # Capacities provided by wind farms
    var_wf_gen = model.addVars(n_wf, n_hour, lb=-gp.GRB.INFINITY, vtype=GRB.CONTINUOUS, name='wf_gen')
    # Demand capacities fullfilled
    var_dem = model.addVars(n_dem, n_hour, lb=-gp.GRB.INFINITY, vtype=GRB.CONTINUOUS, name='dem')
    # Capacities provided to the electrolyzer
    var_elec = model.addVars(len(index_elec), n_hour, lb=-gp.GRB.INFINITY, vtype=GRB.CONTINUOUS, name='elec')
    
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
            # The provided capacity of a conventional unit should not be more than the maximum capacity minus the reserved capacity
            model.addConstr(var_conv_gen[g,t] <= Generators['Capacity'][g] - optimal_up[t][g])
            model.addConstr(var_conv_gen[g,t] >= optimal_down[t][g])
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

optimal_up, optimal_down, up_prices, down_prices = Reserve_optimization(Generators_reserve, Demands)
optimal_conv_gen, optimal_wf_gen, optimal_dem, optimal_elec, equilibrium_prices = DA_optimization(Generators, Wind_Farms, Demands, optimal_up, optimal_down)



########################
""" Results analysis """
########################

# Function to put all results on a text file 
def Results_reserve(Generators_reserve, Wind_Farms, Demands, optimal_up, optimal_down, up_prices, down_prices, optimal_conv_gen, optimal_wf_gen, optimal_dem, optimal_elec, equilibrium_prices) :
    #Write output in a text file
    # Define the folder path and the file name
    folder_path = "results/reserve"
    file_name = "Capacity reserved and results of the day-ahead market.txt"
    
    # Create the folder if it doesn't exist
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    
    # Combine the folder path and file name
    file_path = os.path.join(folder_path, file_name)
    
    # Open the file in write mode
    with open(file_path, 'w') as file:
        file.write("Capacity reserved and results of the day-ahead market :")
        file.write("\n\n")
        
        for t in range(n_hour) :
            file.write(f"\n------------------------------------------------")
            file.write(f"\n-------------------- Hour {t+1} --------------------")
            file.write("\n\n")
            
            file.write(f"---------- Reserve market ----------\n\n")
            
            file.write(f"*Up reserve price : {round(up_prices[t],2)}*\n")
            for g in range(len(Generators_reserve)) :
                if optimal_up[t][g] != 0 :
                    file.write(f"{Generators_reserve['Name'][g]} : {optimal_up[t][g]} MW -> {Generators_reserve['Up reserve price'][g]} €/MW\n")
            
            file.write(f"\n*Down reserve price : {round(down_prices[t],2)}*\n")
            for g in range(len(Generators_reserve)) :
                if optimal_down[t][g] != 0 :
                    file.write(f"{Generators_reserve['Name'][g]} : {optimal_down[t][g]} MW -> {Generators_reserve['Down reserve price'][g]} €/MW\n")
                    
            file.write('\n')
            file.write(f"---------- Day-ahead market ----------\n\n")

            file.write(f"*Equilibrium price : {round(equilibrium_prices[t],2)}*\n")
            for g in range(len(Generators_reserve)) :
                file.write(f"{Generators_reserve['Name'][g]} : {optimal_conv_gen[t][g]}/{Generators_reserve['Capacity'][g]} MW -> {Generators_reserve['Bid price'][g]} €/MW\n")
            for w in range(len(Wind_Farms)) :
                file.write(f"{Wind_Farms['Name'][w]} : {optimal_wf_gen[t][w]}/{Wind_Farms['Capacity'][w][t]} MW\n")
            file.write('\n')
            for d in range(len(Demands)) :
                file.write(f"{Demands['Name'][d]} : {optimal_dem[t][d]}/{Demands['Load'][d][t]} MW\n")
            
            file.write('\n')

optimal_conv_gen_2, optimal_wf_gen_2, optimal_dem_2, optimal_elec_2, equilibrium_prices_2 = Multiple_hour_optimization(Generators, Wind_Farms, Demands)            
            
# Function to plot the prices of the reserve market and the prices of the day-ahead market
def Plot_prices(up_prices, down_prices, equilibrium_prices, equilibrium_prices_2) :
    t = list(range(1,25))

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10))
    plt.rcParams["font.size"] = 20
    
    ax1.step(t, up_prices, 'm--', label='Up reserve prices')
    ax1.step(t, down_prices, 'b--', label='Down reserve prices')
    ax1.set_ylabel('Market price [$/MW]')
    ax1.set_ylim(4, 16.5)
    ax1.legend(loc=8)
    
    ax2.step(t, equilibrium_prices_2, 'r-', label='Day-ahead prices without reserve market')
    ax2.step(t, equilibrium_prices, 'k-', label='Day-ahead prices with reserve market')
    ax2.set_xlabel('Hours')
    ax2.set_ylabel('Market price [$/MWh]')
    ax2.set_ylim(2.5, 11.5)
    ax2.legend(loc=8)
    
    output_folder = os.path.join(os.getcwd(), 'plots\\step_6')
    pdf_name = 'Prices reserve and da.pdf'
    pdf_filename = os.path.join(output_folder, pdf_name)
    plt.savefig(pdf_filename,  bbox_inches='tight')
    plt.show()
                
            
# Function to plot the social welfare, with and without reserve market
def Plot_SW(Generators_reserve, Wind_Farms, Demands, optimal_conv_gen, optimal_wf_gen, optimal_dem, optimal_up, optimal_down, equilibrium_prices, up_prices, down_prices, optimal_conv_gen_2, optimal_wf_gen_2, optimal_dem_2, optimal_elec_2, equilibrium_prices_2) :
    
    list_hour = list(range(1,25))
    
    # Computing the social welfare hour by hour with the reserve market 
    SW_DA_res = [0]*24
    for t in range(24) :
        for w in range(len(Wind_Farms)) :
            SW_DA_res[t] += equilibrium_prices[t] * optimal_wf_gen[t][w]
        for g in range(len(Generators_reserve)) :
            SW_DA_res[t] += (equilibrium_prices[t] - Generators_reserve['Bid price'][g]) * optimal_conv_gen[t][g]
        for d in range(len(Demands)) :
            SW_DA_res[t] += (Demands['Offer price'][d][t] - equilibrium_prices[t]) * optimal_dem[t][d]
            
    # Cost of the reserve market
    Reserve_cost = [0]*24
    for t in range(24) :
        Reserve_cost[t] += up_prices[t]*sum(optimal_up[t]) + down_prices[t]*sum(optimal_down[t])
    
    # Social welfare with cost of reserve market
    SW_DA_resp = SW_DA_res.copy()
    for t in range(24) :
        SW_DA_resp[t] += -Reserve_cost[t]          
            
    # Social welfare without taking into account the reserve market
    SW_DA = [0]*24
    for t in range(24) :
        for w in range(len(Wind_Farms)) :
            SW_DA[t] += equilibrium_prices_2[t] * optimal_wf_gen_2[t][w]
        for g in range(len(Generators_reserve)) :
            SW_DA[t] += (equilibrium_prices_2[t] - Generators_reserve['Bid price'][g]) * optimal_conv_gen_2[t][g]
        for d in range(len(Demands)) :
            SW_DA[t] += (Demands['Offer price'][d][t] - equilibrium_prices_2[t]) * optimal_dem_2[t][d]
    
    plt.figure(figsize = (15, 10))
    plt.rcParams["font.size"] = 20
    plt.plot(list_hour, SW_DA, 'r-', label='Social welfare without reserve market')
    # plt.plot(list_hour, SW_DA_res, 'k--', label='Social welfare without cost of reserve market')
    plt.plot(list_hour, SW_DA_resp, 'k-', label='Social welfare with reserve market')
    plt.plot(list_hour, Reserve_cost, 'b-', label='Cost of the reserve market')
    plt.xlabel('Hours')
    plt.ylabel('Social welfare [$]')
    plt.ylim(0, 89900)
    plt.legend(loc=2)
    output_folder = os.path.join(os.getcwd(), 'plots\\step_6')
    pdf_name = 'Social welfare comparison.pdf'
    pdf_filename = os.path.join(output_folder, pdf_name)
    plt.savefig(pdf_filename,  bbox_inches='tight')
    plt.show()
    

# Function to compute the benefit of every generators
def Benefits(Generators_reserve, Wind_Farms, Demands, optimal_conv_gen, optimal_wf_gen, optimal_dem, optimal_elec, equilibrium_prices, optimal_up, optimal_down, up_prices, down_prices, option) :
    #Write output in a text file
    # Define the folder path and the file name
    folder_path = "results/reserve"
    file_name = "Benefits with reserve market.txt"
    
    # Create the folder if it doesn't exist
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    
    # Combine the folder path and file name
    file_path = os.path.join(folder_path, file_name)
    
    # Compute the benefits of generators on the day
    Gen_ben = [0]*len(Generators_reserve)
    Wf_ben = [0]*len(Wind_Farms)
    for t in range(24) :
        for g in range(len(Generators_reserve)) :    
            Gen_ben[g] += (equilibrium_prices[t] - Generators_reserve['Bid price'][g]) * optimal_conv_gen[t][g]
            if option == 1 :
                Gen_ben[g] += up_prices[t]*optimal_up[t][g] + down_prices[t]*optimal_down[t][g]
        for w in range(len(Wind_Farms)) :
            Wf_ben[w] += equilibrium_prices[t] * optimal_wf_gen[t][w]
    
    
    # Open the file in write mode
    with open(file_path, 'w') as file:
        file.write("Benefit of every generators")
        file.write("\n\n")
        
        for w in range(len(Wind_Farms)) :
            file.write(f"{Wind_Farms['Name'][w]} : {round(Wf_ben[w],2)} $\n")
        file.write("\n\n")
        for g in range(len(Generators_reserve)) :
            file.write(f"{Generators_reserve['Name'][g]} : {round(Gen_ben[g],2)} $\n")
            
        
Results_reserve(Generators_reserve, Wind_Farms, Demands, optimal_up, optimal_down, up_prices, down_prices, optimal_conv_gen, optimal_wf_gen, optimal_dem, optimal_elec, equilibrium_prices)
Benefits(Generators_reserve, Wind_Farms, Demands, optimal_conv_gen, optimal_wf_gen, optimal_dem, optimal_elec, equilibrium_prices, optimal_up, optimal_down, up_prices, down_prices, 1)
# Benefits(Generators_reserve, Wind_Farms, Demands, optimal_conv_gen_2, optimal_wf_gen_2, optimal_dem_2, optimal_elec_2, equilibrium_prices_2, optimal_up, optimal_down, up_prices, down_prices, 0)
Plot_prices(up_prices, down_prices, equilibrium_prices, equilibrium_prices_2)
Plot_SW(Generators_reserve, Wind_Farms, Demands, optimal_conv_gen, optimal_wf_gen, optimal_dem, optimal_up, optimal_down, equilibrium_prices, up_prices, down_prices, optimal_conv_gen_2, optimal_wf_gen_2, optimal_dem_2, optimal_elec_2, equilibrium_prices_2)
# Plotting the market equilibrium with the reserve market
# for t in range(1,25) :
#     Generators_hour, Wind_Farms_hour, Demands_hour, optimal_conv_gen_hour, optimal_wf_gen_hour, optimal_dem_hour, optimal_elec_hour, equilibrium_price = Select_one_hour(Generators, Demands, optimal_conv_gen, optimal_wf_gen, optimal_dem, optimal_elec, equilibrium_prices, t)
#     Supply_hour, Demands_hour, optimal_sup_hour, optimal_dem_hour = Right_order(Generators_hour, Wind_Farms_hour, Demands_hour, optimal_conv_gen_hour, optimal_wf_gen_hour, optimal_dem_hour)
#     Single_hour_plot(Supply_hour, Demands_hour, equilibrium_price, optimal_sup_hour, optimal_dem_hour, f"Day-ahead market hour with reserve market {t}")

            
            
            
            
            
            
            
            
            
            
            
            

