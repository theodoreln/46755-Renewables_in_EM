
from Data import Generators, Demands, Wind_Farms, Transmission, Nodes, Line_susceptance, Line_capacity
from Step_1 import Single_hour_plot, Commodities
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import gurobipy as gp
import copy
import os
from math import pi
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
Hydro_demand = 0
#Number of nodes in network
n_nodes=24


# Taking the hour supply and load information and optimizing on 24 hours
def Nodal_optimization(Generators, Wind_Farms, Demands, Nodes, Transmission) :
    # Global variables
    # global optimal_conv_gen, optimal_wf_gen, optimal_dem, optimal_elec
    #Number of units to take into account (based on data)
    # Number of conventional generation units
    n_gen = len(Generators)
    # Number of wind farms generation units
    n_wf = len(Wind_Farms)
    # Number of demand units
    n_dem = len(Demands)
    #number of nodes of the network
    n_nodes=len(Nodes)
    
    
    #Optimization part
    # Create the model
    model = gp.Model()
    #Initialize the decision variables, each one of them is now time dependant
    # Capacities provided by conventional generation units
    var_conv_gen = model.addVars(n_gen, n_hour, lb=-gp.GRB.INFINITY, ub=gp.GRB.INFINITY, vtype=GRB.CONTINUOUS, name='conv_gen')
    # Capacities provided by wind farms
    var_wf_gen = model.addVars(n_wf, n_hour, lb=-gp.GRB.INFINITY, ub=gp.GRB.INFINITY, vtype=GRB.CONTINUOUS, name='wf_gen')
    # Demand capacities fullfilled
    var_dem = model.addVars(n_dem, n_hour, lb=-gp.GRB.INFINITY, ub=gp.GRB.INFINITY, vtype=GRB.CONTINUOUS, name='dem')
    # Capacities provided to the electrolyzer
    var_elec = model.addVars(len(index_elec), n_hour, lb=-gp.GRB.INFINITY, ub=gp.GRB.INFINITY, vtype=GRB.CONTINUOUS, name='elec')
    #Load angle for each node
    var_theta = model.addVars(n_nodes, n_hour, lb=-pi, ub=pi, vtype=GRB.CONTINUOUS, name='Theta')
    
    # Add the objective function to the model, sum on every t, separation of the conventional generation units and the wind farms
    model.setObjective(gp.quicksum(gp.quicksum(Demands['Offer price'][d][t]*var_dem[d,t] for d in range(n_dem))-gp.quicksum(Generators['Bid price'][g]*var_conv_gen[g,t] for g in range(n_gen))-gp.quicksum(Wind_Farms['Bid price'][wf][t]*var_wf_gen[wf,t] for wf in range(n_wf)) for t in range(n_hour)), GRB.MAXIMIZE)
    
    #Add constraints to the model
    # Constraint that must be fullfilled for every time ste
    for t in range(n_hour) :
        for g in range(n_gen) :
            # The provided capacity of a conventional unit should not be more than the maximum capacity
            name = f"Generator {g+1} : dual "
            model.addConstr(var_conv_gen[g,t] <= Generators['Capacity'][g], name=name+'up ')
            model.addConstr(var_conv_gen[g,t] >= 0, name=name+'down ')
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
        
    #Constraint on line capacity. Must be fulfilled for each time and for each line
    for t in range(n_hour):
        # for n in range(1,25):
        #     L = Nodes[n]["L"] #take the line of information corresponding to each node
        #     for l in L:
        #         node_to, susceptance,capacity=l
        #         name = f'dual {t+1}, {n}, {node_to} '
        #         model.addConstr(susceptance*(var_theta[n-1,t]-var_theta[node_to-1,t]) <= capacity, name=name+'up') #not sure abt the syntax in this one. upper limit on capacity
        #         model.addConstr(susceptance*(var_theta[n-1,t]-var_theta[node_to-1,t]) >= -capacity, name=name+'down') #lower limit on capacity
        # Another version of the code above for the transmission constraints
        # for tr in range(len(Transmission)) :
        #     nf, nt, sus, cap = Transmission['From'][tr], Transmission['To'][tr], Transmission['Susceptance'][tr], Transmission['Capacity'][tr]
        #     print(nf,nt,sus,cap)
        #     name = f'dual {t+1}, {nf}, {nt} '
        #     model.addConstr(sus*(var_theta[nf-1,t]-var_theta[nt-1,t]) <= cap, name=name+'up')
        #     model.addConstr(sus*(var_theta[nf-1,t]-var_theta[nt-1,t]) >= -cap, name=name+'down')
        # Version with dataframe
        for n in range(24) :
            for m in range(24) :
                name = f'dual {t+1}, {n+1}, {m+1} '
                model.addConstr(Line_susceptance[n,m]*(var_theta[n,t]-var_theta[m,t]) <= Line_capacity[n,m], name=name+'up')
                model.addConstr(Line_susceptance[n,m]*(var_theta[n,t]-var_theta[m,t]) >= -Line_capacity[n,m], name=name+'down')
            
    #Set theta on node 1 to 0 for all t
    for t in range(n_hour):
        model.addConstr(var_theta[0,t] == 0)
        
    #Power balance for each node
    for t in range(n_hour):
        for n in range(1,25):
            constr_name = f"Power balance hour {t+1} in node {n}"
            L = Nodes[n]["L"] #take the line of information corresponding to each node
            D = Nodes[n]["D"] #take the line of information corresponding to each node
            G = Nodes[n]["G"] #take the line of information corresponding to each node
            W = Nodes[n]["W"] #take the line of information corresponding to each node
            # for j in range(len(L)) :
            #     print(n, L[j][0], L[j][1], L[j][2])
                
            model.addConstr(gp.quicksum(var_dem[d-1,t] for d in D) 
                - gp.quicksum(var_conv_gen[g-1,t] for g in G) 
                - gp.quicksum(var_wf_gen[wf-1,t] for wf in W) 
                + gp.quicksum(Line_susceptance[n-1,m]*(var_theta[n-1,t]-var_theta[m,t]) for m in range(24)) == 0, name=constr_name)

    #Solve the problem
    model.optimize()
    
    #Get the optimal values
    if model.status == GRB.OPTIMAL:
        # Create a list to store the optimal values of the variables
        optimal_conv_gen = [[round(var_conv_gen[g,t].X,2) for g in range(n_gen)] for t in range(n_hour)]
        optimal_wf_gen = [[round(var_wf_gen[wf,t].X,2) for wf in range(n_wf)] for t in range(n_hour)]
        optimal_dem = [[round(var_dem[d,t].X,2) for d in range(n_dem)] for t in range(n_hour)]
        optimal_elec = [[round(var_elec[e,t].X,2) for e in range(2)] for t in range(n_hour)]
        optimal_theta = [[round(var_theta[n-1,t].X,2) for n in range(1,25)] for t in range(n_hour)] 
        equilibrium_prices = np.zeros((n_nodes, n_hour), dtype=np.float64)
        # Take the values of the quantity exchanged on the transmission lines 
        quantity_trade = np.zeros((n_nodes, n_nodes, n_hour), dtype=np.float64)
        for t in range(n_hour) :
            for n in range(n_nodes) :
                L = Nodes[n+1]["L"] 
                for l in L :
                    node_to, susceptance, capacity = l
                    quantity_trade[n, node_to-1, t] = round(susceptance*(var_theta[n,t].X-var_theta[node_to-1,t].X),2)
                    quantity_trade[node_to-1, n, t] = round(-susceptance*(var_theta[n,t].X-var_theta[node_to-1,t].X),2)
        # Value of the optimal objective
        optimal_obj = model.ObjVal
        
        """ KKTs output """
        # Write KKT's condition in a text file
        # Define the folder path and the file name
        folder_path = "results/nodal"
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
            # To help identify to which zones correspond the price
            ind = n_nodes
            temps = -1
            for c in model.getConstrs():
                file.write(f"{c.ConstrName}: {c.Pi} : {c.Sense} \n")
                if 'Power balance hour' in c.constrName :
                    remind = ind % n_nodes
                    if remind == 0:
                        temps += 1
                    equilibrium_prices[remind,temps] = c.Pi
                    ind += 1
        
    else:
        print("Optimization did not converge to an optimal solution.")
    
    return(optimal_conv_gen, optimal_wf_gen, optimal_dem, optimal_elec, optimal_theta, quantity_trade, equilibrium_prices)

    
##########################################################
""" Written output of transmission decision and prices """
##########################################################
                
def Nodal_prices(Nodes, Generators, Wind_Farms, Demands, optimal_conv_gen, optimal_wf_gen, optimal_dem, optimal_elec, quantity_trade, equilibrium_prices) :
    # Number of zones
    n_nodes = len(Nodes)
    
    #Write output in a text file
    # Define the folder path and the file name
    folder_path = "results/nodal"
    file_name = "Transmission and prices.txt"
    
    # Create the folder if it doesn't exist
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    
    # Combine the folder path and file name
    file_path = os.path.join(folder_path, file_name)
    
    # Open the file in write mode
    with open(file_path, 'w') as file:
        file.write("Transmission and equilibrium prices for the nodal optimization :")
        file.write("\n\n")
        
        for t in range(n_hour) :
            file.write(f"\n------------------------------------------------")
            file.write(f"\n-------------------- Hour {t+1} --------------------")
            file.write("\n\n")
            
            for n in range(n_nodes) :
                file.write(f"---------- Node {n+1} ----------\n")
                file.write(f"*Equilibrium price : {round(equilibrium_prices[n,t],2)}*\n")
                
                D = Nodes[n+1]["D"] #take the demand information corresponding to each node
                G = Nodes[n+1]["G"] #take the generator information corresponding to each node
                W = Nodes[n+1]["W"] #take the wind farm information corresponding to each node
                Gen = round(sum([optimal_conv_gen[t][g-1] for g in G]) + sum([optimal_wf_gen[t][w-1] for w in W]),2)
                Gen_tot = round(sum([Generators['Capacity'][g-1] for g in G]) + sum([Wind_Farms['Capacity'][w-1][t] for w in W]) - sum([optimal_elec[t][index_elec[w-1]] for w in W if w-1 in index_elec]),2)
                Dem = round(sum([optimal_dem[t][d-1] for d in D]),2)
                Dem_tot = round(sum([Demands["Load"][d-1][t] for d in D]),2)
                
                file.write(f"\nPower generation : {Gen}/{Gen_tot} MW\n")
                for g in G :
                    file.write(f"      {Generators['Name'][g-1]} : {optimal_conv_gen[t][g-1]}/{Generators['Capacity'][g-1]} MW\n")
                for w in W :
                    if w-1 in index_elec :
                        file.write(f"      {Wind_Farms['Name'][w-1]} : {optimal_wf_gen[t][w-1]}/{round(Wind_Farms['Capacity'][w-1][t] - optimal_elec[t][index_elec[w-1]],2)} MW\n")
                        file.write(f"          Electrolyzer : {optimal_elec[t][index_elec[w-1]]} MW\n")
                    else :
                        file.write(f"      {Wind_Farms['Name'][w-1]} : {optimal_wf_gen[t][w-1]}/{Wind_Farms['Capacity'][w-1][t]} MW\n")
                file.write(f"\nPower demand : {Dem}/{Dem_tot} MW\n")
                for d in D :
                    if optimal_dem[t][d-1] != Demands['Load'][d-1][t] :
                        file.write(f"      {Demands['Name'][d-1]} : {optimal_dem[t][d-1]}/{Demands['Load'][d-1][t]} MW ----> Demand not fulfilled\n")
                    else :
                        file.write(f"      {Demands['Name'][d-1]} : {optimal_dem[t][d-1]}/{Demands['Load'][d-1][t]} MW\n")
                
                file.write("\n")
                for m in range(n_nodes) :
                    if m+1 in [Nodes[n+1]["L"][l][0] for l in range(len(Nodes[n+1]["L"]))]:
                        quantity = quantity_trade[n,m,t]
                        if quantity > 0 :
                            index_lim = next((index for index, sublist in enumerate(Nodes[n+1]["L"]) if sublist[0]==m+1), None)
                            if quantity == Nodes[n+1]['L'][index_lim][2] :
                                file.write(f"Transmission with node {m+1} : {quantity}/{Nodes[n+1]['L'][index_lim][2]} MW ----> Line saturated !!!\n")
                            else :
                                file.write(f"Transmission with node {m+1} : {quantity}/{Nodes[n+1]['L'][index_lim][2]} MW\n")
                        else :
                            index_lim = next((index for index, sublist in enumerate(Nodes[n+1]["L"]) if sublist[0]==m+1), None)
                            if quantity == -Nodes[n+1]['L'][index_lim][2] :
                                file.write(f"Transmission with node {m+1} : {quantity}/{-Nodes[n+1]['L'][index_lim][2]} MW ----> Line saturated !!!\n")
                            else :
                                file.write(f"Transmission with node {m+1} : {quantity}/{-Nodes[n+1]['L'][index_lim][2]} MW\n")
                file.write("\n")

optimal_conv_gen, optimal_wf_gen, optimal_dem, optimal_elec, optimal_theta, quantity_trade, equilibrium_prices = Nodal_optimization(Generators, Wind_Farms, Demands, Nodes, Transmission)
Nodal_prices(Nodes, Generators, Wind_Farms, Demands, optimal_conv_gen, optimal_wf_gen, optimal_dem, optimal_elec, quantity_trade, equilibrium_prices)



node_prices = [equilibrium_prices[0,:].tolist()]  # Change the index to match the desired node number

#change the transmission capacity
Line_capacity[15-1, 21-1] = 400
Line_capacity[21-1, 15-1] = 400
Line_capacity[14-1, 16-1] = 250
Line_capacity[16-1, 14-1] = 250
Line_capacity[13-1, 23-1] = 250
Line_capacity[23-1, 13-1] = 250

#run again the function to get equilibrium price. Comment next line to obtain Objective value for non-constrained model
optimal_conv_gen, optimal_wf_gen, optimal_dem, optimal_elec, optimal_theta, quantity_trade, equilibrium_prices = Nodal_optimization(Generators, Wind_Farms, Demands, Nodes, Transmission)
Nodal_prices(Nodes, Generators, Wind_Farms, Demands, optimal_conv_gen, optimal_wf_gen, optimal_dem, optimal_elec, quantity_trade, equilibrium_prices)


#store the prices of the nodes that I want 
node_prices.append(equilibrium_prices[7,:].tolist())  # Node 21 (adjusted index)
node_prices.append(equilibrium_prices[14,:].tolist())  # Node 15 (adjusted index)
node_prices.append(equilibrium_prices[20,:].tolist())  # Node 21 (adjusted index)

node_numbers = [1,8, 15, 21]  # Node numbers corresponding to each sublist



def plot_congested1_nodal(node_prices, node_numbers, save_path=None):
    fig, ax1 = plt.subplots(figsize=(20, 12))
    plt.rcParams["font.size"] = 16
    
    # Set x-axis limits and ticks
    hours = np.arange(1, 25)
    
    list_names=['Non-congested', 'Node 8', 'Node 15', 'Node 21']
    
    # Plot step line for node prices
    for i, prices in enumerate(node_prices):
        ax1.step(hours, prices, where='post', linestyle='-',linewidth=2.5, label=list_names[i])
    
    # Set x and y axis label
    ax1.set_xlabel('Hours')
    ax1.set_ylabel('Nodal Market Price ($/MWh)')
    ax1.legend(loc='upper left')
    
    # Save the plot as an image if save_path is provided
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


# Define the path where you want to save the plot
folder_path = "plots/step4_nodal"
file_name = "congested1_plot.png"
save_path = os.path.join(folder_path, file_name)

# Create the folder if it doesn't exist
if not os.path.exists(folder_path):
    os.makedirs(folder_path)

# Plot the equilibrium prices for all nodes and save the plot as an image
plot_congested1_nodal(node_prices, node_numbers, save_path=save_path)

# Call the function to write equilibrium prices to a file
#Nodal_prices(Nodes, Generators, Wind_Farms, Demands, optimal_conv_gen, optimal_wf_gen, optimal_dem, optimal_elec, quantity_trade, equilibrium_prices)
