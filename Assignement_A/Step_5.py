from Data import Generators, Demands, Wind_Farms
from Step_1 import Single_hour_plot, Commodities, Single_hour_optimization
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import gurobipy as gp
import copy
import os
GRB = gp.GRB

# This function solves the balacing market problem for a single hour and it being called inside the global function 'Balancing_market_clearing'
# It takes as inputs:
    # The selected hour in 'b_hour'
    # The dataframe 'Generators_up' with the information about conventional generators that can provide up regulation in BM 
    # The dataframe 'Generators_dw' with the information about conventional generators that can provide down regulation in BM
    # The system's deviation from DA scheduled production program, after implementing the changes of WFs production and outages of CGs
# And gives as an output information:
    # The cost of the system : 'optimal_obj'
    # The decision of generators providing up regulation in a list : 'optimal_gen_up'
    # The decision of generators providing down regulation in a list : 'optimal_gen_dw'
    # The BM's equilibrium price obtained from the dual variable : 'balancing_price'

def Single_hour_balancing(b_hour,Generators_up, Generators_dw, Imbalance) :
    # Global variables
    global optimal_gen_up, optimal_gen_dw
    # Numbers of generators and demanding units
    n_gen_up = len(Generators_up)
    n_gen_dw = len(Generators_dw)
    
    #Optimization part
    # Create the model
    model = gp.Model()
    # Initialize the decision variables
    var_gen_up = model.addVars(n_gen_up, lb=-gp.GRB.INFINITY, vtype=GRB.CONTINUOUS, name='gen_up')
    var_gen_dw = model.addVars(n_gen_dw, lb=-gp.GRB.INFINITY, vtype=GRB.CONTINUOUS, name='gen_dw')
    # Add the objective function to the model
    model.setObjective(gp.quicksum(Generators_up['Bid price'][i]*var_gen_up[i] for i in range(n_gen_up-1)) + Generators_up['Bid price'].iloc[-1]*var_gen_up.select()[-1] - gp.quicksum(Generators_dw['Bid price'][i]*var_gen_dw[i] for i in range(n_gen_dw)) , GRB.MINIMIZE)
    #Constraint about power balance
    model.addConstr(gp.quicksum(var_gen_up[i] for i in range(n_gen_up)) - gp.quicksum(var_gen_dw[i] for i in range(n_gen_dw)) == Imbalance, name='Power balance')
    # Quantities supplied can't be higher than maximum quantities
    for i in range(n_gen_up) :
        name_constr = f"{Generators_up['Name'][i]}_uplim_{i}"
        model.addConstr(var_gen_up[i] <= Generators_up['Capacity'][i], name=name_constr)
        name_constr = f"{Generators_up['Name'][i]}_downlim_{i}"
        model.addConstr(var_gen_up[i] >= 0.0, name=name_constr)
    
    for i in range(n_gen_dw) :
        name_constr = f"{Generators_dw['Name'][i]}_uplim_{i}"
        model.addConstr(var_gen_dw[i] <= Generators_dw['Capacity'][i], name=name_constr)
        name_constr = f"{Generators_dw['Name'][i]}_downlim_{i}"
        model.addConstr(var_gen_dw[i] >= 0.0, name=name_constr)
    # Solve the problem
    model.optimize()    

    #Get the optimal values
    if model.status == GRB.OPTIMAL:
        # Create a list to store the optimal values of the variables
        optimal_gen_up = [var_gen_up[i].X for i in range(n_gen_up)]
        optimal_gen_dw = [var_gen_dw[i].X for i in range(n_gen_dw)]
        # Value of the optimal objective
        optimal_obj = model.ObjVal
        
        # Print
        print("\n")
        print("Up regulation :")
        for i, value in enumerate(optimal_gen_up):
            print(Generators_up["Name"][i] + f" : {round(value,2)} MW")
        print("\n")
        print("Down regulation :")
        for i, value in enumerate(optimal_gen_dw):
            print(Generators_dw["Name"][i] + f" : {round(value,2)} MW")

        """ KKTs output """
        # Write KKT's condition in a text file
        # Define the folder path and the file name
        folder_path = "results/single_hour"
        file_name = f"balancing_{b_hour}.txt"
        
        # Create the folder if it doesn't exist
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        
        # Combine the folder path and file name
        file_path = os.path.join(folder_path, file_name)
        
        # Open the file in write mode
        with open(file_path, 'w') as file:
            file.write(f"KKTs for the single hour balancing market in hour {b_hour} :")
            file.write("\n\n")
            # Write lines to the file
            for c in model.getConstrs():
                file.write(f"{c.ConstrName}: {c.Pi} : {c.Sense} \n")
                if c.ConstrName == 'Power balance' :
                    balancing_price = c.Pi
                    
        print("\n")
        print(f"Equilibrium price : {balancing_price} $/MWh")

    else:
        print("Optimization did not converge to an optimal solution.")

    # Return the cost and the optimal values
    return(optimal_obj, optimal_gen_up, optimal_gen_dw, balancing_price)


# This function is the global function of the Balance market clearing, for a single houre and with intertemporal constraints, but without network constraints
# It take as inputs for ALL HOURS:
    # The dataframe 'Generators_BM' with the information about conventional generators
    # The dataframe 'Demands_BM' with all the demand to fullfill
    # The dataframe 'Wind_Farms_BM' with the information about wind farms
    # The selected hour in 'hour'
    # The selected conventional generators that fail in a list: 'CG_outage'
    # The selected Wind Farms with lower production than scheduled in a list: 'WF_lower'
    # The selected Wind Farms with higher production than scheduled in a list: 'WF_higher'
    # The selected percentage of lower production for Wind Farms in 'WF_down_imbalance'
    # The selected percentage of higher production for Wind Farms in 'WF_up_imbalance'
# And gives as an output information FOR ONE HOUR the dataframe 'BM_clearing_output',  that contains:
    # The name of conventional generators and wind farms in column named 'Name'
    # The DA scheduled production program, of all generators, in column named 'DA_production'
    # The DA bid price, of all generators, in column named 'DA_bid_price'
    # The BM bid price for up regulation, where zero means the respective generator is not capable of up regulation, in column named 'M_bid_price_up'
    # The BM bid price for down regulation, where zero means the respective generator is not capable of down regulation, in column named 'M_bid_price_dw'
    # The imbalance concerning the DA scheduled production program in column named 'Imbalance'abs
    # The up regulation energy provided in column named 'BM_up'
    # The down regulation energy provided in column named 'BM_dw'abs
    # The total profits of generator, after DA and Balancing market clearings, based on single-pricing model in column named 'single-price'
    # The total profits of generator, after DA and Balancing market clearings, based on dual-pricing model in column named 'dual-price'

def Balancing_market_clearing(Generators_BM, Demands_BM, Wind_Farms_BM, hour, CG_outage, WF_lower, WF_higher, WF_down_imbalance, WF_up_imbalance) :
    for i in range(len(Demands_BM)) :
        Demands_BM.loc[i, 'Load'] = Demands_BM['Load'][i][hour-1]
        Demands_BM.loc[i, 'Offer price'] = Demands_BM['Offer price'][i][hour-1]

    # Adding the wind farms
    for i in range(len(Wind_Farms_BM)) :
        Wind_Farms_BM.loc[i, 'Capacity'] = round(Wind_Farms_BM['Capacity'][i][hour-1],2)
        Wind_Farms_BM.loc[i, 'Bid price'] = Wind_Farms_BM['Bid price'][i][hour-1]
    Generators_BM = pd.concat([Generators_BM, Wind_Farms_BM], axis=0)
    # Order in ascending orders, necessary for the optimization 
    Generators_BM = Generators_BM.sort_values(by=['Bid price','Name'], ascending=[True, 'Wind farm' in Generators_BM['Name'].values]).reset_index(drop=True)
    # Order in descending orders, necessary for the optimization
    Demands_BM = Demands_BM.sort_values(by='Offer price', ascending=False).reset_index(drop=True)

    DA_obj, DA_gen0, DA_dem0, DA_price = Single_hour_optimization(hour,Generators_BM,Demands_BM)

    DA_gen = pd.DataFrame({'Name': Generators_BM["Name"], 'Capacity' : Generators_BM['Capacity'], 'Production': DA_gen0, 'Bid price': Generators_BM['Bid price']})

    #Dataframe to calculate later the earnings
    BM_clearing = DA_gen[['Name','Production']].copy()

    #Implementing changes of pruduction to wind farms and setting production of conventional generators that failed to 0
    for name in DA_gen['Name']:
        if name in CG_outage:
            DA_gen.loc[DA_gen['Name'] == name, 'Production'] = 0.0
        if name in WF_lower:
            DA_gen.loc[DA_gen['Name'] == name, 'Production'] *= 1-WF_down_imbalance
        if name in WF_higher:
            DA_gen.loc[DA_gen['Name'] == name, 'Production'] *= 1+WF_up_imbalance

    #total imbalance  
    Dp = sum(DA_dem0) - DA_gen['Production'].sum()

    #Creating new dataframe where only the generators that are elidgible for balancing are included, alongside their remaining capacities
    #and their day ahead prices
    Balancing_gen = DA_gen.copy()

    #Removing wind farms, and generators that failed
    for name in Balancing_gen['Name']:
        if name in CG_outage:
            Balancing_gen = Balancing_gen[Balancing_gen['Name'] != name]
        if name in WF_lower:
            Balancing_gen = Balancing_gen[Balancing_gen['Name'] != name]
        if name in WF_higher:
            Balancing_gen = Balancing_gen[Balancing_gen['Name'] != name]

    #Generators that can provide up regulation
    Balancing_gen_up = Balancing_gen.copy()
    Balancing_gen_up = Balancing_gen_up[Balancing_gen_up['Capacity'] != Balancing_gen_up['Production']] 
    Balancing_gen_up['Capacity'] = Balancing_gen_up['Capacity'] - Balancing_gen_up['Production']
    Balancing_gen_up = Balancing_gen_up.drop('Production', axis=1)

    #Generators that can provide down regulation
    Balancing_gen_dw = Balancing_gen.copy()
    Balancing_gen_dw = Balancing_gen_dw[Balancing_gen_dw['Production'] != 0]
    Balancing_gen_dw['Capacity'] = Balancing_gen_dw['Production']
    Balancing_gen_dw = Balancing_gen_dw.drop('Production', axis=1)
    Balancing_gen_dw.reset_index(drop=True, inplace=True)

    #implementing the changes on the bidding price
    Balancing_gen_up['Bid price'] = Balancing_gen_up['Bid price']*0.1 + DA_price
    Balancing_gen_dw['Bid price'] = DA_price - Balancing_gen_dw['Bid price']*0.13

    #adding curtailment to available generators for up regulation
    curtailment = pd.DataFrame({'Name': 'Curtailment', 'Capacity': sum(DA_dem0), 'Bid price': 400.0}, index=[0])
    Balancing_gen_up = pd.concat([Balancing_gen_up, curtailment], axis=0) 
    #reseting the indexes
    Balancing_gen_up.reset_index(drop=True, inplace=True)

    #calling the function Single_hour_balancing
    BM_obj, BM_up, BM_dw, BM_price = Single_hour_balancing(hour,Balancing_gen_up,Balancing_gen_dw, Dp)

    Balancing_gen_up['BM production'] = BM_up
    Balancing_gen_up = Balancing_gen_up[Balancing_gen_up['Name'] != 'Curtailment']
    Balancing_gen_dw['BM production'] = BM_dw

    #Formulating the balancing market dataframe
    BM_clearing.rename(columns={'Production': 'DA_production'}, inplace=True)
    BM_clearing['DA_bid_price'] = [0] * len(BM_clearing)
    BM_clearing = BM_clearing.merge(Balancing_gen_up[['Name', 'Bid price']], on='Name', how='left', suffixes=('', ''))
    BM_clearing = BM_clearing.merge(Balancing_gen_dw[['Name', 'Bid price']], on='Name', how='left', suffixes=('', ' dw'))
    BM_clearing['Imbalance'] = [0] * len(BM_clearing)
    BM_clearing['BM_up'] = [0] * len(BM_clearing)
    BM_clearing['BM_dw'] = [0] * len(BM_clearing)
    BM_clearing['DA_wo_BM'] = [0] * len(BM_clearing)
    BM_clearing['single-price'] = [0] * len(BM_clearing)
    BM_clearing['dual-price'] = [0] * len(BM_clearing)
    BM_clearing.rename(columns={'Bid price': 'BM_bid_price_up', 'Bid price dw': 'BM_bid_price_dw'}, inplace=True)
    BM_clearing.fillna(0, inplace=True)

    #incorporating the DA bid prices (marginal costs of generators)
    for name in BM_clearing['Name'].values:
        BM_clearing.loc[BM_clearing['Name'] == name, 'DA_bid_price'] = Generators_BM.loc[Generators_BM['Name'] == name, 'Bid price']

    for name in BM_clearing['Name'].values:
        #Imbalances
        if name in CG_outage:
            BM_clearing.loc[BM_clearing['Name'] == name, 'Imbalance'] = -BM_clearing.loc[BM_clearing['Name'] == name, 'DA_production']
        if name in WF_lower:
            BM_clearing.loc[BM_clearing['Name'] == name, 'Imbalance'] = -BM_clearing.loc[BM_clearing['Name'] == name, 'DA_production']*WF_down_imbalance
        if name in WF_higher:
            BM_clearing.loc[BM_clearing['Name'] == name, 'Imbalance'] = BM_clearing.loc[BM_clearing['Name'] == name, 'DA_production']*WF_up_imbalance

        #Balancing market up & down productions
        if name in Balancing_gen_up['Name'].values:
            BM_clearing.loc[BM_clearing['Name'] == name, 'BM_up'] = Balancing_gen_up.loc[Balancing_gen_up['Name'] == name, 'BM production'].values
        if name in Balancing_gen_dw['Name'].values:
            BM_clearing.loc[BM_clearing['Name'] == name, 'BM_dw'] = Balancing_gen_dw.loc[Balancing_gen_dw['Name'] == name, 'BM production'].values

    #single and dual pricing
    for index, row in BM_clearing.iterrows():
        BM_clearing.loc[index, 'DA_wo_BM'] =  row['DA_production']*(DA_price - row['DA_bid_price'])
        #we are subtracting DA_bid_price from BM clearing price because DA_bid_price is the marginal cost of the generators
        BM_clearing.loc[index, 'single-price'] = row['DA_production']*(DA_price - row['DA_bid_price']) + row['Imbalance']*BM_price + row['BM_up']*(BM_price - row['DA_bid_price']) 
        #for dual price we need to distinquish between positive and negative imbalances
        if row['Imbalance'] > 0:
            BM_clearing.loc[index, 'dual-price'] = row['DA_production']*(DA_price - row['DA_bid_price']) + row['Imbalance']*DA_price + row['BM_up']*(BM_price - row['DA_bid_price'])
        else:
            BM_clearing.loc[index, 'dual-price'] = row['DA_production']*(DA_price - row['DA_bid_price']) + row['Imbalance']*BM_price + row['BM_up']*(BM_price - row['DA_bid_price'])
    
    return(BM_clearing)



if __name__ == "__main__": 
    # Select only one hour
    hour_input = 10

    #Conventional generetors that fails
    CG_outage_input = ['Generator 9']

    #Wind farms with lower production
    WF_lower_input = ['Wind farm 1', 'Wind farm 2', 'Wind farm 3']

    #Wind farms with higher production
    WF_higher_input = ['Wind farm 4', 'Wind farm 5', 'Wind farm 6']

    #Wind farm imbalances on percentages
    WF_down_imbalance_input = 0.1
    WF_up_imbalance_input = 0.15

    BM_clearing_output = Balancing_market_clearing(Generators, Demands, Wind_Farms,hour_input, CG_outage_input, WF_lower_input, WF_higher_input, WF_down_imbalance_input, WF_up_imbalance_input)

    print(BM_clearing_output)






# Plotting the solution of the clearing for an hour and demands and generators entries
def Balancing_plot(hour, Balancing_gen_dw, Balancing_gen_up, Dp, BM_price) :
    # Size of the figure
    plt.figure(figsize = (20, 12))
    plt.rcParams["font.size"] = 16
    
    # Positions of the generation units bars
    x_dw, y_dw, w_dw = [],[],[]
    x_up, y_up, w_up = [],[],[]
    max_down = Balancing_gen_dw['Capacity'].sum()
    xpos = -max_down
    for i in range(len(Balancing_gen_dw)) :
        x_dw.append(xpos)
        y_dw.append(-Balancing_gen_dw['Bid price'][i])
        w_dw.append(Balancing_gen_dw['Capacity'][i])
        xpos += Balancing_gen_dw['Capacity'][i]
    for i in range(len(Balancing_gen_up)) :
        x_dw.append(xpos)
        y_dw.append(Balancing_gen_up['Bid price'][i])
        w_dw.append(Balancing_gen_up['Capacity'][i])
        xpos += Balancing_gen_up['Capacity'][i]
    
    # Names of the generators
    name_dw = Balancing_gen_dw['Name'].tolist()
    name_up = Balancing_gen_up['Name'].tolist()
    names = name_dw + name_up 
    # names = list(set(names))
    
    # Concatenate all bars
    x = x_dw + x_up
    y = y_dw + y_up
    w = w_dw + w_up
        
    # Different colors for each suppliers
    colors = sns.color_palette('flare', len(x))
    
    # Plot the bar for the supply
    fig_bar = plt.bar(x, 
            height = y,
            width = w,
            fill = True,
            color = colors,
            align = 'edge')
    
    # Legend with name of suppliers
    plt.legend(fig_bar.patches, names,
              loc = "best",
              ncol = 3)
    
    # Balance need
    plt.vlines(x = Dp, ymin=-100, ymax=100,
                color = "red",
                linestyle = "dashed")
    
    # Limit of the figure
    plt.xlim(-max_down*1.1, Balancing_gen_up['Capacity'].sum()*1.1)
    plt.ylim(-Balancing_gen_dw['Bid price'].max()*1.1, Balancing_gen_up['Bid price'].max()*1.1)

    plt.xlabel("Quantity (MW)")
    plt.ylabel("Bid price ($/MWh)")
    # plt.title(Title)
    output_folder = os.path.join(os.getcwd(), 'plots')
    pdf_name = f'Balancing market in hour {hour}'+'.pdf'
    pdf_filename = os.path.join(output_folder, pdf_name)
    plt.savefig(pdf_filename,  bbox_inches='tight')
    plt.show()
    
#Balancing_plot(hour, Balancing_gen_dw, Balancing_gen_up, Dp, BM_price)
