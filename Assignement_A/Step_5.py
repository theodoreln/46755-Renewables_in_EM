
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

# Select only one hour
hour = 10
for i in range(len(Demands)) :
    Demands.loc[i, 'Load'] = Demands['Load'][i][hour-1]
    Demands.loc[i, 'Offer price'] = Demands['Offer price'][i][hour-1]

# Adding the wind farms
for i in range(len(Wind_Farms)) :
    Wind_Farms.loc[i, 'Capacity'] = round(Wind_Farms['Capacity'][i][hour-1],2)
    Wind_Farms.loc[i, 'Bid price'] = Wind_Farms['Bid price'][i][hour-1]
Generators = pd.concat([Generators, Wind_Farms], axis=0)
# Order in ascending orders, necessary for the optimization 
Generators = Generators.sort_values(by=['Bid price','Name'], ascending=[True, 'Wind farm' in Generators['Name'].values]).reset_index(drop=True)
# Order in descending orders, necessary for the optimization
Demands = Demands.sort_values(by='Offer price', ascending=False).reset_index(drop=True)

DA_obj, DA_gen0, DA_dem0, DA_price = Single_hour_optimization(hour,Generators,Demands)

DA_gen = pd.DataFrame({'Name': Generators["Name"], 'Capacity' : Generators['Capacity'], 'Production': DA_gen0, 'Bid price': Generators['Bid price']})

#Dataframe to calculate later the earnings
BM_clearing = DA_gen[['Name','Production']].copy()

#Conventional generetor that fails
CG_outage = ['Generator 9']

#Wind farms with lower production
WF_lower = ['Wind farm 1', 'Wind farm 2', 'Wind farm 3']

#Wind farms with higher production
WF_higher = ['Wind farm 4', 'Wind farm 5', 'Wind farm 6']


#Implementing changes of production to wind farms and setting production of convntional generators that failed to 0
for name in DA_gen['Name']:
    if name in CG_outage:
        DA_gen.loc[DA_gen['Name'] == name, 'Production'] = 0.0
    if name in WF_lower:
        DA_gen.loc[DA_gen['Name'] == name, 'Production'] *= 0.9
    if name in WF_higher:
        DA_gen.loc[DA_gen['Name'] == name, 'Production'] *= 1.15

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

#adding curtailment to available demands for up regulation
curtailment = pd.DataFrame({'Name': 'Curtailment', 'Capacity': sum(DA_dem0), 'Bid price': 400.0}, index=[0])
Balancing_gen_up = pd.concat([Balancing_gen_up, curtailment], axis=0) 
#reseting the indexes
Balancing_gen_up.reset_index(drop=True, inplace=True)

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
        # Create a list to store the optimal values of the variables ---> We can also decide to store them back in Generators and Demands ????
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
        print(f"Equilibrium price : {round(balancing_price,3)} $/MWh\n")

    else:
        print("Optimization did not converge to an optimal solution.")

    # Return the cost and the optimal values
    return(optimal_obj, optimal_gen_up, optimal_gen_dw, balancing_price)

#calling the function
BM_obj, BM_up, BM_dw, BM_price = Single_hour_balancing(hour,Balancing_gen_up,Balancing_gen_dw, Dp)

Balancing_gen_up['BM production'] = BM_up
Balancing_gen_up = Balancing_gen_up[Balancing_gen_up['Name'] != 'Curtailment']
Balancing_gen_dw['BM production'] = BM_dw

#print(Balancing_gen)
#print(Balancing_gen_up)  
#print(Balancing_gen_dw)

#Formulating the balancing market dataframe
BM_clearing.rename(columns={'Production': 'DA_production'}, inplace=True)
BM_clearing['Imbalance'] = [0] * len(BM_clearing)
BM_clearing['BM_up'] = [0] * len(BM_clearing)
BM_clearing['BM_dw'] = [0] * len(BM_clearing)
BM_clearing['DA_wo_BM'] = [0] * len(BM_clearing)
BM_clearing['single-price'] = [0] * len(BM_clearing)
BM_clearing['dual-price'] = [0] * len(BM_clearing)

for name in BM_clearing['Name'].values:
    #Imbalances
    if name in CG_outage:
        BM_clearing.loc[BM_clearing['Name'] == name, 'Imbalance'] = -BM_clearing.loc[BM_clearing['Name'] == name, 'DA_production']
    if name in WF_lower:
        BM_clearing.loc[BM_clearing['Name'] == name, 'Imbalance'] = -BM_clearing.loc[BM_clearing['Name'] == name, 'DA_production']*0.1
    if name in WF_higher:
        BM_clearing.loc[BM_clearing['Name'] == name, 'Imbalance'] = BM_clearing.loc[BM_clearing['Name'] == name, 'DA_production']*0.15

    #Balancing market up & down productions
    if name in Balancing_gen_up['Name'].values:
        BM_clearing.loc[BM_clearing['Name'] == name, 'BM_up'] = Balancing_gen_up.loc[Balancing_gen_up['Name'] == name, 'BM production'].values
    if name in Balancing_gen_dw['Name'].values:
        BM_clearing.loc[BM_clearing['Name'] == name, 'BM_dw'] = Balancing_gen_dw.loc[Balancing_gen_dw['Name'] == name, 'BM production'].values


#distinquishing between energy deficit and surplus imbalances
if Dp > 0:
    for index, row in BM_clearing.iterrows():
        BM_clearing.loc[index, 'DA_wo_BM'] = row['DA_production']*DA_price
        BM_clearing.loc[index, 'single-price'] = row['DA_production']*DA_price + row['Imbalance']*BM_price + row['BM_up']*BM_price 
        #for dual price we need to distinquish whether individual imbalances have a positive or negative impact so to apply either DA or 
        # balancing price, conventional generator outages can be implemented here since always balancing price is applied 
        if row['Imbalance'] > 0:
            BM_clearing.loc[index, 'dual-price'] = row['DA_production']*DA_price + row['Imbalance']*DA_price + row['BM_up']*BM_price 
        else:
            BM_clearing.loc[index, 'dual-price'] = row['DA_production']*DA_price + row['Imbalance']*BM_price + row['BM_up']*BM_price
else:
    for index, row in BM_clearing.iterrows():
        BM_clearing.loc[index, 'DA_wo_BM'] = row['DA_production']*DA_price
        #we need to distinquish for the  outage of conventional genrators since they always need to pay when they fail to deliver
        if row['Name'] in CG_outage:
            BM_clearing.loc[index, 'single-price'] = row['DA_production']*DA_price + row['Imbalance']*BM_price 
        else:
            BM_clearing.loc[index, 'single-price'] = row['DA_production']*DA_price + row['Imbalance']*BM_price - row['BM_dw']*BM_price
            if row['Imbalance'] < 0:
                BM_clearing.loc[index, 'dual-price'] = row['DA_production']*DA_price + row['Imbalance']*DA_price - row['BM_dw']*BM_price
            else:
                BM_clearing.loc[index, 'dual-price'] = row['DA_production']*DA_price + row['Imbalance']*BM_price - row['BM_dw']*BM_price


print(BM_clearing)


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
    
Balancing_plot(hour, Balancing_gen_dw, Balancing_gen_up, Dp, BM_price)