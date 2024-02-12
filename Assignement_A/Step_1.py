""" Code for the copper-plate system with one hour """

from Data import Generators, Demands, Wind_Farms
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import gurobipy as gp
GRB = gp.GRB


""" Variables used in the file """
# Informations on generating units are in the dataframe 'Generators' with the columns names :
    # 'Name' for the name of the generator
    # 'Capacity' for the capacity of the generator
    # 'Bid price' for the bid price of the generator
    
# Informations on demanding units are in the dataframe 'Demands' with the columns names :
    # 'Name' for the name of the demand
    # 'Load' for the load of the demand
    # 'Offer price' for the offer price of the demand
    
    
""" Fake Variables version defined just for trying the problem """

# Generators = pd.DataFrame([
#     ['Gas 1',40,80],['Gas 2',25,85],
#     ['Coal 1',30,70],['Coal 2',30,65],
#     ['Biomass',20,40],['Nuclear',80,20],
#     ['Wind 1',20,0],['Wind 2',5,0]],
#     columns=['Name','Capacity','Bid price'])

# Demands = pd.DataFrame([
#     ['Houses',120,120],['Industry 1',50,100],
#     ['Industry 2',15,80]],
#     columns=['Name','Load','Offer price'])

""" With real variable, select one hour """

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
# Order in ascending orders of bid price for easier treatment
Generators = Generators.sort_values(by='Bid price').reset_index(drop=True)
# Order in descending orders of offer price for easier treatment
Demands = Demands.sort_values(by='Offer price', ascending=False).reset_index(drop=True)


""" Function useful for clearing the market """
        
# Taking the hour supply and load information and optimizing (see lecture 2)
def Single_hour_optimization(Generators, Demands) :
    # Global variables
    global optimal_gen, optimal_dem
    # Numbers of generators and demanding units
    n_gen = len(Generators)
    n_dem = len(Demands)
    
    #Optimization part
    # Create the model
    model = gp.Model()
    # Initialize the decision variables
    var_gen = model.addVars(n_gen, vtype=GRB.CONTINUOUS, name='gen')
    var_dem = model.addVars(n_dem, vtype=GRB.CONTINUOUS, name='dem')
    # Add constraints to the model
    # Quantity supplied = Quantity used
    model.addConstr(gp.quicksum(var_dem[i] for i in range(n_dem)) - gp.quicksum(var_gen[i] for i in range(n_gen)) == 0)
    # Quantities supplied can't be higher than maximum quantities
    for i in range(n_gen) :
        model.addConstr(var_gen[i] <= Generators['Capacity'][i])
        model.addConstr(var_gen[i] >= 0)
    # Quantities used can't be hihher than maximum demands
    for i in range(n_dem) :
        model.addConstr(var_dem[i] <= Demands['Load'][i])
        model.addConstr(var_dem[i] >= 0)
    # Add the objective function to the model
    model.setObjective(gp.quicksum(Demands['Offer price'][i]*var_dem[i] for i in range(n_dem))-gp.quicksum(Generators['Bid price'][i]*var_gen[i] for i in range(n_gen)), GRB.MAXIMIZE)
    # Solve the problem
    model.optimize()
    
    #Get the optimal values
    if model.status == GRB.OPTIMAL:
        # Create a list to store the optimal values of the variables ---> We can also decide to store them back in Generators and Demands ????
        optimal_gen = [var_gen[i].X for i in range(n_gen)]
        optimal_dem = [var_dem[i].X for i in range(n_dem)]
        # Value of the optimal objective
        optimal_obj = model.ObjVal
        
        # Print
        print("\n")
        print("Power generation :")
        for i, value in enumerate(optimal_gen):
            print(Generators["Name"][i] + f" : {round(value,2)} MW")
        print("\n")
        print("Demand provided :")
        for i, value in enumerate(optimal_dem):
            print(Demands["Name"][i] + f" : {round(value,2)} MW")
    else:
        print("Optimization did not converge to an optimal solution.")
        
    # Return the cost and the optimal values
    return(optimal_obj, optimal_gen, optimal_dem)
    

# Taking the optimization and giving the clearing price
def Single_hour_price(Generators, Demands, optimal_gen, optimal_dem) :
    #Go through the different suppliers to find the clearing price
    # Condition for clearing and initialization
    clearing = False
    clearing_price = 0
    i = 0
    while (clearing == False) and (i <= len(Generators)) :
        # For this producer, we take the quantity supplied and the maximum quantity
        max_cap = Generators['Capacity'][i]
        eff_cap = optimal_gen[i]
        # If the quantity supplied is higher than 0 but lower than the max quantity, the clearing price is equal to the bid price of this producer
        if eff_cap > 0 and eff_cap < max_cap :
            clearing_price = Generators['Bid price'][i]
            # Finish the loop
            clearing = True
        # If the quantity supplied is equal to the maximum quantity and that we are not at the last producer
        elif eff_cap == max_cap and i!= len(Generators)-1 :
            # Quantity supplied by the next producer
            next_eff_cap = optimal_gen[i+1]
            # If this quantity is different than 0, then we will indent i and go to the next producer by starting again the loop, storing current max clearing price
            if next_eff_cap != 0 :
                clearing_price = Generators['Bid price'][i]
                i += 1
            # If this quantity is equal to 0, that means that our bid price is in a range between the bid price of this producer and the bid price of the next one
            else :
                next_bid_price = Generators['Bid price'][i+1]
                clearing_price = [clearing_price, next_bid_price]
                clearing = True
        # If we arrive at the last producer without finishing the loop, then we have more demand and the clearing price is the bid price of the last producer
        else :
            clearing_price = Generators['Bid price'][i]
            clearing = True
    
    # Print the clearing price and the quantity supplied
    print("\n")
    print(f"Clearing price : {clearing_price} $/MWh")
    print("Quantity provided : " + str(round(sum(optimal_dem),2)) + " MW")
    return(clearing_price)


# Plotting the solution of the clearing for an hour and demands and generators entries
def Single_hour_plot(Generators, Demands, clearing_price, optimal_gen, optimal_dem) :
    # Size of the figure
    plt.figure(figsize = (20, 12))
    plt.rcParams["font.size"] = 16
    
    # Different colors for each suppliers
    colors = sns.color_palette('flare', len(Generators))
    # Positions of the suppliers bars
    xpos = [0]
    for i in range(1,len(Generators)) :
        xpos.append(Generators["Capacity"][i-1] + xpos[i-1])
    y = Generators["Bid price"].values.tolist()
    # Width of each suppliers bars
    w = Generators["Capacity"].values.tolist()
    # Plot the bar for the supply
    fig = plt.bar(xpos, 
            height = y,
            width = w,
            fill = True,
            color = colors,
            align = 'edge')
    # Legend with name of suppliers
    plt.legend(fig.patches, Generators["Name"].values.tolist(),
              loc = "best",
              ncol = 3)
    
    
    # Demands plotting
    max_demand = sum(Demands["Load"].values.tolist())
    xpos = 0
    for i in range(len(Demands)) :
        plt.hlines(y = Demands["Offer price"][i],
                  xmin = xpos,
                  xmax = Demands["Load"][i] + xpos,
                  color = "red",
                  linestyle = "dashed")
        xpos = Demands["Load"][i] + xpos
        if i != len(Demands)-1:
            plt.vlines(x = xpos,
                        ymin = Demands["Offer price"].values.tolist()[i+1],
                        ymax = Demands["Offer price"].values.tolist()[i],
                        color = "red",
                        linestyle = "dashed",
                        label = "Demand")
    plt.vlines(x = max_demand,
                ymin = 0,
                ymax = Demands["Offer price"].values.tolist()[-1],
                color = "red",
                linestyle = "dashed",
                label = "Demand")
    
    # Small text for the clearing price and the quantity supplied
    if type(clearing_price) == list :
        plt.text(x = sum(optimal_gen) - 10,
                y = clearing_price[-1] + 10,
                s = f"Electricity price: {clearing_price} $/MWh \n Quantity : " + str(round(sum(optimal_dem),2)) + " MW")
    else :
        plt.text(x = sum(optimal_gen) - 10,
                y = clearing_price + 10,
                s = f"Electricity price: {clearing_price} $/MWh \n Quantity : " + str(round(sum(optimal_dem),2)) + "MW")
    
    # Limit of the figure
    plt.xlim(0, max(Generators["Capacity"].sum(),Demands["Load"].sum()+5))
    plt.ylim(0, max(Generators["Bid price"].max(),Demands["Offer price"].max()) + 15)

    plt.xlabel("Power plant capacity (MW)")
    plt.ylabel("Bid price ($/MWh)")
    plt.title("Market clearing for the copper plate single hour")
    plt.show()
    
    
# Calculating social welfare, profits of suppliers and utility of demands
def Commodities(Generators, Demands, optimal_gen, optimal_dem, optimal_obj, clearing_price) :
    Social_welfare = optimal_obj
    Profits_of_suppliers = []
    Utility_of_demands = []
    for i, value in enumerate(optimal_gen): 
        Profits_of_suppliers.append([Generators['Name'][i],(clearing_price - Generators['Bid price'][i])*value])
    for i, value in enumerate(optimal_dem): 
        Utility_of_demands.append([Demands['Name'][i],(Demands['Offer price'][i] - clearing_price)*value])
    print("\n")
    print(f"Social welfare : {Social_welfare} $")
    print("\n")
    print("Profits of suppliers :")
    for item in Profits_of_suppliers:
        print(item[0] + f" : {item[1]} $")
    print("\n")
    print("Utility of demands :")
    for item in Utility_of_demands:
        print(item[0] + f" : {item[1]} $")
    return(Social_welfare, Profits_of_suppliers, Utility_of_demands)



""" Global function """

def Copper_plate_single_hour(Generators, Demands) :
    # Solving the optimization problem
    optimal_obj, optimal_gen, optimal_dem = Single_hour_optimization(Generators, Demands)
    # Clearing the price
    clearing_price = Single_hour_price(Generators, Demands, optimal_gen, optimal_dem)
    # Calculating commodities
    Social_welfare, Profits_of_suppliers, Utility_of_demands = Commodities(Generators, Demands, optimal_gen, optimal_dem, optimal_obj, clearing_price)
    # Plotting the results
    Single_hour_plot(Generators, Demands, clearing_price, optimal_gen, optimal_dem)
    
Copper_plate_single_hour(Generators, Demands)

















        

