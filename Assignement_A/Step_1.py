# The goal of this file is to do the first step of the assignment
# The copper-plate system for one single hour
# Example of function call are put at the end 

########################
""" Relevant modules """
########################

from Data import Generators, Demands, Wind_Farms
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import gurobipy as gp
import os
GRB = gp.GRB


####################################
""" Function to clear the market """
####################################
        
# This function is the main optimization function of the copper-plate problem for one hour
# It takes in entry dataframe that contains information for ONLY ONE HOUR :
    # An hour as an input ??
    # A dataframe 'Supply' with the information about all generations possibility ('Generators' + 'Wind_Farms')
    # A dataframe 'Demands' with all the demand to fullfill
# And gives as an output :
    # The cost of the system : 'optimal_obj'
    # The decision of generating unit in a list : 'optimal_gen'
    # The decision of fullfilled demand in a list : 'optimal_dem'
    # The equilibrium price obtained from the dual variable : 'equilibrium_price'

def Single_hour_optimization(hour, Supply, Demands) :
    # Numbers of generating and demanding units
    n_gen = len(Supply)
    n_dem = len(Demands)
    
    #Optimization part
    # Create the model
    model = gp.Model()
    # Initialize the decision variables
    var_gen = model.addVars(n_gen, lb=-gp.GRB.INFINITY, vtype=GRB.CONTINUOUS, name='gen')
    var_dem = model.addVars(n_dem, lb=-gp.GRB.INFINITY, vtype=GRB.CONTINUOUS, name='dem')
    # Add the objective function to the model
    model.setObjective(gp.quicksum(Demands['Offer price'][i]*var_dem[i] for i in range(n_dem))-gp.quicksum(Supply['Bid price'][i]*var_gen[i] for i in range(n_gen)), GRB.MAXIMIZE)
    # Add constraints to the model
    # Quantity supplied = Quantity used
    model.addConstr(gp.quicksum(var_dem[i] for i in range(n_dem)) - gp.quicksum(var_gen[i] for i in range(n_gen)) == 0, name='Power balance')
    # Quantities supplied can't be higher than maximum quantities
    for i in range(n_gen) :
        name_constr = f"{Supply['Name'][i]}_uplim_{i}"
        model.addConstr(var_gen[i] <= Supply['Capacity'][i], name=name_constr)
        name_constr = f"{Supply['Name'][i]}_downlim_{i}"
        model.addConstr(var_gen[i] >= 0.0, name=name_constr)
    # Quantities used can't be higher than maximum demands
    for i in range(n_dem) :
        name_constr = f"{Demands['Name'][i]}_uplim_{i}"
        model.addConstr(var_dem[i] <= Demands['Load'][i], name=name_constr)
        name_constr = f"{Demands['Name'][i]}_downlim_{i}"
        model.addConstr(var_dem[i] >= 0.0, name=name_constr)
    # Solve the problem
    model.optimize()
    
    #Get the optimal values
    if model.status == GRB.OPTIMAL:
        # Create a list to store the optimal values of the variables ---> We can also decide to store them back in Supply and Demands ????
        optimal_gen = [var_gen[i].X for i in range(n_gen)]
        optimal_dem = [var_dem[i].X for i in range(n_dem)]
        # Value of the optimal objective
        optimal_obj = model.ObjVal
        
        # Print in the console the power output of each generation unit and the fullfilled demand
        print("\n")
        print("Power generation :")
        for i, value in enumerate(optimal_gen):
            print(Supply["Name"][i] + f" : {round(value,2)} MW")
        print("\n")
        print("Demand provided :")
        for i, value in enumerate(optimal_dem):
            print(Demands["Name"][i] + f" : {round(value,2)} MW")
            

        """ KKTs output """
        # Write KKT's condition in a text file
        # Define the folder path and the file name
        folder_path = "results/single_hour"
        file_name = f"KKTs_{hour}.txt"
        
        # Create the folder if it doesn't exist
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        
        # Combine the folder path and file name
        file_path = os.path.join(folder_path, file_name)
        
        # Open the file in write mode
        with open(file_path, 'w') as file:
            file.write(f"KKTs for the single hour optimization in hour {hour} :")
            file.write("\n\n")
            # Write lines to the file
            for c in model.getConstrs():
                file.write(f"{c.ConstrName}: {c.Pi} : {c.Sense} \n")
                # Get the equilibrium price
                if c.ConstrName == 'Power balance' :
                    equilibrium_price = c.Pi
             
        # Print in the console the equilibrium price and the quantity provided
        print("\n")
        print(f"Equilibrium price : {equilibrium_price} $/MWh")
        print(f"Quantity provided : {round(sum(optimal_gen),2)} $/MWh")
       
    else:
        print("Optimization did not converge to an optimal solution.")
        
    # Return the cost and the optimal values
    return(optimal_obj, optimal_gen, optimal_dem, equilibrium_price)


###############################
""" Vizualization functions """
###############################

# This function is the function used to plot the solution of the market clearing FOR ONE HOUR
# It takes in entry dataframes and list that contains information for ONLY ONE HOUR :
    # A dataframe 'Supply' with the information about all generations possibility ('Generators' + 'Wind_Farms')
    # A dataframe 'Demands' with all the demand to fullfill
    # The equilibrium price frome the optimization
    # The optimal demand list from the optimization
    # The optimal generation list from the optimization
    # A Title as a string to save the plot
# This plotting function is quite complexe cause it has been modified to observe some effects in other step
# WARNING !! The dataframe 'Supply' should be ordered in ascending order and the df 'Demands' should be ordered in descending order of bid price !!
# WARNING !! The 'optimal_gen' and 'optimal_dem' should also be in the right order !!
# If the optimization problem have been solved previously with the dataframe in the right order, no need to put in the right order the 'optimal_dem' and 'optimal_gen'

def Single_hour_plot(Supply, Demands, equilibrium_price, optimal_gen, optimal_dem, Title) :
    # Size of the figure and the font size
    plt.figure(figsize = (20, 12))
    plt.rcParams["font.size"] = 20
    
    # Positions of the generation units bars
    Gen_conv, Gen_constr, Gen_not = [], [], []
    xpos_wf, y_wf, w_wf = [],[],[]
    xpos_conv, y_conv, w_conv = [],[],[]
    xpos_constr, y_constr, w_constr = [],[],[]
    xpos_not, y_not, w_not = [],[],[]
    xpos = 0
    x_not_select = -1
    price_last_selected = 0
    # Algorithm to have the right format
    for i in range(len(Supply)) :
        if 'Wind farm' in Supply['Name'][i] :
            xpos_wf.append(xpos)
            y_wf.append(Supply['Bid price'][i])
            w_wf.append(optimal_gen[i])
            Gen_conv.append(Supply['Name'][i])
            xpos += optimal_gen[i]
        else :
            if optimal_gen[i] == 0 :
                xpos_not.append(xpos)
                y_not.append(Supply['Bid price'][i])
                w_not.append(Supply['Capacity'][i])
                if Supply['Name'][i] not in Gen_conv+Gen_constr :
                    Gen_not.append(Supply['Name'][i])
                if x_not_select == -1 :
                    x_not_select = xpos
                    price_last_selected = Supply['Bid price'][i-1]
                xpos += Supply['Capacity'][i]
            else :
                if x_not_select == -1 :
                    xpos_conv.append(xpos)
                    y_conv.append(Supply['Bid price'][i])
                    w_conv.append(optimal_gen[i])
                    Gen_conv.append(Supply['Name'][i])
                    if optimal_gen[i] == Supply['Capacity'][i] :
                        xpos += optimal_gen[i]
                    else :
                        xpos_not.append(xpos+optimal_gen[i])
                        y_not.append(Supply['Bid price'][i])
                        w_not.append(Supply['Capacity'][i] - optimal_gen[i])
                        if Supply['Name'][i] not in Gen_conv+Gen_constr :
                            Gen_not.append(Supply['Name'][i])
                        x_not_select = xpos + optimal_gen[i]
                        xpos += Supply['Capacity'][i]
                        price_last_selected = Supply['Bid price'][i]
                elif x_not_select != -1 and price_last_selected == Supply['Bid price'][i] :
                    xpos_conv.append(x_not_select)
                    y_conv.append(Supply['Bid price'][i])
                    w_conv.append(optimal_gen[i])
                    Gen_conv.append(Supply['Name'][i])
                    for j in range(len(xpos_not)) :
                        xpos_not[j] += optimal_gen[i]
                    x_not_select += optimal_gen[i]
                    xpos += optimal_gen[i]
                    if optimal_gen[i] != Supply['Capacity'][i] :
                        xpos_not.append(xpos)
                        y_not.append(Supply['Bid price'][i])
                        w_not.append(Supply['Capacity'][i] - optimal_gen[i])
                        if Supply['Name'][i] not in Gen_conv+Gen_constr :
                            Gen_not.append(Supply['Name'][i])
                        xpos += Supply['Capacity'][i] - optimal_gen[i]
                else :
                    xpos_constr.append(x_not_select)
                    y_constr.append(Supply['Bid price'][i])
                    w_constr.append(optimal_gen[i])
                    Gen_constr.append(Supply['Name'][i])
                    for j in range(len(xpos_not)) :
                        xpos_not[j] += optimal_gen[i]
                    x_not_select += optimal_gen[i]
                    xpos += optimal_gen[i]
                    if optimal_gen[i] != Supply['Capacity'][i] :
                        xpos_not.append(xpos)
                        y_not.append(Supply['Bid price'][i])
                        w_not.append(Supply['Capacity'][i] - optimal_gen[i])
                        if Supply['Name'][i] not in Gen_conv+Gen_constr :
                            Gen_not.append(Supply['Name'][i])
                        xpos += Supply['Capacity'][i] - optimal_gen[i]
    # Different colors for each suppliers
    colors = sns.color_palette('flare', len(xpos_wf)+len(xpos_conv))
    col_wf = colors[:len(xpos_wf)]
    col_conv = colors[len(xpos_conv):]
    Generators_name = Gen_conv + Gen_constr + Gen_not
    # Plot the bar for the supply
    fig_wf = plt.bar(xpos_wf, 
            height = y_wf,
            width = w_wf,
            fill = True,
            color = col_wf,
            align = 'edge')
    fig_conv = plt.bar(xpos_conv, 
            height = y_conv,
            width = w_conv,
            fill = True,
            color = col_conv,
            edgecolor = 'black',
            align = 'edge')
    fig_constr = plt.bar(xpos_constr, 
            height = y_constr,
            width = w_constr,
            fill = True,
            color = "red",
            align = 'edge')
    fig_not = plt.bar(xpos_not, 
            height = y_not,
            width = w_not,
            fill = True,
            color = 'darkgreen',
            edgecolor = 'black',
            align = 'edge',
            alpha = 0.5)
    
    # Legend with name of suppliers
    plt.legend(fig_wf.patches+fig_conv.patches+fig_constr.patches+fig_not.patches, Generators_name,
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
    plt.text(x = sum(optimal_gen) - 1400 ,
            y = equilibrium_price + 2 ,
            ma = 'right',
            s = f"Electricity price: {equilibrium_price} $/MWh\n Quantity : " + str(round(sum(optimal_dem),2)) + " MW")
    
    # Limit of the figure
    plt.xlim(0,4200)
    plt.ylim(0, max(Supply["Bid price"].max(),Demands["Offer price"].max()) + 10)

    # Label for the axes
    plt.xlabel("Quantity [MW]")
    plt.ylabel("Bid price [$/MWh]")
    
    # Saving the plot
    output_folder = os.path.join(os.getcwd(), 'plots')
    pdf_name = Title+'.pdf'
    pdf_filename = os.path.join(output_folder, pdf_name)
    plt.savefig(pdf_filename,  bbox_inches='tight')
    plt.show()
    

# This function is used to compute the social welfare, the profit of every suppliers and the utility of each demands
# It takes in entry dataframes and list that contains information for ONLY ONE HOUR :
    # A dataframe 'Supply' with the information about all generations possibility ('Generators' + 'Wind_Farms')
    # A dataframe 'Demands' with all the demand to fullfill
    # The optimal demand list from the optimization
    # The optimal generation list from the optimization
    # The optimal objective of the optimization problem
    # The equilibrium price frome the optimization
# And gives as an output :
    # The social welfare
    # The profits of suppliers in a list
    # The utility of demands in a list

def Commodities(Supply, Demands, optimal_gen, optimal_dem, optimal_obj, equilibrium_price) :
    Social_welfare = optimal_obj
    Profits_of_suppliers = []
    Utility_of_demands = []
    for i, value in enumerate(optimal_gen): 
        Profits_of_suppliers.append([Supply['Name'][i],(equilibrium_price - Supply['Bid price'][i])*value])
    for i, value in enumerate(optimal_dem): 
        Utility_of_demands.append([Demands['Name'][i],(Demands['Offer price'][i] - equilibrium_price)*value])
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


##################################
""" Global function for step 1 """
##################################

# This function is a global function to compute everything easily for the first step
# This function will select only the values for the hour we want
# Then it will compute the optimization problem and plot all the results we want
# It takes in entry dataframes and list that contains information FOR EVERY HOUR :
    # The dataframe 'Generators' with the information about conventional generators
    # The dataframe 'Wind_Farms' with the information about wind farms
    # The dataframe 'Demands' with all the demand to fullfill
# And gives as an output :
    # The dataframe 'Supply' concatenation of the generators and wind farms in bid price ascending order
    # The dataframe 'Demands' in offer price descending order
    # 'optimal_gen' in the right order
    # 'optimal_dem' in the right order

def Copper_plate_single_hour(hour, Generators, Wind_Farms, Demands) :
    # Select the right hour for the demands
    for i in range(len(Demands)) :
        Demands.loc[i, 'Load'] = Demands['Load'][i][hour-1]
        Demands.loc[i, 'Offer price'] = Demands['Offer price'][i][hour-1]
    # Select the right hour for the generators and the wind farms
    for i in range(len(Wind_Farms)) :
        Wind_Farms.loc[i, 'Capacity'] = round(Wind_Farms['Capacity'][i][hour-1],2)
        Wind_Farms.loc[i, 'Bid price'] = Wind_Farms['Bid price'][i][hour-1]
    Supply = pd.concat([Generators, Wind_Farms], axis=0).reset_index(drop=True)
    
    # # Order in ascending orders, necessary for the optimization 
    # Supply = Supply.sort_values(by=['Bid price','Name'], ascending=[True, 'Wind farm' in Generators['Name'].values]).reset_index(drop=True)
    # # Order in descending orders, necessary for the optimization
    # Demands = Demands.sort_values(by='Offer price', ascending=False).reset_index(drop=True)
    
    # Solving the optimization problem
    optimal_obj, optimal_gen, optimal_dem, equilibrium_price = Single_hour_optimization(hour, Supply, Demands)
    # Calculating commodities
    Social_welfare, Profits_of_suppliers, Utility_of_demands = Commodities(Supply, Demands, optimal_gen, optimal_dem, optimal_obj, equilibrium_price)
    
    #Necessary before plotting the values
    # Order in ascending orders of bid price for easier treatment
    Supply['Optimal'] = optimal_gen
    Supply = Supply.sort_values(by=['Bid price', 'Optimal'], ascending=[True, False]).reset_index(drop=True)
    # Order in descending orders of offer price for easier treatment
    Demands['Optimal'] = optimal_dem
    Demands = Demands.sort_values(by='Offer price', ascending=False).reset_index(drop=True)
    # Get back the sorted optimal values
    optimal_gen = Supply['Optimal'].to_list()
    optimal_dem = Demands['Optimal'].to_list()

    # Plotting the results
    Single_hour_plot(Supply, Demands, equilibrium_price, optimal_gen, optimal_dem, "Single hour "+str(hour))
    return (Supply, Demands, optimal_gen, optimal_dem)

############
""" KKTs """
############

# Function to verify one by one every KKTs

def KKTs(optimal_gen, optimal_dem, Supply, Demands):
    my_G_underline=[]
    my_G_overline=[]
    my_D_underline=[]
    my_D_overline=[]
    
    #equality constraints
    equality_constraint = sum(optimal_dem) - sum(optimal_gen)
    print("\n")
    if equality_constraint != 0:
        print("Equality constraint NOT fulfilled:", equality_constraint)
    else:
        print("Equality constraint fulfilled")
    

    #inequality constriants generators
    for i in range(len(optimal_gen)):
        if -optimal_gen[i]>0:
            print("Inequality constraint 1 NOT fulfilled:", optimal_gen['Name'][i])
        elif optimal_gen[i]==0:
            print("Inequality constraint 1 = 0:", Supply['Name'][i])
            my_G_underline.append(1)
        else:
            print("Inequality constraint 1 < 0:", Supply['Name'][i])
            my_G_underline.append(0)
            
        if (optimal_gen[i]-Supply['Capacity'][i])>0:
            print("Inequality constraint 2 NOT fulfilled:", Supply['Name'][i])
        elif (optimal_gen[i]-Supply['Capacity'][i])==0:
            print("Inequality constraint 2 = 0:", Supply['Name'][i])
            my_G_overline.append(1)
        else:
            print("Inequality constraint 2 < 0:", Supply['Name'][i])
            my_G_overline.append(0)
           
    print("\n")
    
    #inequality constriants generators
    for i in range(len(optimal_dem)):
        if -optimal_dem[i]>0:
            print("Inequality constraint 3 NOT fulfilled:", optimal_dem[i])
        elif optimal_dem[i]==0:
            print("Inequality constraint 3 = 0:", Demands['Name'][i])
            my_D_underline.append(None)
        else:
            print("Inequality constraint 3 < 0:", Demands['Name'][i])
            my_D_underline.append(0)
            
        if (optimal_dem[i]-Demands['Load'][i])>0:
            print("Inequality constraint 4 NOT fulfilled:", Demands['Name'][i])
        elif (optimal_dem[i]-Demands['Load'][i])==0:
            print("Inequality constraint 4 = 0:", Demands['Name'][i])
            my_D_overline.append(None)
        else:
            print("Inequality constraint 4 < 0:", Demands['Name'][i])
            my_D_overline.apped(0)
            
    print("\n"+"my_G_underline")            
    for i, value in enumerate(optimal_gen):
            print(my_G_underline[i])
    print("\n"+"my_G_overline")
    for i, value in enumerate(optimal_gen):
            print(my_G_overline[i])
    print("\n"+"my_D_underline")
    for i, value in enumerate(optimal_dem):
        print(my_D_underline[i]) 
    print("\n"+"my_D_overline")
    for i, value in enumerate(optimal_dem):
        print(my_D_overline[i])  
        
    
    #check for which generator my_G_underline and my_G_overline is equal to zero
    #->last generator producing
    for i in range(len(my_G_overline)):
        if my_G_overline[i]==0 and my_G_underline[i]==0:
            print("\n"+"Last Generator producing:", Supply['Name'][i]) 
            
            #get market price lambda out of derivative of lagrange
            KKT_market_price= Supply['Bid price'][i]-my_G_underline[i]+my_G_overline[i]
            print("\n"+"KKT market price: ", KKT_market_price) 

    return()


################################
""" How to use the functions """
################################

# Computed only if we launch this file alone
if __name__ == "__main__":
    # Select only one hour
    hour = 10
    # Launch the main function
    Supply, Demands, optimal_gen, optimal_dem = Copper_plate_single_hour(hour, Generators, Wind_Farms, Demands)
    # If you want to verify every KKT one by one
    # KKTs(optimal_gen, optimal_dem, Supply, Demands)





        

