""" Code for the copper-plate system with one hour """

import numpy as np
import matplotlib.pyplot as plt
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
    
    
""" Fake Variables defined just for trying the problem """

Generators = pd.DataFrame([
    ['Gas 1',40,80],['Gas 2',25,85],
    ['Coal 1',30,70],['Coal 2',30,65],
    ['Biomass',20,40],['Nuclear',80,20],
    ['Wind 1',20,0],['Wind 2',5,0]],
    columns=['Name','Capacity','Bid price'])

Demands = pd.DataFrame([
    ['Houses',120,120],['Industry 1',60,100],
    ['Industry 2',20,80]],
    columns=['Name','Load','Offer price'])


""" Optimization problem """
        
def Coplate_single_hour(Generators, Demands) :
    # Numbers of generators and demanding units
    n_gen = len(Generators)
    n_dem = len(Demands)
    #Create the model
    model = gp.Model()
    #Initialize the decision variables
    var_gen = model.addVars(range(n_gen), vtype=GRB.CONTINUOUS, name='gen')
    var_dem = model.addVars(range(n_dem), vtype=GRB.CONTINUOUS, name='dem')
    #Add constraints to the model
    model.addConstr(gp.quicksum(var_dem[i] for i in range(n_dem)) - gp.quicksum(var_gen[i] for i in range(n_gen)) == 0)
    for i in range(n_gen) :
        model.addConstr(var_gen[i] <= Generators['Capacity'][i])
    for i in range(n_dem) :
        model.addConstr(var_dem[i] <= Demands['Load'][i])
    # Add the objective function to the model
    model.setObjective(gp.quicksum([Demands['Offer price'][i]*var_dem[i] for i in range(n_dem)])-gp.quicksum([Generators['Bid price'][i]*var_gen[i] for i in range(n_gen)]), GRB.MAXIMIZE)
    #Solve the problem
    model.optimize()
    
    # Get the optimal values
    if model.status == GRB.OPTIMAL:
        # Create a list to store the optimal values of the variables
        optimal_gen = [var_gen[i].X for i in range(n_gen)]
        optimal_dem = [var_dem[i].X for i in range(n_dem)]
        # Value of the optimal objective
        optimal_obj = model.ObjVal
        
        # Print
        print("\n")
        for i, value in enumerate(optimal_gen):
            print(f"Generator {i}: {value}")
        for i, value in enumerate(optimal_dem):
            print(f"Demand {i}: {value}")
    else:
        print("Optimization did not converge to an optimal solution.")
        
    # Return the cost and the optimal values
    return(optimal_obj, optimal_gen, optimal_dem)
    
optimal_obj, optimal_gen, optimal_dem = Coplate_single_hour(Generators, Demands)        

        

