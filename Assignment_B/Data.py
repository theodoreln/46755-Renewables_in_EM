# The goal of this file is to input the main data of the problem
# Those variables will be imported by the others files 
# Most of the values are stored in dataframes

########################
""" Relevant modules """
########################

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import copy


###################################
""" Variables used in this file """
###################################

# Informations on wind power forecasts are in the dataframe 'WP_forecast'
# Informations on day ahead price forecasts are in the dataframe 'DA_forecast'
# Informations on balancing stage of system are in the dataframe 'system_needs'
    
  
#############################################
""" Import values of the generators units """
#############################################

# Informations on generating units are in the dataframe 'Generators' with the columns names :
    # 'Name' for the name of the generator
    # 'Node' for the node related
    # 'Capacity' for the maximum capacity of the generator
    # 'Ramp up' for the ramp up limit of the generator
    # 'Ramp down' for the ramp down limit of the generator
    # 'Initial power' for the capacity provided at t=0
    # 'Bid price' for the bid price of the generator
    # 'Maximum up reserve' and 'Maximum down reserve' to limit the quantity offer to the reserve market
    # 'Up reserve price' and 'Down reserve price' correspond to the price at which the reserve is offered

Generators = pd.DataFrame([
    ['Generator 1', 1, 152, 120, 120, 76, 13.32, 40, 40, 15, 14],
    ['Generator 2', 2, 152, 120, 120, 76, 13.32, 40, 40, 15, 14],
    ['Generator 3', 7, 350, 350, 350, 0, 20.7, 70, 70, 10, 9],
    ['Generator 4', 13, 591, 240, 240, 0, 20.93, 180, 180, 8, 7],
    ['Generator 5', 15, 60, 60, 60, 0, 26.11, 60, 60, 7, 5],
    ['Generator 6', 15, 155, 155, 155, 0, 10.52, 30, 30, 16, 14],
    ['Generator 7', 16, 155, 155, 155, 124, 10.52, 30, 30, 16, 14],
    ['Generator 8', 18, 400, 280, 280, 240, 6.02, 0, 0, 0, 0],
    ['Generator 9', 21, 400, 280, 280, 240, 5.47, 0, 0, 0, 0],
    ['Generator 10', 22, 300, 300, 300, 240, 0, 0, 0, 0, 0],
    ['Generator 11', 23, 310, 180, 180, 248, 10.52, 60, 60, 17, 16],
    ['Generator 12', 23, 350, 240, 240, 280, 10.89, 40, 40, 16, 14]],
    columns=['Name', 'Node', 'Capacity', 'Ramp up', 'Ramp down', 'Initial power', 'Bid price', 'Maximum up reserve', 'Maximum down reserve', 'Up reserve price', 'Down reserve price'])


#########################################
""" Import values of the demand units """
#########################################

# Informations on demanding units are in the dataframe 'Demands' with the columns names :
    # 'Name' for the name of the demand
    # 'Node' for the number of the node
    # 'Load' for the load of the demand
    # 'Offer price' for the offer price of the demand

# This dataframe represents the total demand load at each time step
Load_profile = pd.DataFrame([
    [1, 1775.835], [2, 1669.815], [3, 1590.3],
    [4, 1563.795], [5, 1563.795], [6, 1590.3],
    [7, 1961.37], [8, 2279.43], [9, 2517.975],
    [10, 2544.48], [11, 2544.48], [12, 2517.975],
    [13, 2517.975], [14, 2517.975], [15, 2464.965],
    [16, 2464.965], [17, 2623.995], [18, 2650.5],
    [19, 2650.5], [20, 2544.48], [21, 2411.955],
    [22, 2199.915], [23, 1934.865], [24, 1669.815]],
    columns=['Hour', 'Load'])

# This dataframe represents the load distribution over the nodes
Load_distribution = pd.DataFrame([
    ['Demand 1', 1, 0.038], ['Demand 2', 2, 0.034],
    ['Demand 3', 3, 0.063], ['Demand 4', 4, 0.026],
    ['Demand 5', 5, 0.025], ['Demand 6', 6, 0.048],
    ['Demand 7', 7, 0.044], ['Demand 8', 8, 0.06],
    ['Demand 9', 9, 0.061], ['Demand 10', 10, 0.068],
    ['Demand 11', 13, 0.093], ['Demand 12', 14, 0.068],
    ['Demand 13', 15, 0.111], ['Demand 14', 16, 0.035],
    ['Demand 15', 18, 0.117], ['Demand 16', 19, 0.064],
    ['Demand 17', 20, 0.045]],
    columns=['Name', 'Node', 'Percentage'])

# Creating the 'Demands' dataframe from the two others dataframe
# The max offer price should be higher than the last generator bid price, to be sure all demand is meet
max_bid_price = Generators['Bid price'].max()
nb_demand = len(Load_distribution)
nb_hour = len(Load_profile)
offer_price_step = round(max_bid_price*0.5/nb_demand,2)
demands = []
for i in range(nb_demand):
    demands.append([Load_distribution['Name'][i], Load_distribution['Node'][i],
                    [round(Load_distribution['Percentage'][i]*Load_profile['Load'][j],2) for j in range(nb_hour)],
                    [round(max_bid_price*1.5-offer_price_step*i,2) for j in range(nb_hour)]])
Demands = pd.DataFrame(demands,
                       columns=['Name', 'Node', 'Load', 'Offer price'])


#############################################
""" Import values of the wind farms units """
#############################################

# Informations on wind farms units are in the dataframe 'Wind_Farms' with the columns names :
    # 'Name' for the name of the wind farm
    # 'Node' for the number of the node
    # 'Capacity' for the maximum capacity of the wind farm
    # 'Bid price' for the bid price of the wind farm

# Zones 1,2,3,7,8,9 with wind profils from the 24-bus paper
# The maximum capacity depends of the time step, wind profils are list of 24 numbers
Wind_Farms = pd.DataFrame([
    ['Wind farm 1', 3, [105.85, 118.54, 136.00, 147.67, 153.35, 154.79, 155.74, 157.22, 155.57, 154.26, 149.86, 140.65, 134.44, 130.41, 127.36, 129.23, 127.01, 127.16, 131.63, 134.49, 136.04, 140.69, 136.66, 137.92], [0]*nb_hour],
    ['Wind farm 2', 5, [132.04, 139.63, 142.69, 145.88, 145.68, 146.06, 146.08, 145.28, 143.46, 144.04, 145.52, 144.69, 141.30, 140.87, 143.54, 142.99, 143.36, 141.71, 143.28, 144.06, 145.94, 147.88, 141.53, 136.34], [0]*nb_hour],
    ['Wind farm 3', 7, [121.76, 135.50, 144.34, 148.94, 150.94, 150.27, 152.54, 152.00, 148.40, 149.21, 148.78, 147.79, 145.36, 142.18, 142.78, 143.77, 142.71, 141.69, 142.54, 146.74, 147.56, 153.40, 147.92, 141.59], [0]*nb_hour],
    ['Wind farm 4', 16, [96.22, 110.38, 125.43, 134.76, 137.83, 139.24, 142.33, 139.47, 135.79, 135.21, 137.06, 138.50, 136.06, 133.07, 133.77, 133.51, 128.99, 127.32, 132.40, 140.03, 141.45, 141.94, 136.48, 127.46], [0]*nb_hour],
    ['Wind farm 5', 21, [89.38, 102.73, 114.62, 125.45, 127.22, 129.28, 133.16, 133.62, 130.83, 130.87, 131.07, 129.52, 128.38, 128.02, 128.86, 131.18, 130.88, 129.45, 135.31, 141.03, 141.03, 138.99, 134.75, 130.12], [0]*nb_hour],
    ['Wind farm 6', 23, [67.81, 72.66, 77.33, 81.73, 81.60, 81.69, 82.84, 82.25, 80.33, 81.27, 81.52, 80.33, 80.37, 79.47, 80.72, 82.35, 83.24, 83.18, 85.83, 86.03, 85.41, 84.39, 82.34, 80.89], [0]*nb_hour]],
    columns=['Name', 'Node', 'Capacity', 'Bid price'])


##########################
""" Transmission lines """
##########################

# The data frame 'Transmission' store the informations about all lines as we get them from the article
# It is used to create other variables that are more easy to use for our problems
Transmission = pd.DataFrame([
    ['Line 1', 1, 2, 0.0146, 175], ['Line 2',1, 3, 0.2253, 175], ['Line 3',1, 5, 0.0907, 350], ['Line 4',2, 4, 0.1356, 175], ['Line 5',2, 6, 0.205, 175], ['Line 6',3, 9, 0.1271, 175],
    ['Line 7',3, 24, 0.084, 400], ['Line 8',4, 9, 0.111, 175], ['Line 9',5, 10, 0.094, 350], ['Line 10',6, 10, 0.0642, 175], ['Line 11',7, 8, 0.0652, 350], ['Line 12',8, 9, 0.1762, 175],
    ['Line 13',8, 10, 0.1762, 175], ['Line 14',9, 11, 0.0084, 400], ['Line 15',9, 12, 0.084, 400], ['Line 16',10, 11, 0.084, 400], ['Line 17',10, 12, 0.084, 400], ['Line 18',11, 13, 0.0488, 500],
    ['Line 19',11, 14, 0.0426, 500], ['Line 20',12, 13, 0.0488, 500], ['Line 21',12, 23, 0.0985, 500], ['Line 22',13, 23, 0.0884, 500], ['Line 23',14, 16, 0.0594, 500], ['Line 24',15, 16, 0.0172, 500],
    ['Line 25',15, 21, 0.0249, 1000], ['Line 26',15, 24, 0.0529, 500], ['Line 27',16, 17, 0.0263, 500], ['Line 28',16, 19, 0.0234, 500], ['Line 29',17, 18, 0.0143, 500], ['Line 30',17, 22, 0.1069, 500],
    ['Line 31',18, 21, 0.0132, 1000], ['Line 32',19, 20, 0.0203, 1000], ['Line 33',20, 23, 0.0112, 1000], ['Line 34',21, 22, 0.0692, 500]],
    columns=['Name', 'From', 'To', 'Reactance', 'Capacity'])
Transmission = Transmission.rename(columns={'Reactance': 'Susceptance'})

# The array 'Line_susceptance' represent the susceptance of the line between two nodes (24x24), 0 if not line
# The array 'Line_capacity' represent the capacity of the line between two nodes (24x24), 0 if not line
Line_susceptance = np.zeros((24,24))
Line_capacity = np.zeros((24,24))
for i in range(len(Transmission)) :
    Transmission.loc[i, 'Susceptance'] = 1/Transmission['Susceptance'][i]
    # It has been decided to put all susceptance to 500 for the purpose of the assignement
    Transmission.loc[i, 'Susceptance'] = 500
    # Extract one by one each element of the 'Transmission' dataframe and put it in the right table
    t = Transmission.iloc[i].tolist()
    name, idf, idt, sus, cap = t
    Line_susceptance[idf-1,idt-1] = sus
    Line_susceptance[idt-1,idf-1] = sus
    Line_capacity[idf-1,idt-1] = cap
    Line_capacity[idt-1,idf-1] = cap


#############################
""" Nodal and zonal study """
#############################

# The dictionnay 'Nodes', 'Zones_2', and 'Zones_3' helps representing the spatial distribution of the network
# Each element of the spatial distribution (one node, one zone) is represented by a key (nb of the node or nb of the zone) and associated with a dictionnary containing :
    # At the key 'D' the id(s) of the demand related in a list
    # At the key 'G' the id(s) of the conventional generators related in a list
    # At the key 'W' the id(s) of the wind-farms related in a list
    # At the key 'L' the informations about the transmission line in a list of lists on the format :
        # First element, the id of the node/zone the line is going to
        # Second element, the susceptance of the line
        # Third element, the capacity of the line
# The difference of Zones_2 and Zones_3 is the number of zone. For this study, it has been considered 3 zones.
# WARNING : The index in the Nodes/Zones dictionnary need to be reduced by one when we search in the index of the optimization variables
        

Nodes = {}

Zones_2 = {}
Zones_2_1 = [1,2,3,4,5,6,7,8,9,10,11,12,13]
Zones_2_2 = [14,15,16,17,18,19,20,21,22,23,24]

Zones_3 = {}
Zones_3_1 = [1,2,3,4,5,6,7,8,9,10]
Zones_3_2 = [11,12,13,14,19,20,22,23,24]
Zones_3_3 = [15,16,17,18,21]


# Initialization of the dictionnaries
for i in range(1,25) :
    Nodes[i] = {"D" : [], "G" : [], "W" : [], "L" : []}
for i in range(1,3) :
    Zones_2[i] = {"D" : [], "G" : [], "W" : [], "L" : []}
for i in range(1,4) :
    Zones_3[i] = {"D" : [], "G" : [], "W" : [], "L" : []}
    
# Read Demand dataframe
for j in range(len(Demands)) :
    node = Demands['Node'][j]
    Nodes[node]["D"].append(j+1)
    if node in Zones_2_1 :
        Zones_2[1]["D"].append(j+1)
    elif node in Zones_2_2 :
        Zones_2[2]["D"].append(j+1)
    if node in Zones_3_1 :
        Zones_3[1]["D"].append(j+1)
    elif node in Zones_3_2 :
        Zones_3[2]["D"].append(j+1)
    elif node in Zones_3_3 :
        Zones_3[3]["D"].append(j+1)
    
# Read Generators dataframe
for j in range(len(Generators)) :
    node = Generators['Node'][j]
    Nodes[node]["G"].append(j+1)
    if node in Zones_2_1 :
        Zones_2[1]["G"].append(j+1)
    elif node in Zones_2_2 :
        Zones_2[2]["G"].append(j+1)
    if node in Zones_3_1 :
        Zones_3[1]["G"].append(j+1)
    elif node in Zones_3_2 :
        Zones_3[2]["G"].append(j+1)
    elif node in Zones_3_3 :
        Zones_3[3]["G"].append(j+1)
    
# Read Wind Farm dataframe
for j in range(len(Wind_Farms)) :
    node = Wind_Farms['Node'][j]
    Nodes[node]["W"].append(j+1)
    if node in Zones_2_1 :
        Zones_2[1]["W"].append(j+1)
    elif node in Zones_2_2 :
        Zones_2[2]["W"].append(j+1)
    if node in Zones_3_1 :
        Zones_3[1]["W"].append(j+1)
    elif node in Zones_3_2 :
        Zones_3[2]["W"].append(j+1)
    elif node in Zones_3_3 :
        Zones_3[3]["W"].append(j+1)
    
# Read Transmission dataframe
for j in range(len(Transmission)) :
    node_from = Transmission['From'][j]
    node_to = Transmission['To'][j]
    susceptance = Transmission['Susceptance'][j]
    capacity = Transmission['Capacity'][j]
    Nodes[node_from]["L"].append([node_to, susceptance, capacity])
    Nodes[node_to]["L"].append([node_from, susceptance, capacity])
    

# Function to input easily the transmission capacity in the Zones dict
def Transmission_input(Zones, T) :
    Zones_new = copy.deepcopy(Zones)
    for t in T :
        node, nodep, back, forth = t
        Zones_new[node]["L"].append([nodep,back,forth])
        Zones_new[nodep]["L"].append([node,forth,back])
    return(Zones_new)








