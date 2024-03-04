""" Code for importing the relevant data """

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


###################################
""" Variables used in this file """
###################################

# Informations on generating units are in the dataframe 'Generators' with the columns names :
    # 'Name' for the name of the generator
    # 'Node' for the number of the node
    # 'Capacity' for the maximum capacity of the generator
    # 'Ramp up' for the ramp up limit of the generator
    # 'Ramp down' for the ramp down limit of the generator
    # 'Bid price' for the bid price of the generator
    
# Informations on wind farms units are in the dataframe 'Wind_Farms' with the columns names :
    # 'Name' for the name of the wind farm
    # 'Node' for the number of the node
    # 'Capacity' for the maximum capacity of the wind farm
    # 'Bid price' for the bid price of the wind farm
    
# Informations on demanding units are in the dataframe 'Demands' with the columns names :
    # 'Name' for the name of the demand
    # 'Node' for the number of the node
    # 'Load' for the load of the demand
    # 'Offer price' for the offer price of the demand
    
  
#############################################
""" Import values of the generators units """
#############################################

Generators = pd.DataFrame([
    ['Generator 1', 1, 152, 120, 120, 76, 13.32],
    ['Generator 2', 2, 152, 120, 120, 76, 13.32],
    ['Generator 3', 7, 350, 350, 350, 0, 20.7],
    ['Generator 4', 13, 591, 240, 240, 0, 20.93],
    ['Generator 5', 15, 60, 60, 60, 0, 26.11],
    ['Generator 6', 15, 155, 155, 155, 0, 10.52],
    ['Generator 7', 16, 155, 155, 155, 124, 10.52],
    ['Generator 8', 18, 400, 280, 280, 240, 6.02],
    ['Generator 9', 21, 400, 280, 280, 240, 5.47],
    ['Generator 10', 22, 300, 300, 300, 240, 0],
    ['Generator 11', 23, 310, 180, 180, 248, 10.52],
    ['Generator 12', 23, 350, 240, 240, 280, 10.89]],
    columns=['Name', 'Node', 'Capacity', 'Ramp up', 'Ramp down', 'Initial power', 'Bid price'])


#########################################
""" Import values of the demand units """
#########################################

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

# I took zone 1,2,3,7,8,9
Wind_Farms = pd.DataFrame([
    ['Wind farm 1', 3, [105.8518942, 118.5416375, 136.0025846, 147.6747565, 153.3498415, 154.7855366, 155.7408754, 157.2236407, 155.5708732, 154.2616933, 149.8631035, 140.6540638, 134.4390929, 130.4065771, 127.3607862, 129.230793, 127.0150452, 127.1557626, 131.630263, 134.4943655, 136.0450335, 140.6879798, 136.6616767, 137.9242212], [0]*nb_hour],
    ['Wind farm 2', 5, [132.0365009, 139.6351285, 142.6908722, 145.885015, 145.6857226, 146.0575584, 146.0833968, 145.2777323, 143.4593029, 144.0384271, 145.5188664, 144.6885944, 141.2975305, 140.8724154, 143.5424666, 142.9933626, 143.3615159, 141.7135793, 143.2837817, 144.0653615, 145.9442153, 147.8793794, 141.532961, 136.3428801], [0]*nb_hour],
    ['Wind farm 3', 7, [121.7655941, 135.5020455, 144.338065, 148.9438578, 150.9452044, 150.2716884, 152.5396741, 151.9984083, 148.4043069, 149.2154754, 148.7762769, 147.7897076, 145.3573624, 142.1846647, 142.7763901, 143.7663529, 142.711523, 141.6913861, 142.5436727, 146.7421655, 147.5593436, 153.3961561, 147.9233616, 141.5863522], [0]*nb_hour],
    ['Wind farm 4', 16, [96.22143521, 110.375753, 125.4343489, 134.7639093, 137.833016, 139.2377607, 142.3293848, 139.4687097, 135.7889317, 135.2141593, 137.057534, 138.5010498, 136.0619849, 133.0730888, 133.7750521, 133.5091541, 128.9947943, 127.3198906, 132.3990695, 140.0305324, 141.455082, 141.9415908, 136.4852693, 127.4651946], [0]*nb_hour],
    ['Wind farm 5', 21, [89.37566837, 102.7333585, 114.6163661, 125.4500056, 127.2155529, 129.2787442, 133.1578469, 133.6247378, 130.8260382, 130.8722442, 131.071553, 129.5253217, 128.3853801, 128.0208993, 128.8564491, 131.1777688, 130.8831642, 129.4530387, 135.3147408, 141.0256722, 141.0290586, 138.9928173, 134.7478994, 130.120545], [0]*nb_hour],
    ['Wind farm 6', 23, [67.81164861, 72.65732623, 77.33228393, 81.72946758, 81.60447568, 81.68703202, 82.84271004, 82.24553017, 80.32702269, 81.26731038, 81.52175236, 80.33102453, 80.37375837, 79.47286975, 80.7187049, 82.34650655, 83.23783329, 83.18344086, 85.83028167, 86.0307012, 85.40761746, 84.38889089, 82.34007379, 80.89224889], [0]*nb_hour]],
    columns=['Name', 'Node', 'Capacity', 'Bid price'])


##########################
""" Transmission lines """
##########################

Transmission = pd.DataFrame([
    ['Line 1', 1, 2, 0.0146, 175], ['Line 2',1, 3, 0.2253, 175], ['Line 3',1, 5, 0.0907, 350], ['Line 4',2, 4, 0.1356, 175], ['Line 5',2, 6, 0.205, 175], ['Line 6',3, 9, 0.1271, 175],
    ['Line 7',3, 24, 0.084, 400], ['Line 8',4, 9, 0.111, 175], ['Line 9',5, 10, 0.094, 350], ['Line 10',6, 10, 0.0642, 175], ['Line 11',7, 8, 0.0652, 350], ['Line 12',8, 9, 0.1762, 175],
    ['Line 13',8, 10, 0.1762, 175], ['Line 14',9, 11, 0.0084, 400], ['Line 15',9, 12, 0.084, 400], ['Line 16',10, 11, 0.084, 400], ['Line 17',10, 12, 0.084, 400], ['Line 18',11, 13, 0.0488, 500],
    ['Line 19',11, 14, 0.0426, 500], ['Line 20',12, 13, 0.0488, 500], ['Line 21',12, 23, 0.0985, 500], ['Line 22',13, 23, 0.0884, 500], ['Line 23',14, 16, 0.0594, 500], ['Line 24',15, 16, 0.0172, 500],
    ['Line 25',15, 21, 0.0249, 1000], ['Line 26',15, 24, 0.0529, 500], ['Line 27',16, 17, 0.0263, 500], ['Line 28',16, 19, 0.0234, 500], ['Line 29',17, 18, 0.0143, 500], ['Line 30',17, 22, 0.1069, 500],
    ['Line 31',18, 21, 0.0132, 1000], ['Line 32',19, 20, 0.0203, 1000], ['Line 33',20, 23, 0.0112, 1000], ['Line 34',21, 22, 0.0692, 500]],
    columns=['Name', 'From', 'To', 'Reactance', 'Capacity'])

for i in range(len(Transmission)) :
    Transmission.loc[i, 'Reactance'] = 1/Transmission['Reactance'][i]
    
Transmission = Transmission.rename(columns={'Reactance': 'Susceptance'})

Nodes = {}

for i in range(1,25) :
    Nodes[i] = {"D" : [], "G" : [], "W" : [], "L" : []}
    
# Read Demand dataframe
for j in range(len(Demands)) :
    node = Demands['Node'][j]
    Nodes[node]["D"].append(j+1)
    
# Read Generators dataframe
for j in range(len(Generators)) :
    node = Generators['Node'][j]
    Nodes[node]["G"].append(j+1)
    
# Read Wind Farm dataframe
for j in range(len(Wind_Farms)) :
    node = Wind_Farms['Node'][j]
    Nodes[node]["W"].append(j+1)
    
# Read Transmission dataframe
for j in range(len(Transmission)) :
    node_from = Transmission['From'][j]
    node_to = Transmission['To'][j]
    susceptance = Transmission['Susceptance'][j]
    capacity = Transmission['Capacity'][j]
    Nodes[node_from]["L"].append([node_to, susceptance, capacity])
    Nodes[node_to]["L"].append([node_from, susceptance, capacity])
    
# WARNING : The index in the Nodes dictionnary need to be reduced by one when we search in the index of the optimization variables


















