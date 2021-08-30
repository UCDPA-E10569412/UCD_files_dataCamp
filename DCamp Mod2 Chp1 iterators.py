# -*- coding: utf-8 -*-
"""
Created on Sun Aug 29 10:43:05 2021

@author: micha
"""


"""Iterating with a for loop"""
#EX.1
employees = ['Nick', 'Lore', 'Hugo']
for employee in employees:    
    print(employee)
#EX.2
for letter in'DataCamp':
    print(letter)
#EX.3
for i in range(3,13):# give 3 to 12 #range(4): gives 0,1,2,3   
    print(i)
                            
                            
                            
                            
"""Iterating over dictionaries  """                         
pythonistas = {'hugo': 'bowne-anderson', 'francis': 'castro'}
for key, value in pythonistas.items():    
    print(key, value)
    
    
                            
                            
"""Usingenumerate()"""
#EX.1
avengers = ['hawkeye', 'iron man', 'thor', 'quicksilver']
e = enumerate(avengers)
print(type(e))

e_list = list(e)
print("\nlist(e) : ", e_list) #[(0, 'hawkeye'), (1, 'iron man'), (2, 'thor'), (3, 'quicksilver')]
                            
#EX.2 
print()                           
avengers = ['hawkeye', 'iron man', 'thor', 'quicksilver']
for key, value in enumerate(avengers):   #enumerate(avengers, start=10): 
    print("\nindex, value ", key, value) # 0 hawkeye 1 iron man 2 thor 3 quicksilver  
    print("\n index only : ", key)    
    print("\n value only : ", value)                      
                            
                            
    
                            
                            
"""zip() and unpack"""
#EX.1
avengers    = ['hawkeye', 'iron man', 'thor', 'quicksilver']
names       = ['barton', 'stark', 'odinson', 'maximoff']
for z1, z2 in zip(avengers, names):    
    print(z1, z2)     

#EX.2
mutants = ['charles xavier', 'bobby drake', 'kurt wagner', 'max eisenhardt', 'kitty pryde'] 
aliases = ['prof x', 'iceman', 'nightcrawler', 'magneto', 'shadowcat'] 
powers  = ['telepathy', 'thermokinesis', 'teleportation', 'magnetokinesis', 'intangibility']

# Create a list of tuples: mutant_data
mutant_data = list(zip(mutants, aliases, powers))

# Print the list of tuples
print("\nMutant data List of ZIP: \n",mutant_data)

# Create a zip object using the three lists: mutant_zip
mutant_zip = zip(mutants, aliases, powers)

# Print the zip object
print("\nMutant data  ZIP: \n", mutant_zip)
# Unpack the zip object and print the tuple values
for value1, value2, value3 in mutant_zip:
    print("For Loop, :\n",value1, value2, value3)
    




# """Usingiteratorstoloadlargefilesintomemory """
# #EX.1
# import pandas as pd
# result = []
# for chunk in pd.read_csv('data.csv', chunksize=1000):    
#     result.append(sum(chunk['x']))
#     total = sum(result)
#     print(total)
    

# #EX.2
# # Define count_entries()
# def count_entries(csv_file, c_size, colname):
#     """Return a dictionary with counts of
#     occurrences as value for each key."""
    
#     # Initialize an empty dictionary: counts_dict
#     counts_dict = {}

#     # Iterate over the file chunk by chunk
#     for chunk in pd.read_csv(csv_file, chunksize=c_size):

#         # Iterate over the column in DataFrame
#         for entry in chunk[colname]:
#             if entry in counts_dict.keys():
#                 counts_dict[entry] += 1
#             else:
#                 counts_dict[entry] = 1

#     # Return counts_dict
#     return counts_dict

# # Call count_entries(): result_counts
# result_counts = count_entries('tweets.csv', 10, 'lang')

# # Print result_counts
# print(result_counts)
