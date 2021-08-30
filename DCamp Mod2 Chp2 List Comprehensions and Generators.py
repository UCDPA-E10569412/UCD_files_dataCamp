# -*- coding: utf-8 -*-
"""
Created on Sun Aug 29 11:29:52 2021

@author: micha
"""

print("""For loop and list comprehension syntax""")
# new_nums = [num + 1for num in nums]


# for num in nums:    
#     new_nums.append(num + 1)
#     print(new_nums)
    
#EX.1 
# Populate a list with a FOR LOOP
nums = [12, 8, 21, 3, 16] 
new_nums = []
for num in nums:    
    new_nums.append(num + 1)
    print(new_nums)
print("New Numbers from FOR loop: ~", new_nums)

""" [output expression for iterator variable in iterable] """

# A list comprehension 
nums = [12, 8, 21, 3, 16]
new_nums2 = [num + 1 for num in nums]
print("New Numbers from COMPREHENSION: >",new_nums2)


#EX.2
#List comprehension with range()
result = [num for num in range(1, 11)]
print("List comprehension with range(1, 11) is: \n",result)





print("\nConditionals in comprehensions \n")
#EX.1
# Create a list of strings: fellowship
fellowship = ['frodo', 'samwise', 'merry', 'aragorn', 'legolas', 'boromir', 'gimli']
# Create list comprehension: new_fellowship
new_fellowship = [member for member in fellowship if len(member) >= 7]
# Print the new list
print("\nnew_fellowship : ", new_fellowship)


#EX.2
z = [num ** 2for num in range(20) if num % 2 == 0]
print("\nZ is equal to : : ", z)
print("\nConditionals in comprehensions : ", num)



#EX.3
print("\nGenerator expressions\n")
#Use ( ) instead of [ ] 
result = (2 * num for num in range(10))
print("\nGenerator: ",result)
for num in result:    
    print("for num in result: ",num)
    
   
result = (num for num in range(10))
print("\nPrint list(result) : ",list(result))


#EX.4
# Create a list of strings: lannister
lannister = ['cersei', 'jaime', 'tywin', 'tyrion', 'joffrey']
# Create a generator object: lengths
lengths = (len(person) for person in lannister)
# Iterate over and print the values in lengths
i = 0
for value in lengths:
    print("Lenght of first name : ", value, lannister[i])
    i = i + 1
    

