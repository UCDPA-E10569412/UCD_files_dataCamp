# -*- coding: utf-8 -*-
"""
Created on Fri Aug 27 20:04:22 2021

@author: micha
"""

# Def Function(Parameter1, Parameter2):
#     """Docstrings"""
#     new_value = Parameter1 + Parameter2
#     return new_value

#Ans = funtion(Argument1, argument3)
#print(Ans)

"""Function"""

X=5 #Global variable identified in main body of script
z=12

# Define shout_all with parameters word1 and word2
def shout_all(word1="incase nothing is passed", word2="Hi"): #args // for num in args: sum_all+= num
    """Return a tuple of strings"""
    print("z1 is :", z)
    y=5#Local scope as its define in a function
    global X     #can acces and alter variable inm main body
    x=5
    print("x is :", x)
    
    
    # Concatenate word1 with '!!!': shout1
    shout1 = word1 + '!!!'
    
    # Concatenate word2 with '!!!': shout2
    shout2 = word2 + '!!!'
    
    # Construct a tuple with shout1 and shout2: shout_words is a Tuple
    shout_words = (shout1, shout2)

    # Return shout_words
    return shout_words

# Pass 'congratulations' and 'you' to shout_all(): yell1, yell2
# Unpacking Tuple
yell1, yell2 = shout_all('congratulations', 'you')

# Print yell1 and yell2
print(yell1)
print(yell2)



"""Lambda"""
#Example 1
raise_to_power = lambda x,y: x**y
print("lambda = ",raise_to_power(2,3))


#Example 2
# Define echo_word as a lambda function: echo_word
echo_word = (lambda word1, echo : word1 * echo)
# Call echo_word: result
result = echo_word('hey', 5)
# Print result
print(result)

#Example 3
# Create a list of strings: fellowship
fellowship = ['frodo', 'samwise', 'merry', 'pippin', 'aragorn', 'boromir', 'legolas', 'gimli', 'gandalf']
# Use filter() to apply a lambda function over fellowship: result
result = filter(lambda member: len(member) > 6, fellowship)
# Convert result to a list: result_list
result_list = list(result)
# Print result_list
print(result_list)



"""Error handling"""
print("Error handling")
def sqrt(x):
    """Returns the square root of a number."""
    if x < 0:
        raise ValueError('x must be non-negative')
    try:
        return x ** 0.5
    
    except TypeError:       
        print('x must be an int or float')

print("error handling, SQRT (2) = ", sqrt(2))
print("error handling, SQRT(-2) = ", sqrt(-2))