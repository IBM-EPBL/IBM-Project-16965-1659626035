# -*- coding: utf-8 -*-
"""DA_Assignment_3_Python.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1UnwWqw2PCFBDro1w0T__lvTKxX2dcsRL

## Exercises

Answer the questions or complete the tasks outlined in bold below, use the specific method described if applicable.

** What is 7 to the power of 4?**
"""

x=pow(7,4)
print(x)

"""** Split this string:**

    s = "Hi there Sam!"
    
**into a list. **
"""

s="Hi there Sam!"

x=s.split()
print(x)

"""** Given the variables:**

    planet = "Earth"
    diameter = 12742

** Use .format() to print the following string: **

    The diameter of Earth is 12742 kilometers.
"""

txt3 = ("The diameter of {planet} is {diameter} kilometers.").format(planet = "Earth", diameter = 12742)

print(txt3)

"""** Given this nested list, use indexing to grab the word "hello" **"""

lst = [1,2,[3,4],[5,[100,200,['hello']],23,11],1,7]

print(lst[3][1][2])

"""** Given this nest dictionary grab the word "hello". Be prepared, this will be annoying/tricky **"""

d = {'k1':[1,2,3,{'tricky':['oh','man','inception',{'target':[1,2,3,'hello']}]}]}

d['k1'][3]['tricky'][3]['target'][3]

"""** What is the main difference between a tuple and a list? **"""

#tuple is immutable while list is mutable
#Lists consume more memory while tuple consumes less memory
#Duplicates are allowed in list not in tuple
#List can be represented as [],tuples can be represented as ()

"""** Create a function that grabs the email website domain from a string in the form: **

    user@domain.com
    
**So for example, passing "user@domain.com" would return: domain.com**
"""

def splitt(input):
  return input.split('@')[1]

splitt("user@domain.com")

"""** Create a basic function that returns True if the word 'dog' is contained in the input string. Don't worry about edge cases like a punctuation being attached to the word dog, but do account for capitalization. **"""

def check(input):
   return 'dog' in input.lower().split()

check('I am going to adapt this street dog')

"""** Create a function that counts the number of times the word "dog" occurs in a string. Again ignore edge cases. **"""

message = 'dog  is my favourite pet and i love dog so much my dog name is  little programming language'

print('Number of occurrence of dog:', message.count('dog'))

"""### Problem
**You are driving a little too fast, and a police officer stops you. Write a function
  to return one of 3 possible results: "No ticket", "Small ticket", or "Big Ticket". 
  If your speed is 60 or less, the result is "No Ticket". If speed is between 61 
  and 80 inclusive, the result is "Small Ticket". If speed is 81 or more, the result is "Big    Ticket". Unless it is your birthday (encoded as a boolean value in the parameters of the function) -- on your birthday, your speed can be 5 higher in all 
  cases. **
"""

def caught_speeding(speed, is_birthday):
    
    if is_birthday:
        speeding = speed - 5
    else:
        speeding = speed
    
    if speeding > 80:
        return 'Big Ticket'
    elif speeding > 60:
        return 'Small Ticket'
    else:
        return 'No Ticket'

caught_speeding(90,False)

caught_speeding(81,True)

"""Create an employee list with basic salary values(at least 5 values for 5 employees)  and using a for loop retreive each employee salary and calculate total salary expenditure. """

employee=[12000,13456,25577,24565,34578]
tot=0
print("Employee list:")
for i in employee:
  print(i)
  tot=tot+i
print('total salary expenditure:')
print(tot)

"""Create two dictionaries in Python:

First one to contain fields as Empid,  Empname,  Basicpay

Second dictionary to contain fields as DeptName,  DeptId.

Combine both dictionaries. 
"""

def Merge(First, Second):
    return(First.update(Second))
    
First={'Empid':26, 'Empname':'Jamuna', 'Basicpay':'14 lpa'}
Second={'DeptName':'IT','DeptId':56}
Merge(First,Second)
print(First)