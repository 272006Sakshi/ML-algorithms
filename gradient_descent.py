'''
Here we will use Gradient Descent to find minima of a given function.
Let the Given Function be Y=(X+5)^2 and we have to find the minima of this given function.
We already know the minima of this function occurs when X = -5 let's find it out using Gradient Descent.
'''

current_x = 12

learning_rate = 0.01
no_of_steps = 1000
step_counter = 1


def df(x): return 2*(x+5)  # Differentiation of given Function Y=(X+5)^2


while step_counter < no_of_steps:
    prev_x = current_x
    current_x = current_x - learning_rate * \
        df(prev_x)  # Gradient Descent algorithm
    change_in_x = abs(current_x-prev_x)
    step_counter += 1
    print("Step:", step_counter, " || Value of X", current_x)

# Changing the value of no_of_steps and learning rate we can get closer to the minima of a function.
# This was the simple Gradient Descent Algorithm which is used to find minima of a function (generally loss function in machine learning)
