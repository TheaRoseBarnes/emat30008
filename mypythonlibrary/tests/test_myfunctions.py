# importing relevant modules
from scipy.integrate import odeint
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve
from scipy.integrate import solve_ivp
from scipy.optimize import fsolve


# define a test script that runs the shooting code and checks it against its true solution
# works for 2 ODEs
# checking that a function produces the correct output for a given input
def testing_2ODE(solver, initial_guess, args):
    (Hopf,phase_condition_func,(b,s)) = args

    # adding tests to check that the code handles errors gracefully
    if np.size(initial_guess) != 3:
        print("must specify 3 input arguments for a system of 2 ODE's")
    else:

        root = fsolve(solver, initial_guess, args = args)


        # defining the true solution
        #true_sol = [np.sqrt(b)*np.cos(initial_guess[-1]+root[-1]), np.sqrt(b)*np.sin(initial_guess[-1]+root[-1])]
        true_sol = [np.sqrt(b)*np.cos(root[-1]), np.sqrt(b)*np.sin(root[-1])]

        # calculating the error
        error =  root[:-1] - true_sol

        if np.allclose(error,[0,0],rtol=1e-04, atol=1e-04) == True:
            result = print('Test passed')
        else:
            result = print('Test failed')
        return result



# define a test script that runs the shooting code and checks it against its true solution
# works for 3 ODEs
# checking that a function produces the correct output for a given input
def testing_3ODE(solver, initial_guess, args):
    (k,phase_condition_func ,(b,s)) = args

    # adding tests to check that the code handles errors gracefully
    if np.size(initial_guess) != 4:
        print("must specify 4 input arguments for a system of 3 ODE's")
    else:

        root = fsolve(solver, initial_guess,args = args)

        # defining the true solution
        true_sol = [np.sqrt(b)*np.cos(root[-1]), np.sqrt(b)*np.sin(root[-1]), np.exp(-root[-1])]


        # calculating the error
        error =  root[:-1] - true_sol

        if np.allclose(error,[0,0,0],rtol=1e-03, atol=1e-03) == True:
            result = print('Test passed')
        else:
            result = print('Test failed')
        return result
