# importing relevant modules

import numpy as np
from scipy.optimize import fsolve
from math import pi



def testing_2ODE(solver, initial_guess,tol, args):
    (Hopf,phase_condition_func,(b,s)) = args

    if callable(solver) is not True:
        error_message_fun = "The input 'solver' must be a callable function"
        print(error_message_fun)
        return None

    if isinstance(initial_guess,list) is not True:
        error_message_guess = "The initial guess must be list. The period should be the last entry in the list"
        print(error_message_guess)
        return None

    if isinstance(tol, float) is not True:
        error_message_int = "The values for the tolerance must be valid"
        print(error_message_int)
        return None


    # adding tests to check that the code handles errors gracefully
    if np.size(initial_guess) != 3:
        print("must specify 3 input arguments for a system of 2 ODE's")
    else:

        root = fsolve(solver, initial_guess, args = args)

        true_sol = [np.sqrt(b)*np.cos(root[-1]), np.sqrt(b)*np.sin(root[-1])]

        # calculating the error
        error =  root[:-1] - true_sol

        if np.allclose(error,[0,0],rtol=tol, atol=tol) == True:
            result = print('Test passed')
        else:
            result = print('Test failed')
        return result




def testing_3ODE(solver, initial_guess,tol, args):
    (k,phase_condition_func ,(b,s)) = args

    if callable(solver) is not True:
        error_message_fun = "The input 'solver' must be a callable function"
        print(error_message_fun)
        return None

    if isinstance(initial_guess,list) is not True:
        error_message_guess = "The initial guess must be list. The period should be the last entry in the list"
        print(error_message_guess)
        return None

    if isinstance(tol, float) is not True:
        error_message_int = "The values for the tolerance must be valid"
        print(error_message_int)
        return None

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


def testing_dirichlet(solver,k,L,T,initial_condition,tol,args, boundary_condition= 'Dirichlet',mx=20,mt=1000):

    # adding tests to check that the code handles errors gracefully
    if callable(initial_condition) is not True:
        error_message_initial_cond = "The initial condition must be a callable function"
        print(error_message_initial_cond)
        return None

    if callable(solver) is not True:
        error_message_solver= "The solver must be a callable function"
        print(error_message_solver)
        return None

    if isinstance(L,(int,float)) is not True:
        error_message_int = "The values of kappa, L and T should be integers or floats"
        print(error_message_int)
        return None

    if isinstance(tol,float) is not True:
        error_message_tol = "The tolerance must be valid"
        print(error_message_tol)
        return None

    if isinstance(T,(int,float)) is not True:
        error_message_int = "The values of kappa, L and T should be integers or floats"
        print(error_message_int)
        return None


    if isinstance(mx,int) is not True:
        error_message_ints = "mx and mt must be passed as integers"
        print(error_message_ints)
        return None

    if isinstance(mt,int) is not True:
        error_message_ints = "mx and mt must be passed as integers"
        print(error_message_ints)
        return None



    root = solver(k,L,T,initial_condition,args=args, boundary_condition= boundary_condition,mx=mx,mt=mt)

    n = []
    for i in np.linspace(0,L,mx+1):
        true_sol = 4*i - ((64/pi**3)*np.exp(T*(-9/4)*pi**2)*np.sin(pi*i/2))
        n.append(true_sol)

    error = [a - b for a, b in zip(n, root)]
    h = np.zeros(mx+1)

    if np.allclose(error,h,rtol=tol, atol=tol) == True:
        result = print('Test passed')
    else:
        result = print('Test failed')
    return




def testing_periodic(solver,k,L,T,initial_condition,tol,args, boundary_condition= 'Dirichlet',mx=20,mt=1000):

    if callable(initial_condition) is not True:
        error_message_initial_cond = "The initial condition must be a callable function"
        print(error_message_initial_cond)
        return None

    if callable(solver) is not True:
        error_message_solver= "The solver must be a callable function"
        print(error_message_solver)
        return None

    if isinstance(L,(int,float)) is not True:
        error_message_int = "The values of kappa, L and T should be integers or floats"
        print(error_message_int)
        return None

    if isinstance(tol,float) is not True:
        error_message_tol = "The tolerance must be valid"
        print(error_message_tol)
        return None

    if isinstance(T,(int,float)) is not True:
        error_message_int = "The values of kappa, L and T should be integers or floats"
        print(error_message_int)
        return None


    if isinstance(mx,int) is not True:
        error_message_ints = "mx and mt must be passed as integers"
        print(error_message_ints)
        return None

    if isinstance(mt,int) is not True:
        error_message_ints = "mx and mt must be passed as integers"
        print(error_message_ints)
        return None

    root = solver(k,L,T,initial_condition,args=args, boundary_condition= boundary_condition,mx=mx,mt=mt)

    n = []
    for i in np.linspace(0,L,mx):
        true_sol = np.exp(-4*T*pi**2)*np.cos(2*pi*i)
        n.append(true_sol)

    error = [a - b for a, b in zip(n, root)]
    h = np.zeros(mx)

    if np.allclose(error,h,rtol=tol, atol=tol) == True:
        result = print('Test passed')
    else:
        result = print('Test failed')
    return





def testing_neumann(solver,initial_condition,tol,args, boundary_condition= 'Dirichlet',mx=20,mt=1000):
    if callable(initial_condition) is not True:
        error_message_initial_cond = "The initial condition must be a callable function"
        print(error_message_initial_cond)
        return None

    if callable(solver) is not True:
        error_message_solver= "The solver must be a callable function"
        print(error_message_solver)
        return None


    if isinstance(tol,float) is not True:
        error_message_tol = "The tolerance must be valid"
        print(error_message_tol)
        return None


    if isinstance(mx,int) is not True:
        error_message_ints = "mx and mt must be passed as integers"
        print(error_message_ints)
        return None

    if isinstance(mt,int) is not True:
        error_message_ints = "mx and mt must be passed as integers"
        print(error_message_ints)
        return None

    L = 2
    T = 0.5
    k=1
    root = solver(k,L,T,initial_condition,args=args, boundary_condition= boundary_condition,mx=mx,mt=mt)

    n = []
    for i in np.linspace(0,L,mx):
        true_sol = 0.5*(5*(1-np.cos(2))) - ((20*np.cos(2) + 20)/(-4 + pi**2))*np.exp(-(pi**2)*T/2**2)*np.cos(pi*i/2)
        n.append(true_sol)

    error = [a - b for a, b in zip(n, root)]
    h = np.zeros(mx)

    if np.allclose(error,h,rtol=tol, atol=tol) == True:
        result = print('Test passed')
    else:
        result = print('Test failed')
    return





def testing_rhs(solver,T,initial_condition,tol,args, boundary_condition= 'Dirichlet',mx=20,mt=1000):
    if callable(initial_condition) is not True:
        error_message_initial_cond = "The initial condition must be a callable function"
        print(error_message_initial_cond)
        return None

    if callable(solver) is not True:
        error_message_solver= "The solver must be a callable function"
        print(error_message_solver)
        return None


    if isinstance(tol,float) is not True:
        error_message_tol = "The tolerance must be valid"
        print(error_message_tol)
        return None

    if isinstance(T,(int,float)) is not True:
        error_message_int = "The values of kappa, L and T should be integers or floats"
        print(error_message_int)
        return None


    if isinstance(mx,int) is not True:
        error_message_ints = "mx and mt must be passed as integers"
        print(error_message_ints)
        return None

    if isinstance(mt,int) is not True:
        error_message_ints = "mx and mt must be passed as integers"
        print(error_message_ints)
        return None

    L = 1
    k=2
    root = solver(k,L,T,initial_condition,args=args, boundary_condition= boundary_condition,mx=mx,mt=mt)

    n = []
    for i in np.linspace(0,L,mx):
        true_sol = y = 4*np.exp(-T*2*(3*pi)**2)*np.sin(3*pi*i) + (1/(25*2*pi**2))*np.exp(-T*2*(5*pi)**2)*np.sin(5*pi*i) + 9*np.exp(-T*(7*pi)**2)*np.sin(7*pi*i) + (1/(2*25*pi**2))*np.sin(5*pi*i)
        n.append(true_sol)

    error = [a - b for a, b in zip(n, root)]
    h = np.zeros(mx)

    if np.allclose(error,h,rtol=tol, atol=tol) == True:
        result = print('Test passed')
    else:
        result = print('Test failed')
    return
