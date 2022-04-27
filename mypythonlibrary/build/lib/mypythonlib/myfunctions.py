# import relevant modules
import numpy as np
import math
import matplotlib.pyplot as plt
from scipy.integrate import odeint
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve
from scipy.integrate import solve_ivp
import numpy as np
import pylab as pl
from math import pi
from scipy.sparse import diags

def euler_solve_step(fun,h,t,x):
    """
    A function that performs a single step of the euler method

    Parameters
    ----------
    fun : function
        The ODE we wish to solve. The ode function should take
        a single parameter (the state vector) and return the
        right-hand side of the ODE as a numpy.array.

    h : value(int or float)
        step_size

    t:  value(int or float)
        the initial time

    x: value(int or float)
        The initial condition for the ODE.

    Returns
    -------
    Returns the value of x after a single step of the euler method
    """

    x = x + h*fun(x,t)
    return  x



def rk4_solve_step(fun,h,t,x):
    """
    A function that performs a single step of the 4th-order Runge-Kutta method

    Parameters
    ----------
    fun : function
        The ODE we wish to solve. The ode function should take
        a single parameter (the state vector) and return the
        right-hand side of the ODE as a numpy.array.

    h : value(int or float)
        step_size

    t:  value(int or float)
        the initial time

    x: value(int or float)
        The initial condition for the ODE.

    Returns
    -------
    Returns the value of x after a single step of the 4th-order Runge-Kutta method
    """
    k1 = fun(x,t)
    k2 = fun( x + h*(k1/2),t + h/2)
    k3 = fun( x + h*(k2/2),t + h/2)
    k4 = fun( x +h*k3,t + h)
    return  x + (1/6)*h*(k1 + 2*k2 + 2*k3 + k4)



def heuns_solve_step(fun,h,t,x):
    """
    A function that performs a single step of the Heun's method

    Parameters
    ----------
    fun : function
        The ODE we wish to solve. The ode function should take
        a single parameter (the state vector) and return the
        right-hand side of the ODE as a numpy.array.

    h : value(int or float)
        step_size

    t:  value(int or float)
        the initial time

    x: value(int or float)
        The initial condition for the ODE.

    Returns
    -------
    Returns the value of x after a single step of the Heun's method
    """
    f1 = fun(x,t)
    f2 = fun( x + h*(2/3),t + h*(2/3)*f1)
    return  x + h*(f1 + 3*f2)/4


def midpoint_solve_step(fun,h,t,x):
    """
    A function that performs a single step of the Midpoint method

    Parameters
    ----------
    fun : function
        The ODE we wish to solve. The ode function should take
        a single parameter (the state vector) and return the
        right-hand side of the ODE as a numpy.array.

    h : value(int or float)
        step_size

    t:  value(int or float)
        the initial time

    x: value(int or float)
        The initial condition for the ODE.

    Returns
    -------
    Returns the value of x after a single step of the Midpoint method
    """
    x = x + (h*fun(x+(h/2), t +(h/2)*fun(x,t)))
    return  x


def solve_to(f,h,t0,u0,T,solver):
    """
    A function that uses a specified 1-step integration method to solve steps of an ODE.

    Parameters
    ----------
    fun : function
        The ODE we wish to solve. The ode function should take
        a single parameter (the state vector) and return the
        right-hand side of the ODE as a numpy.array.

    h : value(int or float)
        step_size

    t0:  value(int or float)
        the initial time

    u0: value(int or float)
        The initial condition for the ODE.

    T: value(int or float)
        The maximum steps

    Solver: function
            The numerical integration method used to solve the ODE. Specify
            'euler_solve_step' for the Euler method, 'rk4_solve_step' for the
            4th-order Runge-Kutta method, 'heuns_solve_step' for the Heuns method
            or 'midpoint_solve_step' for the midpoint method.



    Returns
    -------
    Returns the numerical solution estimates of ODE between
    x1 at t1 and x2 at t2.
    """

    if h > deltat_max:
        print('The step-size specified is too large')
    else:
        while t0+h < T:
            u0=solver(f,h,t0,u0)
            t0 = t0 + h
        if T!=t0:
            u0=solver(f,T-t0,t0,u0)
            t0 = T
        return u0


def solve_ode(fun,h,t0,u0,L,solver):

    """
    A function that uses a specified 1-step integration method to solve a specified ODE.

    Parameters
    ----------
    fun : function
        The ODE we wish to solve. The ode function should take
        a single parameter (the state vector) and return the
        right-hand side of the ODE as a numpy.array.

    h : value(int or float)
        step_size

    t0:  value(int or float)
        the initial time

    u0: value(int or float)
        The initial condition for the ODE.

    L:

    Solver: function
            The numerical integration method used to solve the ODE. Specify
            'euler_solve_step' for the Euler method, 'rk4_solve_step' for the
            4th-order Runge-Kutta method, 'heuns_solve_step' for the Heuns method
            or 'midpoint_solve_step' for the midpoint method.



    Returns
    -------
    Returns a numpy.array containing the numerical solution estimates of ODE
    for the specified time interval as well as the error.
    """

    m = []
    for l in L:
        u0 = solve_to(fun,h,t0,u0,l,solver)
        t0=l
        m.append(u0)
    error = abs(math.exp(l)-u0)
    return m, error


def phase_condition_func(func, u, T, args):
    return func(T,u,*args)[0]


#Construct the shooting root-finding problem
def shooting(u0, function, phase_condition,args):
    """
    A function that uses numerical shooting to find limit cycles of
    a specified ODE.

    Parameters
    ----------
     u0 : numpy.array
        An initial guess at the initial values for the limit cycle.

    fun : function
        The ODE to apply shooting to. The ode function should take
        a single parameter (the state vector) and return the
        right-hand side of the ODE as a numpy.array.

    phase_condition: function
                    The phase condition for the limit cycle.

    args: tuple
        arguments passed for the numerical shooting

    Returns
    -------
    Returns a numpy.array containing the corrected initial values
    for the limit cycle.
    """
    u, T = u0[:-1], u0[-1]
    if u[0]== 0:
        print('the first value of initial condition cannot be zero')
    else:
        sol = solve_ivp(function, (0,T), u, args = args, rtol = 1e-6)
        final_states = sol.y[:,-1]
        phase = np.array([phase_condition(function,u,T,args)])
        return np.concatenate((u-final_states, phase))



# for the cubic equation
def Numerical_Continuation_x(initial_guess, fun, start, end, h, parameter):
    """
    A function that performs natural parameter continution for specified
    equations or ODE. For example, it increments a parameter by a set amount and
    attempts to find the solution for the new parameter value using the
    last found solution as an initial guess.

    Parameters
    ----------
    initial_guess : np.array
        The initial guess is an array of the initial guess values for the
        equations followed by a period.

    fun : function
        The ODE we wish to solve. The ode function should take
        a single parameter (the state vector) and return the
        right-hand side of the ODE as a numpy.array.

    start:  value(int or float)
        The initial value for the parameter

    end: value(int or float)
        The end value for the parameter

    h: value(int or float)
        The step-size we wish to increment the parameter by

    parameter: variable name
              The parameter for the ODE (fun) we wish to vary

    Returns
    -------
    Returns a graph showing how the solution of the ODE/equations
    changes as the parameter is incremented from start to end value.
    """

    x = []
    y = []
    period = []
    for parameter in np.arange(start,end,h):
        #root = fsolve(shooting,initial_guess,args = (equations, (b,s)))
        root = fsolve(fun,initial_guess,args=(parameter))
        initial_guess = root
        x.append(root[0])
        #y.append(root[1])
        #period.append(root[2])
    plt.xlabel('x')
    plt.ylabel('solution')
    plt.title('Natural parameter continuation of the cubic equation')
    plot = plt.plot(np.arange(start,end,h), x)
    #plot = plt.plot(np.arange(start,end,stepsize), y)
    #plot = plt.plot(np.arange(start,end,stepsize), period)
    plt.legend(['x','y','period'], loc = "upper right")
    return


#def Numerical_Continuation(initial_guess ,start, end, h,, fun):
def Numerical_Continuation(initial_guess ,start, end, h, b, s, fun):
    """
    A function that performs natural parameter continution for a system of
    2 ODE's. It increments the parameter beta (b) by a set amount and
    attempts to find the solution for the new parameter value using the
    last found solution as an initial guess.

    Parameters
    ----------
    initial_guess : np.array
        The initial guess is an array of the initial guess values for the
        equations followed by a period.

    fun : function
        The ODE we wish to solve. The ode function should take
        a single parameter (the state vector) and return the
        right-hand side of the ODE as a numpy.array. Input 'Hopf'
        or 'mod_Hopf'.

    start:  value(int or float)
        The initial value for the parameter

    end: value(int or float)
        The end value for the parameter

    h: value(int or float)
        The step-size we wish to increment the parameter by. Note this
        value should be negative if the start value is positive and the
        end value is negative

    Returns
    -------
    Returns a graph showing how the solution of the ODE/equations
    changes as the value of beta (b) is incremented from start to end value.
    """
    #if fun == Hopf:
       # title = 'the Hopf bifurcation normal form'
   # elif fun == mod_Hopf:
        #title = 'the modified Hopf bifurcation normal form'
    #else:
       # raise ValueError("Invalid function. Please specify 'Hopf' or 'mod_Hopf'")



    x = []
    y = []
    period = []
    #variable_parameter = args1[0]

    for b in np.arange(start,end,h):
        root = fsolve(shooting,initial_guess,args = (fun, phase_condition_func, (b,s)))
        initial_guess = root
        x.append(root[0])
        y.append(root[1])
        period.append(root[2])

    fig=plt.figure(figsize=(8,6))
    plt.xlabel('\u03B2')
    plt.ylabel('solution')
    #plt.title(f'Natural parameter continuation of \u03B2 for the {title} equations')
    plot = plt.plot(np.arange(start,end,h), x)
    plot = plt.plot(np.arange(start,end,h), y)
    #plot = plt.plot(np.arange(start,end,h), period)
    plt.legend(['x','y'], loc = "upper right")
    return



# function to solve PDE using euler method
def PDE_solve_euler(kappa,L,T,initial_condition, args=(), boundary_condition=None,mx=20,mt=1000):

    """
    A function that the euler method to solve the 1D heat equation.

    Parameters
    ----------

    initial_condition:

    args:

    boundary_condition:


    Returns
    -------

    """
    #mt = 1000
    #mx = 20
    # Set up the numerical environment variables
    x = np.linspace(0, L, mx+1)     # mesh points in space
    t = np.linspace(0, T, mt+1)     # mesh points in time
    deltax = x[1] - x[0]            # gridspacing in x
    deltat = t[1] - t[0]            # gridspacing in t
    lmbda = kappa*deltat/(deltax**2)    # mesh fourier number



    u_j = np.zeros(x.size)        # u at current time step
    u_jp1 = np.zeros((mx+1,1))
    k = [lmbda*np.ones(mx-2),(1-2*lmbda)*np.ones(mx-1),lmbda*np.ones(mx-2)]
    offset = [-1,0,1]
    A = diags(k,offset,shape=(mx-1, mx-1)).toarray()

    if boundary_condition == 'periodic':
        u_j = np.zeros(mx)
        k = [lmbda*np.ones(mx-1),(1-2*lmbda)*np.ones(mx),lmbda*np.ones(mx-1)]

        offset = [-1,0,1]
        A = diags(k,offset).toarray()
        A[-1,0] = lmbda
        A[0,-1] = lmbda
        for i in range(0, mx):
            u_j[i] = initial_condition(x[i])
        for j in range(0, mt):
            u_j = periodic_boundary(A, u_j)


    elif boundary_condition == 'rhs':

       # u_jp1 = np.zeros(x.size)
        u_jp1 = np.zeros((mx+1,1))
        k = [lmbda*np.ones(mx-2),(1-2*lmbda)*np.ones(mx-1),lmbda*np.ones(mx-2)]
        F_j = args

       # k = [lmbda*np.ones(mx),(1-2*lmbda)*np.ones(mx+1),lmbda*np.ones(mx)]
        offset = [-1,0,1]
        A = diags(k,offset).toarray()

        for i in range(0, mx+1):
            u_j[i] = initial_condition(x[i])

        for j in range(0, mt):
            u_j = rhs_function(A, u_j,F_j,u_jp1,j)


    else:
        if callable(initial_condition) is False:
            u_j = initial_condition


        else:
            for i in range(0, mx+1):
                u_j[i] = initial_condition(x[i])



        for j in range(0, mt):
            if boundary_condition == 'dirichlet':
                # s = np.zeros((mx-1,1))
                p = args[0]
                q = args[1]
                s = np.zeros((mx-1,1))
                s[0] = p(t[j])
                s[-1] = q(t[j])
                 #u_j = dirichlet(A,s,u_j,u_jp1)
                u_j = dirichlet_boundary(A, u_j,j,s,u_jp1,lmbda,t,mx,args=args)


            elif boundary_condition== 'neumann':
                k = [lmbda*np.ones(mx),(1-2*lmbda)*np.ones(mx+1),lmbda*np.ones(mx)]
                offset = [-1,0,1]
                A = diags(k,offset).toarray()
                A[0,1] = 2*lmbda
                A[-1,-2] = 2*lmbda
                s = np.zeros((mx+1,1))

                P = args[0]
                Q = args[1]
                s[0] = -P(t[j])
                s[-1] = Q(t[j])
                u_j = Neumann_boundary(A,s,u_j,args=args)

            elif boundary_condition == None:
                u_jp1[1:-1] = np.matmul(A , u_j[1:-1]).reshape(19,1)
                u_j = u_jp1

        return u_j
    return u_j


def Numerical_Continuation_kappa(L, T, initial_condition, start, end, h, args, boundary_condition=None, mx=20, mt=1000):



    n = []
    #root = PDE_solve_euler(,L,T,initial_condition, args=args, boundary_condition=boundary_condition,mx=20,mt=1000)
    #initial_condition = root



    for parameter in np.arange(start,end,h):

        #root = PDE_solve_euler(initial_condition, args=args, boundary_condition=boundary_condition)
        root = PDE_solve_euler(parameter,L,T,initial_condition, args=args, boundary_condition=boundary_condition,mx=20,mt=1000)
        initial_condition = root
        n.append(root)

    i=0
    for i in range(len(n)):
    #print(n[0])
        pl.plot(np.linspace(0, L, mx+1),n[i],'ro',label='num')

    #n[-1]
    #pl.plot(np.linspace(0, L, mx+1),root,'ro',label='num')

    return










