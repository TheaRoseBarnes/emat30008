# import relevant modules

import math
import matplotlib.pyplot as plt
from scipy.optimize import fsolve
from scipy.integrate import solve_ivp
import numpy as np
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

deltat_max = 1
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

def solve_ode_system(fun,h,t,u0,L,solver='rk4_solve_step'):
    """
    A function that uses a specified 1-step integration method to solve a system
    of specified ODE's.

    Parameters
    ----------
    fun : function
        The system of ODE's we wish to solve. The ode function should take
        a single parameter (the state vector) and return the
        right-hand side of the ODE as a numpy.array.

    h : value(int or float)
        step_size

    t:  value(int or float)
        the initial time

    u0:  numpy.array
        The initial conditions for the ODE.

    L:

    Solver: function
            The numerical integration method used to solve the ODE. Specify
            'euler_solve_step' for the Euler method, 'rk4_solve_step' for the
            4th-order Runge-Kutta method, 'heuns_solve_step' for the Heuns method
            or 'midpoint_solve_step' for the midpoint method.



    Returns
    -------
    Returns a list of numpy.arrays containing the x and y numerical solution estimates to the
    system of ODE's for the specified time interval.
    """
     # adding tests to check that the code handles errors gracefully
    #if solver != 'euler_solve_step': or 'rk4_solve_step' or 'heuns_solve_step' or 'midpoint_solve_step':
        #print("INVALID SOLVER. Please specify: euler_solve_step, rk4_solve_step, heuns_solve_step or midpoint_solve_step")
   # else:

    m = []
    for l in L:
        u0 = solve_to(fun,h,t,u0,l,solver)
        t=l
        m.append(u0)
    return m

def phase_condition_func(func, u, T, args):
    return func(T,u,*args)[0]



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

    if callable(function) is not True:
        error_message_fun = "The input 'function' must be a callable function"
        print(error_message_fun)
        return None

    #if isinstance(u0,list) is not True:
       # error_message_guess = "The initial guess u0 must be list. The period should be the last entry in the list"
       # print(error_message_guess)
        #return None

    if callable(phase_condition) is not True:
        error_message_phase = "The input 'phase_condition' must be a callable function"
        print(error_message_phase)
        return None

    if isinstance(args, (tuple, None)) is not True:
        error_message_args = "The arguments must be passed as a tuple"
        print(error_message_args)
        return None


    u, T = u0[:-1], u0[-1]
    if u[0]== 0:
        print('the first value of initial condition cannot be zero')
    else:
        sol = solve_ivp(function, (0,T), u, args = args, rtol = 1e-6)
        final_states = sol.y[:,-1]
        phase = np.array([phase_condition(function,u,T,args)])
        return np.concatenate((u-final_states, phase))




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
    for parameter in np.arange(start,end,h):
        root = fsolve(fun,initial_guess,args=(parameter))
        initial_guess = root
        x.append(root[0])

    plt.xlabel('x')
    plt.ylabel('solution')
    plt.title('Natural parameter continuation of the cubic equation')
    plt.plot(np.arange(start,end,h), x)
    plt.legend(['x','y','period'], loc = "upper right")
    plt.show()
    return


def Hopf(t,z,b,s):
    u1, u2 = z[0], z[1]
    return [b*u1 - u2 + s*u1*(u1**2 + u2**2), u1 + b*u2 + s*u2*(u1**2 +u2**2)]


def mod_Hopf(t,z,b,s):
    u1, u2 = z[0], z[1]
    return [b*u1 - u2 + s*u1*(u1**2 + u2**2) - u1*(u1**2 + u2**2)**2, u1 + b*u2 + s*u2*(u1**2 +u2**2) - u2*(u1**2 + u2**2)**2]




def Numerical_Continuation(initial_guess ,start, end, h, s, fun):
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

    if callable(fun) is not True:
        error_message_fun = "The input 'fun' must be a callable function"
        print(error_message_fun)
        return None

    if isinstance(initial_guess,list) is not True:
        error_message_guess = "The initial guess must be list. The period should be the last entry in the list"
        print(error_message_guess)
        return None

    if isinstance(start, (int,float)) is not True:
        error_message_int = "The values for start, end and stepsize must be integers"
        print(error_message_int)
        return None

    if isinstance(end, (int,float)) is not True:
        error_message_int = "The values for start, end and stepsize must be integers"
        print(error_message_int)
        return None

    if isinstance(h, (int,float)) is not True:
        error_message_int = "The values for start, end and stepsize must be integers"
        print(error_message_int)
        return None

    if isinstance(s, (int,float)) is not True:
        error_message_int = "The values for start, end and stepsize must be integers"
        print(error_message_int)
        return None


    if fun == Hopf:
        title = 'the Hopf bifurcation normal form'
    elif fun == mod_Hopf:
        title = 'the modified Hopf bifurcation normal form'
    else:
        raise ValueError("Invalid function. Please specify 'Hopf' or 'mod_Hopf'")



    x = []
    y = []
    period = []

    for b in np.arange(start,end,h):
        root = fsolve(shooting,initial_guess,args = (fun, phase_condition_func, (b,s)))
        initial_guess = root
        x.append(root[0])
        y.append(root[1])
        period.append(root[2])

    plt.xlabel('\u03B2')
    plt.ylabel('solution')
    plt.title(f'Natural parameter continuation of \u03B2 for the {title} equations')
    plt.plot(np.arange(start,end,h), x)
    plt.plot(np.arange(start,end,h), y)
    plt.legend(['x','y'], loc = "upper right")
    plt.show()
    return




def finite_diff(kappa, L, T, initial_condition ,method = 'Forward-Euler', mx=20, mt=1000):


    if callable(initial_condition) is not True:
        error_message_initial_cond = "The initial condition must be a callable function"
        print(error_message_initial_cond)
        return None

    if isinstance(kappa,(int,float)) is not True:
        error_message_int = "The values of kappa, L and T should be integers or floats"
        print(error_message_int)
        return None


    if isinstance(L,(int,float)) is not True:
        error_message_int = "The values of kappa, L and T should be integers or floats"
        print(error_message_int)
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


    # Set up the numerical environment variables
    x = np.linspace(0, L, mx+1)     # mesh points in space
    t = np.linspace(0, T, mt+1)     # mesh points in time
    deltax = x[1] - x[0]            # gridspacing in x
    deltat = t[1] - t[0]            # gridspacing in t
    lmbda = kappa*deltat/(deltax**2)    # mesh fourier number


    # Set up the solution variables
    u_j = np.zeros(x.size)        # u at current time step
    u_jp1 = np.zeros(x.size)      # u at next time step

    # Set initial condition
    for i in range(0, mx+1):
        u_j[i] = initial_condition(x[i])

    if method == 'Forward-Euler':
        u_jp1 = np.zeros(mx+1)
        k = [lmbda*np.ones(mx-2),(1-2*lmbda)*np.ones(mx-1),lmbda*np.ones(mx-2)]
        offset = [-1,0,1]
        A = diags(k,offset).toarray()


        for j in range(0, mt):
            u_jp1[1:-1] = np.matmul(A , u_j[1:-1])
            u_j = u_jp1

    elif method == 'Backward-Euler':
        u_jp1 = np.zeros(mx+1)
        k = [-lmbda*np.ones(mx-2),(1+2*lmbda)*np.ones(mx-1),-lmbda*np.ones(mx-2)]
        offset = [-1,0,1]
        A_BE = diags(k,offset).toarray()


        for j in range(0, mt):
            u_jp1[1:-1] = np.linalg.solve(A_BE , u_j[1:-1])
            u_j = u_jp1

    elif method == 'Crank-Nicholson':
        u_jp1 = np.zeros(mx+1)
        k = [(-lmbda/2)*np.ones(mx-2),(1+lmbda)*np.ones(mx-1),(-lmbda/2)*np.ones(mx-2)]
        f = [(lmbda/2)*np.ones(mx-2),(1-lmbda)*np.ones(mx-1),(lmbda/2)*np.ones(mx-2)]
        offset = [-1,0,1]
        A_CN = diags(k,offset).toarray()
        B_CN = diags(f,offset).toarray()


        for j in range(0, mt):
            b = np.matmul(B_CN , u_j[1:-1])
            u_jp1[1:-1] = np.linalg.solve(A_CN,b)
            u_j = u_jp1

    else:
        raise ValueError("Invalid method, please input 'Forward-Euler', 'Backward-Euler' or 'Crank-Nicholson'.")

    return u_j





def PDE_solve_euler(kappa,L,T,initial_condition, args=(), boundary_condition=None,mx=20,mt=1000):

    """
    A function that user the euler method to solve the 1D heat equation.
    A right-hand-side function or boundary condition can be specified as well as an initial condition.

    Parameters
    ----------
    kappa:  int
            The diffusion constant

    L:  int
        length of the domain

    T:  int
        Total time to solve for

    initial_condition:  function
                        The initial temperature distribution. Can be inputted in the command line using 'lambda' or an
                        initial condition function can be inputted.

    args:  function
            This is where the user can specify the value on the boundaries for certain boundary conditions or a right-
            hand-side function. For a periodic boundary, the args can be left empty.

    boundary_condition: str
                        If you wish to specify a boundary condition or rhs function specify the name here.
                        neumann boundary condition: 'neumann'
                        dirichlet boundary condition: 'dirichlet'
                        periodic boundary condition: 'periodic'
                        right-hand-side function: 'rhs'
                        default boundary condition: None

    mx = int
         The number of gridpoints in space

    mt = int
         The number of gridpoints in time

    Returns
    -------
    returns an array of of the solutions to the specified PDE problem
    """

   # if callable(initial_condition) is not True:
        #error_message_initial_cond = "The initial condition must be a callable function"
        #print(error_message_initial_cond)
        #return None

    if isinstance(kappa,(int,float)) is not True:
        error_message_int = "The values of kappa, L and T should be integers or floats"
        print(error_message_int)
        return None


    if isinstance(L,(int,float)) is not True:
        error_message_int = "The values of kappa, L and T should be integers or floats"
        print(error_message_int)
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

    #if isinstance(args, function) is not True:
        #error_message_args = "The args must be passed as a callable function"
        #print(error_message_args)
        #return None

    # Set up the numerical environment variables
    x = np.linspace(0, L, mx+1)     # mesh points in space
    t = np.linspace(0, T, mt+1)     # mesh points in time
    deltax = x[1] - x[0]            # gridspacing in x
    deltat = t[1] - t[0]            # gridspacing in t
    lmbda = kappa*deltat/(deltax**2)    # mesh fourier number



    u_j = np.zeros(x.size)        # u at current time step
    u_jp1 = np.zeros(mx+1)        # u at the next time step
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

        if callable(initial_condition) is False:
            u_j = initial_condition


        else:
            for i in range(0, mx):
                u_j[i] = initial_condition(x[i])

        for j in range(0, mt):
            u_jp1 = np.matmul(A , u_j)
            u_j = u_jp1


    elif boundary_condition == 'rhs':

        u_jp1 = np.zeros(mx+1)
        k = [lmbda*np.ones(mx-2),(1-2*lmbda)*np.ones(mx-1),lmbda*np.ones(mx-2)]
        F_j = args

        offset = [-1,0,1]
        A = diags(k,offset).toarray()

        if callable(initial_condition) is False:
            u_j = initial_condition


        else:
            for i in range(0, mx+1):
                u_j[i] = initial_condition(x[i])


        for j in range(0, mt):

            u_jp1[1:-1] = np.matmul(A , u_j[1:-1]) + deltat*F_j(x[1:-1],t[j])
            u_j = u_jp1


    else:
        if callable(initial_condition) is False:
            u_j = initial_condition


        else:
            for i in range(0, mx+1):
                u_j[i] = initial_condition(x[i])



        for j in range(0, mt):
            if boundary_condition == 'dirichlet':
                p = args[0]
                q = args[1]
                s = np.zeros(mx-1)
                s[0] = p(t[j])
                s[-1] = q(t[j])

                p= args[0]
                q = args[1]
                u_jp1[1:-1] = np.matmul(A , u_j[1:-1]) + lmbda*s
                u_jp1[0] = p(t[j+1]); u_jp1[mx] = q(t[j+1])
                u_j = u_jp1


            elif boundary_condition== 'neumann':
                k = [lmbda*np.ones(mx),(1-2*lmbda)*np.ones(mx+1),lmbda*np.ones(mx)]
                offset = [-1,0,1]
                A = diags(k,offset).toarray()
                A[0,1] = 2*lmbda
                A[-1,-2] = 2*lmbda
                s = np.zeros(mx+1)

                P = args[0]
                Q = args[1]
                s[0] = -P(t[j])
                s[-1] = Q(t[j])

                u_jp1 = np.matmul(A , u_j) + 2*lmbda*deltax*s
                u_j = u_jp1

            elif boundary_condition == None:
                u_jp1[1:-1] = np.matmul(A , u_j[1:-1])
                u_j = u_jp1

        return u_j
    return u_j




def Numerical_Continuation_kappa(L, T, initial_condition, start, end, h, args, boundary_condition=None, mx=20, mt=1000):

    """
    A function that performs natural parameter continution of the diffusion constant for the 1D heat equation.
    It increments the parameter kappa (k) by a set amount and attempts to find the solution for the new parameter
    value using the last found solution as the initial temperature distribution.

    Parameters
    ----------

    L:  int
        length of the domain

    T:  int
        Total time to solve for

    initial_condition:  function
                        The initial temperature distribution. Can be inputted in the command line using 'lambda' or an
                        initial condition function can be inputted.

    start:  value(int or float)
        The initial value for the parameter

    end: value(int or float)
        The end value for the parameter

    h: value(int or float)
        The step-size we wish to increment the parameter by. Note this value should be negative if the start value is
        positive and the end value is negative

    args:  function
            This is where the user can specify the value on the boundaries for certain boundary conditions or a right-
            hand-side function. For a periodic boundary, the args can be left empty.

    boundary_condition: str
                        If you wish to specify a boundary condition or rhs function specify the name here.
                        neumann boundary condition: 'neumann'
                        dirichlet boundary condition: 'dirichlet'
                        periodic boundary condition: 'periodic'
                        right-hand-side function: 'rhs'
                        default boundary condition: None

    mx = int
         The number of gridpoints in space

    mt = int
         The number of gridpoints in time

    Returns
    -------
    Returns a graph showing how the solution of the PDE changes as the diffusion constant is incremented from start to
    end value.
    """
    if callable(initial_condition) is not True:
        error_message_initial_cond = "The initial condition must be a callable function"
        print(error_message_initial_cond)
        return None

    if isinstance(L,(int,float)) is not True:
        error_message_int = "The values of kappa, L and T should be integers or floats"
        print(error_message_int)
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

    if isinstance(start, (int,float)) is not True:
        error_message_int = "The values for start, end and stepsize must be integers"
        print(error_message_int)
        return None

    if isinstance(end, (int,float)) is not True:
        error_message_int = "The values for start, end and stepsize must be integers"
        print(error_message_int)
        return None

    if isinstance(h, (int,float)) is not True:
        error_message_int = "The values for start, end and stepsize must be integers"
        print(error_message_int)
        return None


    n = []

    for parameter in np.arange(start,end,h):
        root = PDE_solve_euler(parameter,L,T,initial_condition, args=args, boundary_condition=boundary_condition,mx=20,mt=1000)
        initial_condition = root
        n.append(root)

    i=0
    for i in range(len(n)):
        plt.plot(np.linspace(0, L, mx+1),n[i],'ro',label='num')
    plt.xlabel('x')
    plt.ylabel(f'u(x,{T})')
    plt.title(plt.title(f'solution for varying diffusion coefficent for heat equation with {boundary_condition} boundary'))
    plt.show()
    return


