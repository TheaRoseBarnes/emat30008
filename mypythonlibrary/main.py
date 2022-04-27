# Import the library and functions
import mypythonlib
from mypythonlib import myfunctions


# import functions
import numpy as np
import math
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from scipy.optimize import fsolve
from scipy.integrate import solve_ivp
from scipy.sparse import diags
from math import pi

# Define the ODE we wish to solve
def f(x,t):
    return x



# Produce a plot with double logarithmic scale showing how the error depends on the size of the timestep
# for the euler method, 4th order Runge-Kutta method, Heun's method and Midpoint method
error_euler = []
error_runga = []
error_heuns = []
error_midpoint = []

for h in np.arange(0.0001,1,0.0001):
    [a,b]= myfunctions.solve_ode(f,h,0,1,[0.1,0.9,1],myfunctions.euler_solve_step)
    [i,j]= myfunctions.solve_ode(f,h,0,1,[0.1,0.9,1],myfunctions.rk4_solve_step)
    [n,k]= myfunctions.solve_ode(f,h,0,1,[0.1,0.9,1],myfunctions.heuns_solve_step)
    [y,q]= myfunctions.solve_ode(f,h,0,1,[0.1,0.9,1],myfunctions.midpoint_solve_step)
    error_euler.append(b)
    error_runga.append(j)
    error_heuns.append(k)
    error_midpoint.append(q)

plt.loglog(np.arange(0.0001,1,0.0001), error_runga)
plt.loglog(np.arange(0.0001,1,0.0001), error_euler)
plt.loglog(np.arange(0.0001,1,0.0001), error_heuns)
plt.loglog(np.arange(0.0001,1,0.0001), error_midpoint)
#plt.loglog(np.arange(0.0001,1,0.0001), error_rk3)
#plt.axhline(y=0.0006342732424382547, color='g')

plt.legend(['Runge-Kutta error','euler method error','heuns method error', 'midpoint method error'])
plt.title('How does error depend on the size of the timestep for different integration methods')
plt.xlabel('\u0394t')
plt.ylabel('error')
plt.show()

# Define a system of ODEs
def fun(X,t):
    x = X[0]
    y = X[1]
    return np.array([y,-x])

# Define the exact solution of the ODE we wish to solve
def x_exact(t):
    return np.cos(t) + np.sin(t)

def xdot_exact(t):
    return -np.sin(t) + np.cos(t)


#Plotting the results of system of ODEs
t = np.arange(0,10,0.1)
results_euler = myfunctions.solve_ode_system(fun,0.1,0,[1,1],t,myfunctions.euler_solve_step)
results_rk4 = myfunctions.solve_ode_system(fun,0.1,0,[1,1],t,myfunctions.rk4_solve_step)
results_heuns = myfunctions.solve_ode_system(fun,0.1,0,[1,1],t,myfunctions.heuns_solve_step)
plt.plot(t,[state[0] for state in results_rk4])
plt.plot(t,[state[0] for state in results_euler])
plt.plot(t,[state[0] for state in results_heuns])
plt.plot(t,x_exact(t),'k--')
plt.legend(['rk4 method solution','euler method solution','heuns solution','True solution'])
plt.xlabel('t')
plt.ylabel('x')
plt.title('Graph displaying the solution of the system of ODEs')
plt.show()



#Plotting the results of system of ODEs
t = np.arange(0,10,0.1)
results_euler = myfunctions.solve_ode_system(fun,0.1,0,[1,1],t,myfunctions.euler_solve_step)
results_rk4 = myfunctions.solve_ode_system(fun,0.1,0,[1,1],t,myfunctions.rk4_solve_step)
results_heuns = myfunctions.solve_ode_system(fun,0.1,0,[1,1],t,myfunctions.heuns_solve_step)

fig=plt.figure(figsize=(8,6))
plt.plot([state[0] for state in results_euler],[state[1] for state in results_euler])
plt.plot([state[0] for state in results_rk4],[state[1] for state in results_rk4],'r')
plt.plot([state[0] for state in results_heuns],[state[1] for state in results_heuns])
plt.plot(x_exact(t),xdot_exact(t),'k--')

plt.legend(['euler method error','Runge-Kutta error','Heuns method','True value'])
plt.xlabel('x')
plt.ylabel('x\u0307')
plt.title('Graph displaying the solution of the system of ODEs')
plt.show()


# shooting


# setting pararmeters
a = 1
b = 0.2
d= 0.1


# defining a system of 2 ODEs (predator-prey equations)
def PredPrey(t,Z,a,b,d):
    x = Z[0]
    y = Z[1]
    return [x*(1-x) - (a*x*y)/(d+x), (b*y)*(1 - y/x)]

sol = solve_ivp(PredPrey, (0,100),(1,1), args=(a,b,d), rtol=1e-6)

# plotting a graph of the solutions

plt.plot(sol.t,sol.y[0])
plt.plot(sol.t,sol.y[1])
plt.legend(['Predator','Prey'])
plt.xlabel('t')
plt.ylabel('Number of Prey and Predators')
plt.title('Predator-Prey equations using Runge-Kutta method')
plt.show()

#
plt.plot(sol.y[0], sol.y[1])
plt.show()

root = fsolve(myfunctions.shooting,[1.5,1.5,20],args = (PredPrey, myfunctions.phase_condition_func, (a,b,d)))

#plot the solution for one periodic orbit
sol = solve_ivp(PredPrey, (0,root[2]), root[:2], args = (a,b,d), rtol = 1e-10)
plt.plot(sol.t, sol.y[0])
plt.plot(sol.t, sol.y[1])
plt.legend(['Predator','Prey'])
plt.xlabel('t')
plt.ylabel('Number of Prey and Predators')
plt.title('Predator-Prey equations using Runge-Kutta method for one periodic orbit')
plt.show()


# define the varaibles
m = 8
n = 30
c = 8/3

# define a system of 3 first order ODE's
def system2(t,Z,m,n,c):
    x, y, z = Z[0], Z[1], Z[2]
    xdot = -m*x + m*y
    ydot = n*x + y - x*z
    zdot = x*y - c*z
    return [xdot,ydot,zdot]

#root = fsolve(myfunctions.shooting,[-1,0,1,10],args = (system2,myfunctions.phase_condition_func,(m,n,c)))

#plot the solution for one periodic orbit

#sol = solve_ivp(system2, (0,root[-1]), root[:-1], args = (m,n,c), rtol = 1e-6)
#plt.plot(sol.t, sol.y[0])
#plt.plot(sol.t, sol.y[1])
#plt.plot(sol.t, sol.y[2])
#plt.legend(['x(t)','y(t)','z(t)'])
#plt.xlabel('t')
#plt.ylabel('')
#plt.title('Solution of system of three first-order ODEs')
#plt.show()


# code testing

# The Hopf bifurcation equations
def Hopf(t,z,b,s):
    u1, u2 = z[0], z[1]
    return np.array([b*u1 - u2 + s*u1*(u1**2 + u2**2), u1 + b*u2 + s*u2*(u1**2 +u2**2)])

# assign values
b = 1
s = -1

from tests import test_myfunctions
test_myfunctions.testing_2ODE(myfunctions.shooting,[1,0,6.2],(Hopf,myfunctions.phase_condition_func,(1,-1)))

# add another dimension for the Hopf bifurcation equations
#so that we have a system of 3 ODE's
def k(t,z,b,s):
    u1, u2, u3 = z[0], z[1], z[2]
    return [b*u1 - u2 + s*u1*(u1**2 + u2**2), u1 + b*u2 + s*u2*(u1**2 +u2**2), -u3]

test_myfunctions.testing_3ODE(myfunctions.shooting,[3,2,3,6.2],(k,myfunctions.phase_condition_func,(1,-1)))


# continutaion

# The algebraic cubic equation
def cubic(x,c):
    return x**3 - x + c

# The modified Hopf bifurcation equations
def mod_Hopf(t,z,b,s):
    u1, u2 = z[0], z[1]
    return [b*u1 - u2 + s*u1*(u1**2 + u2**2) - u1*(u1**2 + u2**2)**2, u1 + b*u2 + s*u2*(u1**2 +u2**2) - u2*(u1**2 + u2**2)**2]

c=1
b=1
s=-1


myfunctions.Numerical_Continuation_x(1, lambda x,c: x**3 - x + c , -2, 2, 0.1, c)

myfunctions.Numerical_Continuation([1,0,6.2],0.1,2,0.1,b,-1,Hopf)

myfunctions.Numerical_Continuation([1,0,6.2],2,-1,-0.1,b,-1,mod_Hopf)



# PDE BIT

def u_I(x):
    # initial temperature distribution
    y = np.sin(pi*x/L)
    return y

def u_exact(x,t):
    # the exact solution
    y = np.exp(-kappa*(pi**2/L**2)*t)*np.sin(pi*x/L)
    return y

def u_exact1(x,t):
    # the exact solution
    y = 100 - 50*x - (100/pi)*(np.exp(-3*t*pi**2)*np.sin(pi*x))
    return y

def u_I1(x):
    # initial temperature distribution
    y = 50
    return y

def u_exact2(x,t):
    # the exact solution
    y = 47.0449*np.exp(-0.0210*t)*np.sin(0.7249*x) + 45.1413*np.exp(-0.1113*t)*np.sin(1.6679*x)
    return y

def u_I2(x):
    # initial temperature distribution
    y = 100*(1 - (x/3))
    return y

def u_I3(x):
    # initial temperature distribution
    y = 5*np.sin(x)
    return y

def u_exact3(x,t):
    # the exact solution
    y = 0.5*(5*(1-np.cos(2))) - ((20*np.cos(2) + 20)/(-4 + pi**2))*np.exp(-(pi**2)*t/2**2)*np.cos(pi*x/2)
    return y


# differentiated version of u
def u_diff(x,t):
    dudx = (pi*np.exp(-kappa*(pi**2/L**2)*t)*np.cos(pi*x/L))/L
    return dudx


def u_I4(x):
    # initial temperature distribution
    y = 2*x**2
    return y

def u_exact4(x,t):
    # the exact solution
    y = 4*x - ((64/pi**3)*np.exp(t*(-9/4)*pi**2)*np.sin(pi*x/2))
    return y

def u_I5(x):
    # initial temperature distribution
    y = np.sin(pi*x/2)
    return y

def u_exact5(x,t):
    # the exact solution
    y = np.exp(-2*((pi**2)/4))*np.sin(pi*x/2) + np.sin(pi*x/2)*(2/(pi**2) - 2*np.exp((-t*pi**2)/2)/pi**2)
    return y

def u_I6(x):
    # initial temperature distribution
    y = 4*np.sin(3*pi*x) + 9*np.sin(7*pi*x)
    return y

def u_exact6(x,t):
    # the exact solution
    y = 4*np.exp(-t*2*(3*pi)**2)*np.sin(3*pi*x) + (1/(25*2*pi**2))*np.exp(-t*2*(5*pi)**2)*np.sin(5*pi*x) + 9*np.exp(-t*(7*pi)**2)*np.sin(7*pi*x) + (1/(2*25*pi**2))*np.sin(5*pi*x)
    return y

def u_I7(x):
    # initial temperature distribution
    y = np.cos(2*pi*x)
    return y

def u_exact7(x,t):
    # the exact solution
    y = np.exp(-4*t*pi**2)*np.cos(2*pi*x)
    return y

plt.plot(np.linspace(0, 2, 20+1),myfunctions.PDE_solve_euler(9,2,0.5,u_I4,args=(lambda t:0, lambda t:8), boundary_condition='dirichlet',mx=20,mt=1000),'ro',label='num')
xx = np.linspace(0,2,250)
T=2
plt.plot(xx,u_exact4(xx,T),'b-',label='exact')
plt.show()


# plotting for periodic
x = np.linspace(0, 1, 20)
plt.plot(x,myfunctions.PDE_solve_euler(1,1,0.5,u_I7, args=(lambda x: x*2, ), boundary_condition = 'periodic'),'ro',label='num')
xx = np.linspace(0,1,250)
plt.plot(xx,u_exact7(xx,0.5),'b-',label='exact')
plt.xlabel('x')
plt.ylabel('u(x,0.5)')
plt.legend(loc='upper right')
plt.show()



# Plot the final result and exact solution for forward euler
#pl.plot(x,PDE_solve_euler(1000,10,[0,0],u_I,args=(lambda t:u_diff(0,t),lambda t:u_diff(L,t)),dirichlet=None,neumann=None,periodic=None, rhs=rhs_function),'ro',label='num')
#x = np.linspace(0, 2, 20+1)
#plt.plot(x,myfunctions.PDE_solve_euler(2,2,0.5,u_I5, args=(lambda x,t: np.cos(x) + t*x**2), boundary_condition = 'rhs'),'ro',label='num')
#xx = np.linspace(0,2,250)
#plt.plot(xx,u_exact5(xx,0.5),'b-',label='exact')
#plt.xlabel('x')
#plt.ylabel('u(x,0.5)')
#plt.legend(loc='upper right')
#plt.show()



# Plot the final result and exact solution for forward euler
#pl.plot(x,PDE_solve_euler(1000,10,[0,0],u_I,args=(lambda t:u_diff(0,t),lambda t:u_diff(L,t)),dirichlet=None,neumann=None,periodic=None, rhs=rhs_function),'ro',label='num')
x = np.linspace(0, 2, 20+1)
plt.plot(x,myfunctions.PDE_solve_euler(1,2,0.5,u_I3, args=(lambda t:0, lambda t:0), boundary_condition = 'neumann'),'ro',label='num')
xx = np.linspace(0,2,250)
plt.plot(xx,u_exact3(xx,0.5),'b-',label='exact')
plt.xlabel('x')
plt.ylabel('u(x,0.5)')
plt.legend(loc='upper right')
plt.show()


# Plot the final result and exact solution for forward euler
#pl.plot(x,PDE_solve_euler(1000,10,[0,0],u_I,args=(lambda t:u_diff(0,t),lambda t:u_diff(L,t)),dirichlet=None,neumann=None,periodic=None, rhs=rhs_function),'ro',label='num')
x = np.linspace(0, 1, 20+1)
plt.plot(x,myfunctions.PDE_solve_euler(2,1,0.5,u_I6, args=(lambda x,t: np.sin(5*pi*x)), boundary_condition = 'rhs'),'ro',label='num')
xx = np.linspace(0,1,250)
plt.plot(xx,u_exact6(xx,0.5),'b-',label='exact')
plt.xlabel('x')
plt.ylabel('u(x,0.5)')
plt.legend(loc='upper right')
plt.show()
