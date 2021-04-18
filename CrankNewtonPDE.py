#Q. du/dt + u*du/dx = nu*d^2u/dx^2, nu = 1, u(x,0) = sin(pi*x), u(0,t) = 0, u(1,t) = 0, dx = 0.5,0.10,0.05,0.01, r = 1/2
# Crank Nicolson with Newton Linearisation Scheme


import math
import numpy as np
import matplotlib.pyplot as plt


# Thomas Algo for Block Tridiagonal System
def blockTDMA(a, b, c, d):
    
    n = len(b)
    
    """
    Solution of a linear system of algebraic equations with a
        Block-tri-diagonal matrix of coefficients using the Block-Thomas-algorithm.

    Arguments:
        a(array of 2*2 matrix): an array containing lower diagonal
        b(array of 2*2 matrix): an array containing main diagonal 
        c(array of 2*2 matrix): an array containing lower diagonal
        d(array of 2*2 matrix): right hand side of the system
    Returns:
        x(array of 2*1 matrix): solution array of the system
    """

    # elimination
    C_upd = []
    C_upd.append(np.matmul(np.linalg.inv(b[0]), c[0]))
    for i in range(1,n-1):
        C_upd.append(np.matmul(np.linalg.inv(b[i] - np.matmul(a[i],C_upd[i-1])), c[i]))
    
    D_upd = []
    D_upd.append(np.matmul(np.linalg.inv(b[0]), d[0]))
    for i in range(1,n):
        D_upd.append(np.matmul(np.linalg.inv(b[i] - np.matmul(a[i],C_upd[i-1])), d[i]-np.matmul(a[i], D_upd[i-1])))
    
    # back_substitution
    x = []
    x.append(np.array(D_upd[n-1]))
    for i in range(n-2,-1,-1):
        x.insert(0, np.array(D_upd[i] - np.matmul(C_upd[i], x[0])))

    #print(x)
    return x


# Thomas Algorithm for tridiagonal system of equations
def thomas (a, b, c, d):
    
    n = len(b)
    arr = np.zeros(n)
    
    """
    Solution of a linear system of algebraic equations with a
        tri-diagonal matrix of coefficients using the Thomas-algorithm.

    Arguments:
        a(array): an array containing lower diagonal (a[0] is not used)
        b(array): an array containing main diagonal 
        c(array): an array containing lower diagonal (c[-1] is not used)
        d(array): right hand side of the system
    Returns:
        x(array): solution array of the system
    """
    
    # elimination
    for k in range(1,n):
        val  = a[k]/b[k-1]
        b[k] = b[k] - c[k-1]*val
        d[k] = d[k] - d[k-1]*val
        
    # back_substitution
    val = d[n-1]/b[n-1]
    arr[n-1] = val
    for k in range(n-2,-1,-1):
        val = (d[k]-c[k]*val)/b[k]
        arr[k] = val
    
    return arr


def showResult(res,st,h):
    x = []
    print("y at -> ")
    for i in range(len(res)):
        x.append(st+i*h)
        print("x = ",'%.3f'%(st+i*h)," = ",'%.10f'%(res[i]))
    print()
    
    # Plotting the function
    plt.plot(x, res)
    plt.ylabel( 'Values of y(x) ---> ' )
    plt.xlabel( 'Values of x ---> ' )
    plt.show()


def Crank_Newton(Y,dx,dt,r,N):

    # the initial values for current time come from last step of previous time
    Ylast = Y.copy()
    
    # error will tell the max error at (k+1)th iteration
    error = 10000000000.0

    # epsilon is the admissible error
    epsilon = 0.0000001

    # iteration_num keeps count of total iterations
    iteration_num = 0
    # call newtonLinearisation untill error is within bound
    while(error > epsilon and iteration_num < 500):
        iteration_num += 1
        #print("Iteration: ",iteration_num)
        error = Newton(Ylast,Y,dx,dt,r,N)    
    
    return Ylast
    
    
def Newton(Ylast,Yn,dx,dt,r,N):
    
    # set the a,b,c and d array(s)
    a = np.zeros(N-1)
    for i in range(N-1):
        a[i] = -Ylast[i+1]/(4*dx) - 1/(2*dx*dx)
    
    b = np.zeros(N-1)
    for i in range(N-1):
        b[i] = 1/dt + (Ylast[i+2]-Ylast[i])/(4*dx) + 1/(2*dx*dx)
    
    c = np.zeros(N-1)
    for i in range(N-1):
        c[i] = Ylast[i+1]/(4*dx) - 1/(2*dx*dx)
    
    d = np.zeros(N-1)
    for i in range(N-1):
        d[i] = -1*((Ylast[i+1])/dt + (Ylast[i+1]*(Ylast[i+2]-Ylast[i]))/(4*dx) - 1*(Ylast[i+2]-2*Ylast[i+1]+Ylast[i])/(2*dx*dx) - 1*(Yn[i+2]-2*Yn[i+1]+Yn[i])/(2*dx*dx) + (Yn[i+1]*(Yn[i+2]-Yn[i]))/(4*dx) - (Yn[i+1])/dt)
        
    # set the boundary conditions
    a0 = -Ylast[0]/(4*dx) - 1/(2*dx*dx)
    cn_1 = Ylast[-1]/(4*dx) - 1/(2*dx*dx)
    delY0 = 0
    delYn = 0
        
    # update the end point values of d
    d[0] = d[0] - a0*delY0
    d[-1] = d[-1] - cn_1*delYn
    
    #print("a = ",a," b = ",b," c = ",c," d = ",d)
    
    delta_ = thomas(a, b, c, d)
    delta = delta_.tolist()
    # insert and append the BC values
    delta.insert(0,delY0)
    delta.append(delYn)
    
    error = 0.0
    for d in delta:
        if(error < abs(d)):
            error = abs(d)
    #print("error = ",error)
    
    for i in range(N+1):
        Ylast[i] = Ylast[i] + delta[i]
    
    #print("Ylast = ",Ylast)
    
    return error




# dx = grid size in x axis
dx = 0.05

# r
r = 1/2

# nu = coefficient in the equation
nu = 1

# dt = grid size in t axis
# r = nu*dt/(dx*dx)
dt = r*dx*dx/nu

# st -> start point, en -> end point
st = 0
en = 1

# N = number of x-axis points
N = (int)((en - st)/dx)


# Y -> (N+1) array
Y = []

# BC for x0
Y.append(0)

for i in range(1,N):
    # set the formula for filling values at inner grid points
    y = math.sin(math.pi*(i*dx))
    Y.append(y)
    
# BC for xn
Y.append(0)

print("Time = 0")
showResult(Y,st,dx)


iteration_num = 0
# limit sets the number of iterations over time
limit = 10*dt
while(iteration_num < limit):
    iteration_num += dt
    print("\nTime = ",iteration_num)
    Y = Crank_Newton(Y,dx,dt,r,N)
    showResult(Y,st,dx)

