#Q. d^2u/dt^2 = c^2 * d^2u/dx^2, c^2 = 1, u(x,0) = e^(-200(x-0.5)^2) if 0<=x<=1 and 0 otherwise, u(0,t) = 0, u(1,t) = 0, du/dt(x,0) = 0, dx = 0.25,0.10,0.05, r = 0.5
# Leap Frog Scheme / CTCS (Central Time Central Space)


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

def showResult(exact,res,st,h):
    x = []
    print("y at -> ")
    for i in range(len(res)):
        x.append(st+i*h)
        #print("x = ",'%.3f'%(st+i*h)," = ",'%.10f'%(res[i]))
    print()
    
    # Plotting the function
    plt.plot(x, res)
    # plot the exact solution also
    plt.plot(x, exact)
    plt.ylabel( 'Values of y(x) ---> ' )
    plt.xlabel( 'Values of x ---> ' )
    plt.legend(["LeapFrog soln","Exact soln"])
    
    plt.show()



def leapFrog(Y,Yprev,st,h,dt,r,c,t,N):
    
    Y_ = []
    
    # BC at x0
    Y_.append(0)
    for i in range(1,N):
        a = r*r*(Y[i+1] + Y[i-1]) + 2*(1-r*r)*Y[i] - Yprev[i]
        Y_.append(a)
    # BC at xn
    Y_.append(0)
    
    # update the values of Yprev and Y
    for i in range(N+1):
        Yprev[i] = Y[i]
    for i in range(N+1):
        Y[i] = Y_[i]
    
    # the exact solutin is given by (f(x-ct)+f(x+ct))/2
    # in the example,exact soln = ( exp(-200*((x-c*t-0.5)**2)) + exp(-200*((x+c*t-0.5)**2)) )/2
    exact = []
    for i in range(N+1):
        a = ( math.exp(-200*((st+i*h-c*t-0.5)**2)) + math.exp(-200*((st+i*h+c*t-0.5)**2)) )/2
        exact.append(a)
    
    showResult(exact,Y,st,h)
        


# h = grid size in x axis
h = 1/100

# r = c*dt/dx
r = 0.5

# c is the coeff of du/dx in PDE
c = 1

# dt = grid size in t axis
# dt = r*dx/c*c
dt = r*h/c*c

# st -> start point, en -> end point
st = 0
en = 1

# N = number of x-axis points
N = (int)((en - st)/h)


# set the initial values
# u(x,0) = exp(-200*((x-0.5)**2))
Y = []
# here BCs are handeled separately
# BC at x0
Y.append(0)
for i in range(1,N):
    Y.append(math.exp(-200*((st+i*h-0.5)**2)))
# BC at xn
Y.append(0)
print("Time = 0")
showResult(Y,Y,st,h)


# we will have to handle the first time iteration due to the use of n-1 th time stamp
# BC -> du/dx(x,0) = 0
# => Y(-1)j = Y(1)j
Y1 = []
# here BCs are handeled separately
# BC at x0
Y1.append(0)
for i in range(1,N):
    a = r*r*(Y[i+1] + Y[i-1])/2 + (1-r*r)*Y[i]
    Y1.append(a)
# BC at xn
Y1.append(0)
print("Time = ",dt)

# the exact solutin is given by (f(x-ct)+f(x+ct))/2
# in the example,exact soln = ( exp(-200*((x-c*t-0.5)**2)) + exp(-200*((x+c*t-0.5)**2)) )/2
exact = []
for i in range(N+1):
    a = (math.exp(-200*((st+i*h-c*dt-0.5)**2)) + math.exp(-200*((st+i*h+c*dt-0.5)**2)))/2
    exact.append(a)
showResult(exact,Y1,st,h)


iteration_num = 2
# limit sets the number of iterations over time
limit = 50*dt
while(iteration_num*dt < limit):
    iteration_num += 1
    print("Time = ",iteration_num*dt)
    leapFrog(Y1,Y,st,h,dt,r,c,iteration_num*dt,N)