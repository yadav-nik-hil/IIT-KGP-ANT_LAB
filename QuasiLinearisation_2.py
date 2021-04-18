#Q. y'' - (y')^2 - y^2 + y + 1 = 0, y(0) = 0.5, y(pi) = -0.5, h = pi/2,pi/10,pi/20
# Quasi Linearisation Technique

# Error -> This is showing erratic behaviour for different value of h
# here we took the intial value acc to y = -1/2 + pi*x
# Solution -> if we change the initial value assumption to y = cos(pi*x)/2
# then it converges for all values of h


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
    for i in range(len(res)):
        x.append(st+i*h)
        print("x = ",'%6.3f'%(st+i*h)," -> ",'%12.8f'%(res[i]))
    print()

    # Plotting the function
    plt.plot(x, res)
    plt.ylabel( 'Values of y(x) ---> ' )
    plt.xlabel( 'Values of x ---> ' )
    plt.show()



def quasiLinearisation(Y, h, N):
    
    # set the a,b,c and d array(s)
    a = np.zeros(N-1)
    for i in range(1,N):
        a[i-1] = (1/(h*h))+((Y[i+1]-Y[i-1])/(2*h*h))
    
    b = np.zeros(N-1)
    for i in range(1,N):
        b[i-1] = (-2/(h*h))-(2*Y[i])+1
    
    c = np.zeros(N-1)
    for i in range(1,N):
        c[i-1] = (1/(h*h))-((Y[i+1]-Y[i-1])/(2*h*h))
    
    # set di values according to the question
    d = np.zeros(N-1)
    for i in range(1,N):
        d[i-1] = -(Y[i]*Y[i])-(((Y[i+1]-Y[i-1])/(2*h))**2)-1
        
    # set boundary values
    Y0 = 0.5
    Yn = -0.5
    a0 = a[0]
    cn_1 = c[-1]
    
    # update the first and last value in d array
    d[0] = d[0] - Y0*a0
    d[-1] = d[-1] - Yn*cn_1
    
    updValues = thomas(a, b, c, d)
    
    error = 0.0
    for i in range(1,N):
        if(error < abs(updValues[i-1] - Y[i])):
            error = abs(updValues[i-1] - Y[i])
    print("error = ",error)


    for i in range(1,N):
        Y[i] = updValues[i-1]

    return error
    


# error will tell the max error at (k+1)th iteration
error = 10000000000.0

# epsilon is the admissible error
epsilon = 0.00000001

# iteration_num keeps count of total iterations
iteration_num = 0

# h = grid size
h = math.pi/20

# st -> start point, en -> end point
st = 0
en = math.pi

# N = number of x-axis points
N = (int)((en - st)/h)


Y = []
# initial values for Y is set as per intial guess:
'''
# this initial assumption leads to erratic behaviour
# y = -x/pi + 0.5
for i in range(N+1):
    Y.append(0.5 - (st + i*h)/math.pi)
'''
# this is a more stable assumption and leads to stability
# y = cos(x)/2
for i in range(N+1):
    Y.append((math.cos(st + i*h))/2)

print("Initial Guess:")
print("y at -> ")
showResult(Y,st,h)


# call quasiLinearisation until error is within bound
while(error > epsilon and iteration_num < 10):
    iteration_num += 1
    print("Iteration: ",iteration_num)
    error = quasiLinearisation(Y, h, N)
    print("y at -> ")
    showResult(Y,st,h)

