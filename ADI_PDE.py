#Q. du/dt = nu*(d^2u/dx^2 + d^2u/dy^2), nu = 1, u(x,y,0) = cos(pi*x/2)*cos(pi*y/2), u(dR) = 0, R = x = +_1, y = +_1, dx = dy = 0.5,0.10,0.05,0.01, r = 1/6
# Alternating Direction Implicit Scheme (ADI)


import math
import numpy as np
from mpl_toolkits import mplot3d
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


def f(x,y):
    return np.array(XY)

def showResult(XY,stx,sty,dx,dy,N,M):
    print("       y\\x",end="  ")
    for j in range(M+1):
        print("% 10.4f" % (sty+j*dy),end="   ")
    print()
    i=0
    for x in XY:
        print("% 10.4f" % (stx+i*dx),end="")
        i+=1
        for a in x:
            print("% 12.6f" % a,end=" ")
        print()
    print()
    
    # Plotting the function
    x = []
    y = []
    for i in range(N+1):
        x.append(stx+i*dx)
    for j in range(M+1):
        y.append(sty+j*dy)
    X, Y = np.meshgrid(x,y)
    Z = f(X, Y)
    
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    '''
    # 1st plot type
    ax.contour3D(X, Y, Z, 50, cmap='binary')
    ax.set_xlabel( ' <--- Values of x ' )
    ax.set_ylabel( ' Values of y ---> ' )
    ax.set_zlabel( ' <--- Values of f(x,y) ' )
    ax.view_init(60,35)
    fig
    '''
    # 2nd plot type
    ax.plot_wireframe(X, Y, Z, color='green')
    ax.set_xlabel( ' Values of x ---> ' )
    ax.set_ylabel( ' Values of y ---> ' )
    ax.set_zlabel( ' Values of f(x,y) ---> ' )
    plt.show()
    ax = plt.axes(projection='3d')
    ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap='winter', edgecolor='none')



def ADIScheme(XY,dx,dy,dt,r,N,M):
    
    # a new XY (2D)array for t = n+1/2
    XY_1 = []
    
    # step 1:
    
    for j in range(1,M):
        a = np.zeros(N-1)
        for i in range(N-1):
            a[i] = 1/(dx*dx)
    
        b = np.zeros(N-1)
        for i in range(N-1):
            b[i] = -2/(dx*dx) - 1/(dt/2)
    
        c = np.zeros(N-1)
        for i in range(N-1):
            c[i] = 1/(dx*dx)
    
        d = np.zeros(N-1)
        for i in range(1,N):
            d[i-1] = -1*(XY[i][j-1] - 2*XY[i][j] + XY[i][j+1])/(dy*dy) - XY[i][j]/(dt/2)
        
        # set the boundary conditions
        a0 = 1/(dx*dx)
        cn_1 = 1/(dx*dx)
        
        # update the end point values of d
        d[0] = d[0] - a0*XY[0][j]
        d[-1] = d[-1] - cn_1*XY[N][j]
        
        #print("a = ",a," b = ",b," c = ",c," d = ",d)
    
        X = thomas(a, b, c, d)
        X_ = X.tolist()
        # insert and append the BC values
        X_.insert(0,0)
        X_.append(0)
    
        # set new values in XY_1
        XY_1.append(X_)
    
    y = [0]*(M+1)
    XY_1.insert(0,y)
    XY_1.append(y)
    
    
    
    
    # a new XY (2D)array for t = n+1
    XY_2 = []
    
    # step 2:

    for i in range(1,N):
        a = np.zeros(M-1)
        for j in range(M-1):
            a[j] = 1/(dy*dy)
    
        b = np.zeros(M-1)
        for j in range(M-1):
            b[j] = -2/(dy*dy) - 1/(dt/2)
    
        c = np.zeros(M-1)
        for j in range(M-1):
            c[j] = 1/(dy*dy)
    
        d = np.zeros(M-1)
        for j in range(1,M):
            d[j-1] = -1*(XY_1[i-1][j] - 2*XY_1[i][j] + XY_1[i+1][j])/(dx*dx) - XY_1[i][j]/(dt/2)
        
        # set the boundary conditions
        a0 = 1/(dy*dy)
        cn_1 = 1/(dy*dy)
        
        # update the end point values of d
        d[0] = d[0] - a0*XY_1[i][0]
        d[-1] = d[-1] - cn_1*XY_1[i][M]
        
        #print("a = ",a," b = ",b," c = ",c," d = ",d)
    
        X = thomas(a, b, c, d)
        X_ = X.tolist()
        # insert and append the BC values
        X_.insert(0,0)
        X_.append(0)
    
        # set new values in XY_1
        XY_2.append(X_)
    
    y = [0]*(N+1)
    XY_2.insert(0,y)
    XY_2.append(y)
    
    
    return XY_2



# dx = grid size in x axis
dx = 0.1

# dy = grid size in x axis
dy = 0.1

# r
r = 1/6

# nu -> coefficient
nu = 1

# dt = grid size in t axis
# r = nu*dt/(dx*dy)
dt = r*dx*dy/nu

# st -> start point for x and y axis, en -> end point for x and y axis
stx = -1
enx = 1
sty = -1
eny = 1

# N = number of x-axis points
N = (int)((enx - stx)/dx)

# M = number of y-axis points
M = (int)((eny - sty)/dy)


# set initial values

# XY -> (N+1)*(M+1) 2D array
XY = []
# BC for x = -1
y = [0]*(M+1)
XY.append(y)

for i in range(1,N):
    y = []
    # BC for y = -1
    y.append(0)
    # set the formula for filling values at inner grid points
    for j in range(1,M):
        y.append(math.cos(math.pi*(stx + i*dx)/2)*math.cos(math.pi*(sty + j*dy)/2))
    # BC for y = 1
    y.append(0)
    XY.append(y)

# BC for x = 1
y = [0]*(M+1)
XY.append(y)

print("Time = 0")
showResult(XY,stx,sty,dx,dy,N,M)


iteration_num = 0
# limit sets the number of iterations over time
limit = 10*dt
while(iteration_num < limit):
    iteration_num += dt
    print("Time = ",iteration_num)
    XY = ADIScheme(XY,dx,dy,dt,r,N,M)
    showResult(XY,stx,sty,dx,dy,N,M)

