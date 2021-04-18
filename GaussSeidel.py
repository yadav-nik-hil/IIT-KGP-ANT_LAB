#Q. d^2u/dx^2 + d^2u/dy^2 - 2du/dx = -2 , R = 0<x,y<1 , u = 0 on dR, dx = dy = 1/3
# Gauss Seidel Method (Sccessive over relaxation)


import numpy as np
from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt

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





def GaussSeidel(XY,dx,dy,N,M):
    
    # w -> relaxation parameter
    # if w = 1 -> gauss seidel
    w = 1.35
    
    h = dx #(h = dx = dy)
    
    XY_ = []
    # BC at x=x0
    x = [0]*(M+1)
    XY_.append(x)
    # set internal values
    for i in range(1,N):
        y = []
        # BC at y=y0
        y.append(0)
        for j in range(1,M):
            a = (2*h*h + (1-h)*XY[i+1][j] + (1+h)*XY_[i-1][j] + y[j-1] + XY[i][j+1])/4
            # successive over relaxation
            b = XY[i][j] + w*(a - XY[i][j])
            y.append(b)
        # BC at y=yn
        y.append(0)
        
        XY_.append(y)
    # BC at x=xn
    x = [0]*(M+1)
    XY_.append(x)
    
    # find maximum error
    error = 0.0000
    for i in range(N+1):
        for j in range(M+1):
            if(error < abs(XY_[i][j] - XY[i][j])):
                error = abs(XY_[i][j] - XY[i][j])
    
    # update the values
    for i in range(N+1):
        for j in range(M+1):
            XY[i][j] = XY_[i][j]
            
    return error




# dx = grid size in x axis
dx = 1/6

# dy = grid size in x axis
dy = 1/6

# st -> start point, en -> end point for x and y axis
stx = 0
enx = 1
sty = 0
eny = 1

# N = number of x-axis points
N = (int)((enx - stx)/dx)

# M = number of y-axis points
M = (int)((eny - sty)/dy)


# set the initial matrix
XY = []
#set the initial values according to initial guess
# guess -> u = x+y

# BC at x=x0
x = [0]*(M+1)
XY.append(x)
# set internal values acc to the guess
for i in range(1,N):
    y = []
    # BC at y=y0
    y.append(0)
    for j in range(1,M):
        y.append(stx+i*dx + sty+j*dy)
    # BC at y=yn
    y.append(0)

    XY.append(y)
# BC at x=xn
x = [0]*(M+1)
XY.append(x)


print("Initial Guess:")
showResult(XY,stx,sty,dx,dy,N,M)

# set the error limit
epsilon = 0.00000001

# set err variable to keep track of convergence
err = 10000000000

iteration_num = 0
limit = 100
# limit sets the number of iterations

while(iteration_num < limit and err > epsilon):
    iteration_num+=1
    print("Iteration = ",iteration_num)
    err = GaussSeidel(XY,dx,dy,N,M)
    showResult(XY,stx,sty,dx,dy,N,M)

if(iteration_num==limit):
    print("Doesn't converge")
