#Q. -(d^2u/dx^2 + d^2u/dy^2) + 0.1u = 1 , R = 0<x,y<1, u = 0 on x=0,y=0, du/dn = 0 on x=1,y=1, dx = dy = 1/2 
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
    w = 1.5
    
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
            a = (1 + XY[i+1][j]/(h*h) + XY_[i-1][j]/(h*h) + y[j-1]/(h*h) + XY[i][j+1]/(h*h) )/(4/(h*h)+0.1)
            #successive over relaxation
            b = XY[i][j] + w*(a - XY[i][j])
            y.append(b)
        # BC at y=yn
        if(len(y)>1):
            y.append(y[-2])
        else:
            y.append(y[-1])
        
        XY_.append(y)
    # BC at x=xn
    x = []
    for j in range(M+1):
        if(N-2>=0):
            x.append(XY_[N-2][j])
        else:
            x.append(XY_[N-1][j])
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
dx = 1/4

# dy = grid size in x axis
dy = 1/4

# st -> start point, en -> end point for x and y axis
stx = 0
enx = 1
sty = 0
eny = 1

# N = number of x-axis points
N = (int)((enx - stx)/dx)

# M = number of y-axis points
M = (int)((eny - sty)/dy)

# BC is du/dn = 0 => u(i-1,j)=u(i+1,j) and u(i,j-1)=u(i,j+1) on x=1,y=1

# set the initial matrix
XY = []
#set the initial values according to initial guess
# guess -> u = 10

# BC at x=x0
x = [0]*(M+1)
XY.append(x)
# set internal values acc to the guess
for i in range(1,N):
    y = []
    # BC at y=y0
    y.append(0)
    for j in range(1,M):
        y.append(10)
    # BC at y=yn
    if(len(y)>1):
        y.append(y[-2])
    else:
        y.append(y[-1])

    XY.append(y)
# BC at x=xn
x = []
for j in range(M+1):
    if(N-2>=0):
        x.append(XY[N-2][j])
    else:
        x.append(XY[N-1][j])
XY.append(x)


print("Initial Guess:")
showResult(XY,stx,sty,dx,dy,N,M)

# set the error limit
epsilon = 0.000001

# set err variable to keep track of convergence
err = 10000000000

iteration_num = 0
limit = 200
# limit sets the number of iterations

while(iteration_num < limit and err > epsilon):
    iteration_num+=1
    print("Iteration = ",iteration_num)
    err = GaussSeidel(XY,dx,dy,N,M)
    showResult(XY,stx,sty,dx,dy,N,M)

if(iteration_num==limit):
    print("Doesn't converge")
