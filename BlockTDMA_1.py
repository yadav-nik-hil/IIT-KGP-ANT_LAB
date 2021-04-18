'''
Instructions:
set the values of h, start and end points, 
set all ai,bi and ci acc to relations.
set X0 and Xn, a0 and cn-1
'''

#Q. y^(iv) + 81y = 81x^2, y(0) = 0, y(1) = 0, y''(0) = 0, y''(1) = 0, h = 0.25,0.1,0.05


import numpy as np
import matplotlib.pyplot as plt


def showResult(res,st,h):
    x = []
    y = []
    print("y at -> ")
    for i in range(len(res)):
        x.append(st+i*h)
        y.append(res[i][0])
        print("x = ",'%.3f'%(st+i*h)," = ",'%.10f'%(res[i][0]))
    
    p = []
    print("p is y'' here:")
    print("p at -> ")
    for i in range(len(res)):
        p.append(res[i][1])
        print("x = ",'%.3f'%(st+i*h)," = ",'%.10f'%(res[i][1]))
    
    # Plotting the function
    plt.plot(x, y)
    plt.ylabel( 'Values of y(x) ---> ' )
    plt.xlabel( 'Values of x ---> ' )
    plt.show()
    
    plt.plot(x, p)
    plt.ylabel( 'Values of y\'(x) ---> ' )
    plt.xlabel( 'Values of x ---> ' )
    plt.show()
    
    
# Thomas Algo for Block Tridiagonal System
def blockTDMA(a, b, c, d):
    
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
    
    n = len(b)
    
    C_upd = []
    C_upd.append(np.matmul(np.linalg.inv(b[0]), c[0]))
    for i in range(1,n-1):
        C_upd.append(np.matmul(np.linalg.inv(b[i] - np.matmul(a[i],C_upd[i-1])), c[i]))
    
    D_upd = []
    D_upd.append(np.matmul(np.linalg.inv(b[0]), d[0]))
    for i in range(1,n):
        D_upd.append(np.matmul(np.linalg.inv(b[i] - np.matmul(a[i],C_upd[i-1])), d[i]-np.matmul(a[i], D_upd[i-1])))
    
    x = []
    x.append(np.array(D_upd[n-1]))
    for i in range(n-2,-1,-1):
        x.insert(0, np.array(D_upd[i] - np.matmul(C_upd[i], x[0])))
    
    return x





# h -> grid size
h = 0.01

#define the block matrices for Xi-1(ai), Xi(bi) and Xi+1(ci)
ai = [[0.0, 1/(h*h)],[1/(h*h), 0.0]]
bi = [[81.0, -2/(h*h)],[-2/(h*h), -1.0]]
ci = [[0.0, 1/(h*h)],[1/(h*h), 0.0]]

# st -> start point, en -> end point
st = 0
en = 1

# N = number of grid points
N = (int)((en-st)/h)

ai = np.array(ai)
a = []
for i in range(1,N):
    a.append(ai)
a = np.array(a)

bi = np.array(bi)
b = []
for i in range(1,N):
    b.append(bi)
b = np.array(b)

ci = np.array(ci)
c = []
for i in range(1,N):
    c.append(ci)
c = np.array(c)

# set di values according to the question
d = []
di = [0.0, 0.0]
for i in range(1,N):
    di_ = di.copy()
    di_[0] = 81*((i*h)*(i*h))
    d.append(di_)
d = np.array(d)


# set boundary values
X0 = np.array([0, 0])
Xn = np.array([0, 0])

# set values of a0 and cn-1
a0 = [[0.0, 1/(h*h)],[1/(h*h), 0.0]]
cn_1 = [[0.0, 1/(h*h)],[1/(h*h), 0.0]]

# update the first and last value in d array
d[0] = d[0] - np.matmul(a0, X0)
d[-1] = d[-1] - np.matmul(cn_1, Xn)


print("The values are: ")
res = blockTDMA(a,b,c,d)
res.insert(0,X0)
res.append(Xn)
showResult(res,st,h)

