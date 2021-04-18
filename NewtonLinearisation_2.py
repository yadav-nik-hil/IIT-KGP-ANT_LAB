#Q. y''' + (2y+4)y' = 0, y(0) = 0, y''(0) = -k, y'(w) = 0, k = 0.1, w = 0.087, h = 0.0435, 0.0087
# Newton Linearisation Method


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
    for k in range( 1 ,n):
        val = a[k]/b[k -1 ]
        b[k] = b[k] - c[k -1 ]*val
        d[k] = d[k] - d[k -1 ]*val
        
    # back_substitution
    val = d[n -1 ]/b[n -1 ]
    arr[n -1 ] = val
    for k in range(n -2 , -1 , -1 ):
        val = (d[k]-c[k]*val)/b[k]
        arr[k] = val
    
    return arr


def showResult(res,st,h):
    x = []
    for i in range(len(res)):
        x.append(st+i*h)
        print("x = ",'%.3f'%(st+i*h)," = ",'%.10f'%(res[i]))
    print()

    # Plotting the function
    plt.plot(x, res)
    plt.ylabel( 'Values of y(x) ---> ' )
    plt.xlabel( 'Values of x ---> ' )
    plt.show()


def newtonLinearisation(Y, F, h, N):
    
    #define the block matrices for Xi-1(ai), Xi(bi) and Xi+1(ci)
    ai = [[0, 0],[-1, 0]]
    bi = [[0, 0],[ 1, 0]]
    ci = [[0, 0],[ 0, 0]]
    
    k = 0.1 #given in the question
    
    # h = grid size
    h = 0.0435
    # N = number of grid points
    N = (int)((0.087 - 0)/h)
    
    ai = np.array(ai)
    a = []
    a.append([[0,0],[-1,-h/2]])
    for i in range(2,N):
        ai_ = ai.copy()
        ai_[0][1] = (1/(h*h))
        ai_[1][1] = -h/2
        a.append(ai_)
    a = np.array(a)
    
    bi = np.array(bi)
    b = []
    b.append([[3*F[1],(-2/(3*h*h)+(3*Y[1])+4)],[1,-h/2]])
    for i in range(2,N):
        bi_ = bi.copy()
        bi_[0][0] = 2*F[i]
        bi_[0][1] = (-2/(h*h))+(2*Y[i]+4)
        bi_[1][1] = -h/2
        b.append(bi_)
    b = np.array(b)
    
    ci = np.array(ci)
    c = []
    c.append([[0,2/(3*h*h)],[0,0]])
    for i in range(2,N):
        ci_ = ci.copy()
        ci_[0][1] = (1/(h*h))
        c.append(ci_)
    c = np.array(c)
    
    # set boundary values
    Y0 = np.array([0, 0])
    Yn = np.array([0, 0])
    
    # set di values according to the question
    di = [0, 0]
    d = []
    d.append([-(2*(F[2]-F[1])/(3*h*h))-((3*Y[1]+4)*F[1])+(2*k/(3*h)),-Y[1]+Y[0]+h*(F[1]+F[0])/2])
    for i in range(2,N):
        di_ = di.copy()
        di_[0] = (-F[i+1]/(h*h))+(2*F[i]/(h*h))-(F[i-1]/(h*h))-((2*Y[i]+4)*F[i])
        d.append(di_)
    d = np.array(d)
    
    # set the values of a0 and cn_1
    a0 = a[0]
    cn_1 = c[-1]
    
    # update the first and last value in d array
    d[0] = d[0] - np.matmul(a0, Y0)
    d[-1] = d[-1] - np.matmul(cn_1, Yn)
    
    
    delta = blockTDMA(a,b,c,d)
    
    #delta_Y0 = 0, delta_Yn = 0
    #delta_Y0 = 0, delta_Yn = 0
    delta.insert(0,di)
    delta.append(di)
    
    error = 0.0
    for d in delta:
        if(error < abs(d[0])):
            error = abs(d[0])
    print("error = ",error)
    
    Y_ = Y.copy()
    F_ = F.copy()
    
    for i in range(N):
        Y[i] = Y_[i] + delta[i][0]
        F[i] = F_[i] + delta[i][1]
    
    return error
    


# error will tell the max error at (k+1)th iteration
error = 10000000000.0

# epsilon is the admissible error
epsilon = 0.00000001

# iteration_num keeps count of total iterations
iteration_num = 0

k = 0.1 #given in the question

# h = grid size
h = 0.087/20

# st -> start point, en -> end point
st = 0
en = 0.087

# N = number of x-axis points
N = (int)((en - st)/h)


Y = np.zeros(N+1)
F = np.zeros(N+1)

# initial values for Y and F(=y') are as follows:
# y = -k(x+1)^3/6 + k/6  and  F = -k(x+1)^2/2 + k(0.087+1)^2/2
for i in range(N+1):
    Y[i] = -k*((1+i*h)**3)/6 + k/6
    F[i] = -k*((1+i*h)**2)/2 + k*((1+en)**2)/2



# call newtonLinearisation untill error is within bound
while(error > epsilon and iteration_num < 500):
    iteration_num += 1
    print("Iteration: ",iteration_num)
    error = newtonLinearisation(Y, F, h, N)
    print("y at -> ")
    showResult(Y,st,h)
    print("y' at -> ")
    showResult(F,st,h)

