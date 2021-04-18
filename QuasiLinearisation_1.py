#Q. y''' - yy'' - (y')^2 + 1 = 0, y(0) = 0, y'(0) = 0, y'(10) = 1, h = 5, 2, 1, 0.1
# Quasi Linearisation Technique


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
    for k in range(n-2 , -1 , -1):
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


def quasiLinearisation(Y, F, h, N):
    
    #define the block matrices for Xi-1(ai), Xi(bi) and Xi+1(ci)
    ai = [[0.0, 0.0],[-1.0, 0.0]]
    bi = [[0.0, 0.0],[ 1.0, 0.0]]
    ci = [[0.0, 0.0],[ 0.0, 0.0]]
    
    ai = np.array(ai)
    a = []
    for i in range(1,N):
        ai_ = ai.copy()
        ai_[0][1] = (1/(h*h))-(Y[i]/(2*h))
        ai_[1][1] = -h/2
        a.append(ai_)
    a = np.array(a)
    
    bi = np.array(bi)
    b = []
    for i in range(1,N):
        bi_ = bi.copy()
        bi_[0][0] = (F[i+1]-F[i-1])/(2*h)
        bi_[0][1] = (-2/(h*h))-(2*F[i])
        bi_[1][1] = -h/2
        b.append(bi_)
    b = np.array(b)
    
    ci = np.array(ci)
    c = []
    for i in range(1,N):
        ci_ = ci.copy()
        ci_[0][1] = (1/(h*h))+(Y[i]/(2*h))
        c.append(ci_)
    c = np.array(c)
    
    # set di values according to the question
    d = []
    di = [0, 0]
    for i in range(1,N):
        di_ = di.copy()
        di_[0] = (Y[i]*F[i+1]/(2*h))-(Y[i]*F[i-1]/(2*h))-(F[i]*F[i])-1
        d.append(di_)
    d = np.array(d)
    
    
    # set boundary values
    Y0 = np.array([0, 0])
    Yn = np.array([0, 1])
    a0 = a[0]
    cn_1 = c[-1]
    
    # update the first and last value in d array
    d[0] = d[0] - np.matmul(a0, Y0)
    d[-1] = d[-1] - np.matmul(cn_1, Yn)
    
    
    updValue = blockTDMA(a,b,c,d)
    
    updValue.insert(0,[Y[0],F[0]])
    updValue.append([Y[-1],F[-1]])
    
    error = 0.0
    for i in range(N+1):
        if(error < abs(updValue[i][0]-Y[i])):
            error = abs(updValue[i][0]-Y[i])
    print("error = ",error)
    
    for i in range(N):
        Y[i] = updValue[i][0]
        F[i] = updValue[i][1]
    
    return error
    


# error will tell the max error at (k+1)th iteration
error = 10000000000.0

# epsilon is the admissible error
epsilon = 0.00000001

# iteration_num keeps count of total iterations
iteration_num = 0

# h(or dx) = grid size in x axis
h = 0.1

# st -> start point, en -> end point
st = 0
en = 10

# N = number of x-axis points
N = (int)((en - st)/h)


Y = np.zeros(N+1)
F = np.zeros(N+1)

# initial values for Y and F(= y') are as follows:
# y = (x^2)/20, F = x/10
for i in range(N+1):
    Y[i] =  ((st+i*h)**2)/20
    F[i] = (st+i*h)/10

print("Initial Guess:")
print("y at -> ")
showResult(Y,st,h)
print("y' at -> ")
showResult(F,st,h)


# call quasiLinearisation until error is within bound
while(error > epsilon and iteration_num < 500):
    iteration_num += 1
    print("Iteration: ",iteration_num)
    error = quasiLinearisation(Y, F, h, N)
    print("y at -> ")
    showResult(Y,st,h)
    print("y' at -> ")
    showResult(F,st,h)

