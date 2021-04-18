import numpy as np

# Thomas Algorithm for tridiagonal system of equations
def thomas (a, b, c, d):
    
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
    
    n = len(b)
    arr = np.zeros(n)
    
    # elimination
    for k in range(1,n):
        val  = a[k]/b[k-1]
        b[k] = b[k] - c[k-1]*val
        d[k] = d[k] - d[k-1]*val
    # a1 to an-1 are used, b0 to bn-1 are used, c0 to cn-2 are used
    
    # back_substitution
    val = d[n-1]/b[n-1]
    arr[n-1] = val
    for k in range(n-2,-1,-1):
        val = (d[k]-c[k]*val)/b[k]
        arr[k] = val
    
    # (n size) numpy array is returned containing the solution
    return arr