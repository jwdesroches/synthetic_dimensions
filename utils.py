# --------------------------------------------------------------------------------------------------------------------------------
# imports
# --------------------------------------------------------------------------------------------------------------------------------

import numpy as np

# --------------------------------------------------------------------------------------------------------------------------------
# functions
# --------------------------------------------------------------------------------------------------------------------------------

def diagonalize(H, verbose = False, check_reconstruction = False):
    """
    functionality: diagonalizes a matrix using numpy.linalg.eigh(). This implementation is intended for efficiency and should only be used
    for small matrices. 
    
    inputs: 
    H [type: np.ndarray]; a (hermitian) matrix to be diagonalized. 
    verbose [type: bool]; a variable that controls whether the outputs are printed or not.  
    
    outputs:
    D [type: np.ndarray], a matrix that contains the eigenvalues of H along its diagonals.
    V [type: np.ndarray]; a matrix whose columns correspond to the eigenvectors of H.
    """
    
    if np.allclose(np.conjugate(H.T),H) == False:
        print("The matrix is not Hermitian. Please check the input matrix.")
        return None, None
    
    eigenvalues, eigenvectors = np.linalg.eigh(H)
    D = np.diag(eigenvalues)
    V = eigenvectors 
    
    if verbose == True:
        print("D = \n", D, "\n")
        print("V = \n", V, "\n")
        
    if check_reconstruction == True:
        reconstructed_H = V @ D @ np.conjugate(V.T)
        if np.allclose(reconstructed_H,H):
            print("Faithfully reconstructed the matrix.")
        else: 
            print("Reconstruction failed.")
    
    return V, D    

# --------------------------------------------------------------------------------------------------------------------------------

def kronecker_delta(i,j):
    """
    functionality: evaluates the kronecker delta between two inputs, i and j. If i == j, then return 1. Otherwise, return 0.
    
    inputs:
    i [type: float, int]; the first input to evaluate the kronecker delta
    j [type: float, int]; the second input to evaluate the kronecker delta
    
    output:
    result [type: float, int]; the result of the kronecker delta
    
    """
    
    if i == j:
        result = 1
    else: 
        result = 0
        
    return result
    
# --------------------------------------------------------------------------------------------------------------------------------

def placeholder():
    return None
    
# --------------------------------------------------------------------------------------------------------------------------------
