# --------------------------------------------------------------------------------------------------------------------------------------------
# imports
# --------------------------------------------------------------------------------------------------------------------------------------------

import numpy as np
import copy

# --------------------------------------------------------------------------------------------------------------------------------------------
# functions
# --------------------------------------------------------------------------------------------------------------------------------------------

def creation_operator(state, site, synthetic_level, N, M): 
    """ Documentation to be added."""
    
    if state == 0:
        return 0
    if site not in range(0, N):
        print("The site is not in range.")
        return state
    if synthetic_level not in range(0, M):
        print("The synthetic level is not in range.")
        return state
    if state[site] is None:
        state[site] = synthetic_level
    else:
        return 0
    return state

# --------------------------------------------------------------------------------------------------------------------------------------------

def annihilation_operator(state, site, synthetic_level, N, M):
    # add documentation later

    if state == 0:
        return 0
    if site not in range(0, N):
        print("The site is not in range.")
        return state
    if synthetic_level not in range(0, M):
        print("The synthetic level is not in range.")
        return state
    if state[site] == synthetic_level:
        state[site] = None
    else: 
        return 0
    return state

# --------------------------------------------------------------------------------------------------------------------------------------------

def inner_product(state1, state2):
    # add documentation later

    
    if state1 == 0:
        return 0
    elif state2 == 0:
        return 0 
    else:
        if state1 == state2:
            return 1
        else:
            return 0
        
# --------------------------------------------------------------------------------------------------------------------------------------------

def enumerate_states(N, M):
    # add documentation later

    
    """
    Function to enumerate all possible states of a system with N lattice sites and M synthetic levels.
    
    Inputs:
    N (int): Number of lattice sites.
    M (int): Number of synthetic levels.
    
    Returns:
    states (list): List of all possible states, each state represented as a list of integers.
    formatted_states (list): List of formatted state strings for easy visualization.
    """
    
    if N <= 0 or M <= 0:
        return [], []

    states = []
    current_state = [0] * N

    while True:
        states.append(current_state.copy())

        for i in range(N-1, -1, -1):
            if current_state[i] < M-1:
                current_state[i] += 1
                break
            else:
                current_state[i] = 0
        else:
            break

    if M**N == len(states):
        formatted_states = ["|" + ",".join(map(str, state)) + ">" for state in states]
        return states, formatted_states
    
    else:
        print("There was an issue enumerating the states.") 
        return [], []

# --------------------------------------------------------------------------------------------------------------------------------------------

def construct_hamiltonian_slow(N, M, J, V):
    # add documentation later

    neighbors = [(i, i+1) for i in range(N-1)]  # nearest neighbor sites in 1D
    H = np.zeros((M**N, M**N))  # initialize Hamiltonian
    states, _ = enumerate_states(N=N, M=M)

    for alpha, state1 in enumerate(states):
        for beta, state2 in enumerate(states):
            state1_copy = state1[:]
            state2_copy = state2[:]

            # tunneling term
            for n in range(1, M):
                for j in range(N):
                    # first term: c_{n-1,j}^\dagger * c_{n,j}
                    intermediate_state = state1_copy[:]
                    intermediate_state = annihilation_operator(intermediate_state, site=j, synthetic_level=n, N=N, M=M)
                    intermediate_state = creation_operator(intermediate_state, site=j, synthetic_level=n-1, N=N, M=M)
                    H[alpha, beta] += -J * inner_product(intermediate_state, state2_copy)

                    # second term (h.c.): c_{n,j}^\dagger * c_{n-1,j}
                    intermediate_state = state1_copy[:]
                    intermediate_state = annihilation_operator(intermediate_state, site=j, synthetic_level=n-1, N=N, M=M)
                    intermediate_state = creation_operator(intermediate_state, site=j, synthetic_level=n, N=N, M=M)
                    H[alpha, beta] += -J * inner_product(intermediate_state, state2_copy)

            # interaction term
            for n in range(1, M):
                for i, j in neighbors:
                    # first term: c_{n-1,i}^\dagger * c_{n,i}^\dagger * c_{n,j}^\dagger * c_{n-1,j}
                    intermediate_state = state1_copy[:]
                    intermediate_state = annihilation_operator(intermediate_state, site=j, synthetic_level=n-1, N=N, M=M)
                    intermediate_state = creation_operator(intermediate_state, site=j, synthetic_level=n, N=N, M=M)
                    intermediate_state = annihilation_operator(intermediate_state, site=i, synthetic_level=n, N=N, M=M)
                    intermediate_state = creation_operator(intermediate_state, site=i, synthetic_level=n-1, N=N, M=M)
                    H[alpha, beta] += V * inner_product(intermediate_state, state2_copy)
                    
                    # second term (h.c.): c_{n-1,j}^\dagger * c_{n,j} * c_{n,i} *c_{n-1,i}
                    intermediate_state = state1_copy[:]
                    intermediate_state = annihilation_operator(intermediate_state, site=i, synthetic_level=n-1, N=N, M=M)
                    intermediate_state = creation_operator(intermediate_state, site=i, synthetic_level=n, N=N, M=M)
                    intermediate_state = annihilation_operator(intermediate_state, site=j, synthetic_level=n, N=N, M=M)
                    intermediate_state = creation_operator(intermediate_state, site=j, synthetic_level=n-1, N=N, M=M)
                    H[alpha, beta] += V * inner_product(intermediate_state, state2_copy)
                    
    return H

# --------------------------------------------------------------------------------------------------------------------------------------------

def construct_hamiltonian(N, M, J, V):
    # add documentation later

    dim = M**N
    H = np.zeros((dim, dim), dtype=np.complex128)

    # Precompute powers of M for faster state-to-index conversion
    M_powers = np.array([M**i for i in range(N)])

    # Helper function to convert state index to state representation
    def index_to_state(index):
        return np.array([(index // M_powers[i]) % M for i in range(N-1, -1, -1)])

    # Helper function to convert state representation to index
    def state_to_index(state):
        return np.dot(state, M_powers[::-1])

    # Tunneling term
    for alpha in range(dim):
        state = index_to_state(alpha)
        for j in range(N):
            for n in range(1, M):
                if state[j] == n:
                    new_state = state.copy()
                    new_state[j] = n - 1
                    beta = state_to_index(new_state)
                    H[alpha, beta] -= J
                    H[beta, alpha] -= J  # Hermitian conjugate

    # Interaction term
    for alpha in range(dim):
        state = index_to_state(alpha)
        for i in range(N - 1):
            j = i + 1
            for n in range(1, M):
                if state[i] == n and state[j] == n - 1:
                    new_state = state.copy()
                    new_state[i], new_state[j] = n - 1, n
                    beta = state_to_index(new_state)
                    H[alpha, beta] += V
                    H[beta, alpha] += V  # Hermitian conjugate

    return H
# --------------------------------------------------------------------------------------------------------------------------------------------

def exact_diagonalize(H, verbose=False, check_reconstruction=False):
    """
    Diagonalize a matrix using numpy.linalg.eigh().
    
    Inputs:
    H (np.ndarray): Hermitian matrix to be diagonalized.
    verbose (bool): Controls whether the eigenvalues and eigenvectors are printed.
    check_reconstruction (bool): Controls whether the reconstructed matrix is checked against the original.
    
    Outputs: 
    eigenvalues (np.ndarray): Eigenvalues of the matrix H.
    eigenvectors (np.ndarray): Eigenvectors of the matrix H. """
    
    if not np.allclose(np.conjugate(H.T), H):
        print("The matrix is not Hermitian. Please check the input matrix.")
        return None, None
    
    eigenvalues, V = np.linalg.eigh(H)
    D = np.diag(eigenvalues)
    
    if verbose:
        print("D Matrix = \n", D, "\n")
        print("V Matrix = \n", V, "\n")
        
    if check_reconstruction:
        reconstructed_H = V @ D @ np.conjugate(V.T)
        if np.allclose(reconstructed_H, H):
            print("Faithfully reconstructed the matrix.")
        else: 
            print("Reconstruction failed.")
    
    eigenvectors = [V[:, col_idx] for col_idx in range(V.shape[1])]
    
    return eigenvalues, eigenvectors

# --------------------------------------------------------------------------------------------------------------------------------------------

def create_H_key(formatted_states):
    """
    Create a matrix H_key where each element is a formatted string combination of state indices.

    Inputs:
    formatted_states (list): List of formatted state strings.

    Returns:
    H_key (np.ndarray): Matrix where each element is a formatted string combination of state indices.
    """
    
    M_pow_N = len(formatted_states)
    H_key = np.empty((M_pow_N, M_pow_N), dtype=object)

    for x in range(M_pow_N):
        for y in range(M_pow_N):
            H_key[x, y] = "<" + formatted_states[x][::-1][1:] + "H" + formatted_states[y]

    return H_key

# --------------------------------------------------------------------------------------------------------------------------------------------

def sigma_ij(i, j, ground_state_wavefunction, states, N, M):
    # add documentation later
    
    sigma = 0
    dim = M**N
    
    for m in range(M):
        for n in range(M):
            for k in range(dim):
                if states[k][i] == m:
                    if states[k][j] == n:
                        sigma += abs(m-n)*ground_state_wavefunction[k]**2
                    else:
                        sigma += 0
                else:
                    sigma +=0
    
    return sigma

# --------------------------------------------------------------------------------------------------------------------------------------------

def estimate_ground_state_degeneracy_zero_J(N,M):
    # add documentation later
    
    return M * 2**(N/2)