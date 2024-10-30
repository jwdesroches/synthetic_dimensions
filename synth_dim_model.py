# --------------------------------------------------------------------------------------------------------------------------------------------
# imports
# --------------------------------------------------------------------------------------------------------------------------------------------

import numpy as np
import copy
from scipy.integrate import solve_ivp
from scipy.sparse.linalg import eigsh
from scipy.linalg import expm

# --------------------------------------------------------------------------------------------------------------------------------------------
# functions
# --------------------------------------------------------------------------------------------------------------------------------------------

def creation_operator(state, site, synthetic_level, N, M): 
    """
    Applies the creation operator to a given state at a specific site and synthetic level.
    
    Parameters:
    state (list): List representing the current state of the system.
    site (int): Site index where the operator is applied.
    synthetic_level (int): Synthetic level for the creation operation.
    N (int): Total number of sites.
    M (int): Total number of synthetic levels.
    
    Returns:
    list or int: Modified state with the creation operation applied, or 0 if operation is invalid.
    """
    
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
    """
    Applies the annihilation operator to a given state at a specific site and synthetic level.
    
    Parameters:
    state (list): List representing the current state of the system.
    site (int): Site index where the operator is applied.
    synthetic_level (int): Synthetic level for the annihilation operation.
    N (int): Total number of sites.
    M (int): Total number of synthetic levels.
    
    Returns:
    list or int: Modified state with the annihilation operation applied, or 0 if operation is invalid.
    """
    
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
    """
    Computes the inner product of two states.
    
    Parameters:
    state1 (list or int): First state in the inner product.
    state2 (list or int): Second state in the inner product.
    
    Returns:
    int: 1 if the states are identical, 0 otherwise.
    """
    
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
    """
    Enumerates all possible states of a system with N lattice sites and M synthetic levels.
    
    Parameters:
    N (int): Number of lattice sites.
    M (int): Number of synthetic levels.
    
    Returns:
    list: A list of all possible states represented as lists of integers.
    list: A list of formatted state strings for easier visualization.
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

    formatted_states = ["|" + ",".join(map(str, state)) + ">" for state in states]
    return states, formatted_states

# --------------------------------------------------------------------------------------------------------------------------------------------

def construct_hamiltonian_slow(N, M, J, V):
    """
    Constructs the Hamiltonian matrix for a system with nearest-neighbor interactions and tunneling terms.

    Parameters:
    N (int): Number of lattice sites.
    M (int): Number of synthetic levels.
    J (float): Tunneling coefficient.
    V (float): Interaction strength.

    Returns:
    np.ndarray: Hamiltonian matrix of size (M^N x M^N).
    """
    
    neighbors = [(i, i+1) for i in range(N-1)]
    H = np.zeros((M**N, M**N)) 
    states, _ = enumerate_states(N=N, M=M)

    for alpha, state1 in enumerate(states):
        for beta, state2 in enumerate(states):
            state1_copy = state1[:]
            state2_copy = state2[:]

            # tunneling term
            for n in range(1, M):
                for j in range(N):
                    intermediate_state = state1_copy[:]
                    intermediate_state = annihilation_operator(intermediate_state, site=j, synthetic_level=n, N=N, M=M)
                    intermediate_state = creation_operator(intermediate_state, site=j, synthetic_level=n-1, N=N, M=M)
                    H[alpha, beta] += -J * inner_product(intermediate_state, state2_copy)

                    intermediate_state = state1_copy[:]
                    intermediate_state = annihilation_operator(intermediate_state, site=j, synthetic_level=n-1, N=N, M=M)
                    intermediate_state = creation_operator(intermediate_state, site=j, synthetic_level=n, N=N, M=M)
                    H[alpha, beta] += -J * inner_product(intermediate_state, state2_copy)

            # interaction term
            for n in range(1, M):
                for i, j in neighbors:
                    intermediate_state = state1_copy[:]
                    intermediate_state = annihilation_operator(intermediate_state, site=j, synthetic_level=n-1, N=N, M=M)
                    intermediate_state = creation_operator(intermediate_state, site=j, synthetic_level=n, N=N, M=M)
                    intermediate_state = annihilation_operator(intermediate_state, site=i, synthetic_level=n, N=N, M=M)
                    intermediate_state = creation_operator(intermediate_state, site=i, synthetic_level=n-1, N=N, M=M)
                    H[alpha, beta] += V * inner_product(intermediate_state, state2_copy)

                    intermediate_state = state1_copy[:]
                    intermediate_state = annihilation_operator(intermediate_state, site=i, synthetic_level=n-1, N=N, M=M)
                    intermediate_state = creation_operator(intermediate_state, site=i, synthetic_level=n, N=N, M=M)
                    intermediate_state = annihilation_operator(intermediate_state, site=j, synthetic_level=n, N=N, M=M)
                    intermediate_state = creation_operator(intermediate_state, site=j, synthetic_level=n-1, N=N, M=M)
                    H[alpha, beta] += V * inner_product(intermediate_state, state2_copy)

    return H

# --------------------------------------------------------------------------------------------------------------------------------------------

def construct_hamiltonian(N, M, J, V):
    """
    Constructs a Hamiltonian matrix using a more efficient method compared to `construct_hamiltonian_slow`.

    Parameters:
    N (int): Number of lattice sites.
    M (int): Number of synthetic levels.
    J (float): Tunneling coefficient.
    V (float): Interaction strength.

    Returns:
    np.ndarray: Hamiltonian matrix of size (M^N x M^N).
    """
    
    dim = M**N
    H = np.zeros((dim, dim), dtype=np.complex128)

    M_powers = np.array([M**i for i in range(N)])

    def index_to_state(index):
        return np.array([(index // M_powers[i]) % M for i in range(N-1, -1, -1)])

    def state_to_index(state):
        return np.dot(state, M_powers[::-1])

    for alpha in range(dim):
        state = index_to_state(alpha)
        for j in range(N):
            for n in range(1, M):
                if state[j] == n:
                    new_state = state.copy()
                    new_state[j] = n - 1
                    beta = state_to_index(new_state)
                    H[alpha, beta] -= J
                    H[beta, alpha] -= J  

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
                    H[beta, alpha] += V  

    return H

# --------------------------------------------------------------------------------------------------------------------------------------------

def exact_diagonalize(H, use_sparse = False, k = 1, verbose=False, check_reconstruction=False):
    """
    Diagonalizes a Hermitian matrix using numpy's `eigh()` method.

    Parameters:
    H (np.ndarray): Hermitian matrix to diagonalize.
    verbose (bool): If True, prints the eigenvalues and eigenvectors.
    check_reconstruction (bool): If True, checks whether the matrix is faithfully reconstructed.

    Returns:
    np.ndarray: Eigenvalues of the matrix.
    list: List of eigenvectors of the matrix.
    """
    
    if not np.allclose(np.conjugate(H.T), H):
            print("The matrix is not Hermitian.")
            return None, None
    
    if use_sparse:
        eigenvalues, eigenvectors = eigsh(H, k=k)
        
        return eigenvalues, [eigenvectors[:, i] for i in range(k)]
        
    else:
        eigenvalues, eigenvectors = np.linalg.eigh(H)
        
        if verbose:
            print("Eigenvalues:\n", eigenvalues)
            print("Eigenvectors:\n", eigenvectors)

        if check_reconstruction:
            reconstruction = np.matmul(eigenvectors, np.matmul(np.diag(eigenvalues), np.linalg.inv(eigenvectors)))
            print("Original matrix:\n", H)
            print("Reconstructed matrix:\n", reconstruction)
            print("Error:\n", H - reconstruction)

        return eigenvalues, [eigenvectors[:, i] for i in range(H.shape[0])]
    
# --------------------------------------------------------------------------------------------------------------------------------------------

def create_H_key(formatted_states):
    """
    Create a matrix H_key where each element is a formatted string combination of state indices, representing the 
    action of the Hamiltonian operator between states.
    
    Parameters:
    formatted_states (list): List of formatted state strings, where each state is represented in the form "|x_1, ..., x_N>".
    
    Returns:
    np.ndarray: H_key matrix of size (M^N x M^N), where each element is a formatted string "<state_x|H|state_y>".
    """
    
    M_pow_N = len(formatted_states)
    H_key = np.empty((M_pow_N, M_pow_N), dtype=object)

    for x in range(M_pow_N):
        for y in range(M_pow_N):
            # construct formatted string "<state_x|H|state_y>"
            H_key[x, y] = "<" + formatted_states[x][::-1][1:] + "H" + formatted_states[y]

    return H_key

# --------------------------------------------------------------------------------------------------------------------------------------------

def sigma_ij(i, j, ground_state_wavefunction, states, N, M):
    """
    Computes the sigma value, representing the difference in synthetic dimension space between sites i and j, weighted 
    by the ground state wavefunction.

    Parameters:
    i (int): Index of the first site.
    j (int): Index of the second site.
    ground_state_wavefunction (np.ndarray): Ground state wavefunction coefficients for each state.
    states (list of lists): List of all states, where each state is represented as a list of occupation numbers.
    N (int): Number of sites.
    M (int): Number of synthetic levels (states per site).

    Returns:
    float: The sigma value, summing the occupation differences weighted by the ground state wavefunction.
    """
    
    sigma = 0
    dim = M**N
    
    for m in range(M):
        for n in range(M):
            for k in range(dim):
                if states[k][i] == m:  
                    if states[k][j] == n: 
                        sigma += abs(m - n) * np.linalg.norm(ground_state_wavefunction[k])**2
                    else:
                        sigma += 0
                else:
                    sigma += 0
    
    return sigma

# --------------------------------------------------------------------------------------------------------------------------------------------

def estimate_ground_state_degeneracy_zero_J(N, M):
    """
    Estimates the degeneracy of the ground state for a system with zero tunneling (J = 0).

    Parameters:
    N (int): Number of lattice sites.
    M (int): Number of synthetic levels (states per site).

    Returns:
    float: Estimated ground state degeneracy at J = 0.
    """
    
    return M * 2**(N / 2)

# --------------------------------------------------------------------------------------------------------------------------------------------

def construct_initial_hamiltonian(N, M, mu):
    """
    Constructs the initial Hamiltonian with the term H = -mu * sum_{i=0}^{N-1} c_{0,j}^\dagger c_{0,j}, which 
    represents the action of a chemical potential on the states at site 0.
    
    Parameters:
    N (int): Number of real lattice sites (synthetic levels).
    M (int): Number of states per site (synthetic levels per site).
    mu (float): Chemical potential acting on the system.

    Returns:
    np.ndarray: Initial Hamiltonian matrix of size (M^N x M^N), where each diagonal element corresponds to the 
    chemical potential term applied to the site occupation numbers.
    """
    
    dim = M**N
    H = np.zeros((dim, dim), dtype=np.complex128)

    # precompute powers of M for faster state-to-index conversion
    M_powers = np.array([M**i for i in range(N)])

    # helper function to convert state index to state representation
    def index_to_state(index):
        return np.array([(index // M_powers[i]) % M for i in range(N-1, -1, -1)])

    # apply the term -mu * c_{0,j}^\dagger * c_{0,j}
    for alpha in range(dim):
        state = index_to_state(alpha)
        for j in range(N):
            if state[j] == 0:  
                H[alpha, alpha] -= mu  

    return H

# --------------------------------------------------------------------------------------------------------------------------------------------

def interpolate_hamiltonian(H0, Hf, t, t_total):
    """
    Interpolates between the initial Hamiltonian H0 and the final Hamiltonian Hf over time t. This function 
    is used to create a time-dependent Hamiltonian that smoothly transitions from H0 to Hf.
    
    Parameters:
    H0 (np.ndarray): Initial Hamiltonian matrix.
    Hf (np.ndarray): Final Hamiltonian matrix.
    t (float): Current time step.
    t_total (float): Total time for the adiabatic evolution.

    Returns:
    np.ndarray: The interpolated Hamiltonian matrix at time t, representing a linear combination of H0 and Hf.
    """
    return (1 - t / t_total) * H0 + (t / t_total) * Hf

# --------------------------------------------------------------------------------------------------------------------------------------------

def evolve_wavefunction(psi, H, dt, hbar=1.0):
    """
    Evolves the wavefunction psi under the Hamiltonian H using a unitary time evolution operator U = exp(-iH * dt / hbar).
    This function applies the time evolution over a time step dt.
    
    Parameters:
    psi (np.ndarray): The current wavefunction as a column vector.
    H (np.ndarray): The Hamiltonian matrix representing the system at the current time step.
    dt (float): Time step for the evolution.
    hbar (float, optional): Reduced Planck's constant (default is 1.0).

    Returns:
    np.ndarray: The evolved wavefunction after the time step dt.
    """
    U = expm(-1j * H * dt)
    psi = np.dot(U, psi)
    return psi

# --------------------------------------------------------------------------------------------------------------------------------------------

def calculate_adiabatic_properties(N, M, init_mu, init_J, init_V, final_J, final_V, final_mu = 0, t_total=10, dt=0.01, initial_state = None):
    """
    Computes various adiabatic properties during the time evolution of a quantum system, starting with an 
    initial Hamiltonian and ending with a final Hamiltonian. This function simulates the adiabatic evolution, 
    computes energy levels, overlaps, and other properties at each time step.
    
    Parameters:
    N (int): Number of real lattice sites (synthetic levels).
    M (int): Number of states per site (synthetic levels per site).
    mu (float): Chemical potential acting on the system.
    init_J (float): Initial hopping amplitude for the system.
    init_V (float): Initial interaction strength.
    J (float): Final hopping amplitude for the system.
    V (float): Final interaction strength.
    t_total (float): Total time for the adiabatic evolution.
    dt (float): Time step size for numerical evolution.

    Returns:
    tuple: A tuple containing:
        - adiabatic_energies (np.ndarray): The adiabatic energies at each time step.
        - adiabatic_diff (np.ndarray): The difference between the adiabatic and true energies at each time step.
        - adiabatic_wavefunctions (np.ndarray): The adiabatic wavefunctions at each time step.
        - overlaps_all_states (np.ndarray): Overlap between the adiabatically evolved wavefunction and all eigenstates.
        - true_energies (np.ndarray): True energies of the system at each time step.
        - energy_gaps (np.ndarray): Energy gaps between the ground and excited states at each time step.
        - times (np.ndarray): The array of time points during the adiabatic evolution.
    """
    
    n_excited_states = M**N

  
    initial_hamiltonian = construct_initial_hamiltonian(N, M, init_mu) + construct_hamiltonian(N, M, init_J, init_V)
    final_hamiltonian = construct_initial_hamiltonian(N, M, final_mu) + construct_hamiltonian(N, M, final_J, final_V)

    times = np.linspace(0, t_total, int(t_total / dt))

    if np.all(initial_state) == None:
        _, eigenvectors_0 = exact_diagonalize(initial_hamiltonian)
        psi_0 = eigenvectors_0[0]
    else: 
        psi_0 = initial_state

    adiabatic_wavefunctions = []
    true_energies = []
    adiabatic_energies = []
    adiabatic_overlaps = []

    psi = psi_0.copy()

    for t in times:
        instantaneous_hamiltonian = interpolate_hamiltonian(initial_hamiltonian, final_hamiltonian, t, t_total)
        
        eigenvalues, eigenvectors = exact_diagonalize(instantaneous_hamiltonian)
        true_energies.append(eigenvalues)
        
        psi = evolve_wavefunction(psi, instantaneous_hamiltonian, dt)
        psi = psi / np.linalg.norm(psi)
        
        adiabatic_wavefunctions.append(psi)
        
        adiabatic_energy = np.real(np.conj(psi).T @ instantaneous_hamiltonian @ psi)
        adiabatic_energies.append(adiabatic_energy)
        
        overlaps = [np.abs(np.conj(eigenvectors[i]).T @ psi)**2 for i in range(n_excited_states)]
        adiabatic_overlaps.append(overlaps)

    true_energies = np.array(true_energies)
    adiabatic_overlaps = np.array(adiabatic_overlaps)

    adiabatic_diff = adiabatic_energies - true_energies[:, 0]
    adiabatic_diff = np.array(adiabatic_diff)
    
    energy_gaps = [eigenvalues - eigenvalues[0] for eigenvalues in true_energies]
    true_energies = np.array(true_energies)
    energy_gaps = np.array(energy_gaps)
    
    return adiabatic_energies, adiabatic_diff, adiabatic_wavefunctions, adiabatic_overlaps, true_energies, energy_gaps, times

# --------------------------------------------------------------------------------------------------------------------------------------------

def intermediate_hamiltonian(N, M, init_J, init_V, V, mu):
    "test"
    return construct_initial_hamiltonian(N, M, mu) + construct_hamiltonian(N, M, init_J, init_V)