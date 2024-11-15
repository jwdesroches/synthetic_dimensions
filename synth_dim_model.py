# --------------------------------------------------------------------------------------------------------------------------------------------
# imports
# --------------------------------------------------------------------------------------------------------------------------------------------

import numpy as np
import copy
from scipy.integrate import solve_ivp
from scipy.sparse.linalg import eigsh
from scipy.linalg import expm
import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap

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

# --------------------------------------------------------------------------------------------------------------------------------------------

def construct_rescaled_hamiltonian(N, M, V, mu_V_ratio, J_V_ratio):
    """
    Constructs a rescaled Hamiltonian matrix for a quantum system with N sites and M states per site, 
    incorporating chemical potential, tunneling, and interaction terms. The Hamiltonian is normalized 
    by the absolute value of V to produce H_tilde.

    Parameters:
    N (int): Number of sites in the system.
    M (int): Number of states per site.
    V (float): Interaction strength.
    mu_V_ratio (float): Ratio of the chemical potential (mu) to the interaction strength (V).
    J_V_ratio (float): Ratio of the tunneling parameter (J) to the interaction strength (V).

    Returns:
    np.ndarray: The rescaled Hamiltonian matrix H_tilde (normalized by |V|).
    """
    mu = mu_V_ratio * abs(V)
    J = J_V_ratio * abs(V)
    dim = M**N
    H = np.zeros((dim, dim), dtype=np.complex128)

    # Precompute powers of M for faster state-to-index conversion
    M_powers = np.array([M**i for i in range(N)])

    # Helper function to convert a state index to a state representation (array of states)
    def index_to_state(index):
        return np.array([(index // M_powers[i]) % M for i in range(N-1, -1, -1)])
    
    # Helper function to convert a state representation (array of states) back to an index
    def state_to_index(state):
        return np.dot(state, M_powers[::-1])

    # Apply the chemical potential term
    for alpha in range(dim):
        state = index_to_state(alpha)
        for j in range(N):
            if state[j] == 0:
                H[alpha, alpha] -= mu

    # Apply the tunneling term
    for alpha in range(dim):
        state = index_to_state(alpha)
        for j in range(N):
            for n in range(1, M):
                if state[j] == n:
                    new_state = state.copy()
                    new_state[j] = n - 1
                    beta = state_to_index(new_state)
                    H[alpha, beta] -= J
                    H[beta, alpha] -= J  # Ensure Hermitian symmetry

    # Apply the interaction term
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
                    H[beta, alpha] += V  # Ensure Hermitian symmetry
                    
    # Rescale H to H_tilde by dividing by |V|
    H_tilde = H / abs(V)
    
    return H_tilde

# --------------------------------------------------------------------------------------------------------------------------------------------

def simulate_adiabatic_evolution(N, M, V, mu_V_ratio_routine, J_V_ratio_routine, times, dt, initial_state=None):
    """
    Simulates the adiabatic evolution of a quantum system by evolving a wavefunction under a time-dependent Hamiltonian.
    Tracks the evolution of wavefunctions, energies, overlaps, and probabilities throughout the process.

    Parameters:
    N (int): Number of sites in the system.
    M (int): Number of states per site.
    V (float): Interaction strength.
    mu_V_ratio_routine (list or np.ndarray): Time-dependent values of the chemical potential ratio (mu/V).
    J_V_ratio_routine (list or np.ndarray): Time-dependent values of the tunneling ratio (J/V).
    times (list or np.ndarray): Discrete time steps over which the evolution is simulated.
    dt (float, optional): Time step size for evolution (default is 0.1).
    initial_state (np.ndarray, optional): Initial wavefunction as a column vector. If None, uses the ground state of the initial Hamiltonian.

    Returns:
    tuple: A tuple containing:
        - adiabatic_energies (list): Energies of the evolved state at each time step.
        - adiabatic_diff (np.ndarray): Difference between adiabatic energies and the ground state energy at each time step.
        - adiabatic_wavefunctions (list): Wavefunctions evolved adiabatically over the simulation.
        - adiabatic_probabilities (np.ndarray): Probabilities of projection onto each eigenstate at each time step.
        - adiabatic_overlaps (np.ndarray): Overlaps of the evolved state with each instantaneous eigenstate at each time step.
        - true_energies (np.ndarray): Eigenvalues of the instantaneous Hamiltonian at each time step.
        - energy_gaps (np.ndarray): Energy gaps between each eigenvalue and the ground state energy at each time step.
    """
    n_excited_states = M**N
    initial_hamiltonian = construct_rescaled_hamiltonian(N, M, V, mu_V_ratio_routine[0], J_V_ratio_routine[0])
    
    # Determine the initial state
    if initial_state is None:
        _, eigenvectors_0 = exact_diagonalize(initial_hamiltonian)
        psi_0 = eigenvectors_0[0]  # Start with the ground state
    else:
        psi_0 = initial_state

    # Initialize storage lists
    adiabatic_energies = []
    adiabatic_wavefunctions = []
    adiabatic_probabilities = []
    adiabatic_overlaps = []
    true_energies = []

    # Start with the initial state
    psi = psi_0.copy()

    # Time evolution loop
    for index, t in enumerate(times):
        # Construct instantaneous Hamiltonian at the current time step
        instantaneous_hamiltonian = construct_rescaled_hamiltonian(N, M, V, mu_V_ratio_routine[index], J_V_ratio_routine[index])
        
        # Compute exact eigenvalues and eigenvectors
        eigenvalues, eigenvectors = exact_diagonalize(instantaneous_hamiltonian)
        true_energies.append(eigenvalues)
        
        # Evolve the wavefunction
        psi = evolve_wavefunction(psi, instantaneous_hamiltonian, dt)
        psi = psi / np.linalg.norm(psi)  # Normalize the wavefunction
        
        # Store the evolved wavefunction
        adiabatic_wavefunctions.append(psi)
        
        # Compute adiabatic energy
        adiabatic_energy = np.real(np.conj(psi).T @ instantaneous_hamiltonian @ psi)
        adiabatic_energies.append(adiabatic_energy)
        
        # Compute overlaps and probabilities with instantaneous eigenstates
        overlap = [np.dot(np.conj(eigenvectors[i]).T, psi) for i in range(n_excited_states)] 
        probability = [np.abs(np.conj(eigenvectors[i]).T @ psi)**2 for i in range(n_excited_states)]
               
        adiabatic_probabilities.append(probability)
        adiabatic_overlaps.append(overlap)

 
    # Convert collected data to arrays
    adiabatic_energies = np.array(adiabatic_energies)
    adiabatic_wavefunctions = np.array(adiabatic_wavefunctions)
    adiabatic_probabilities = np.array(adiabatic_probabilities)
    adiabatic_overlaps = np.array(adiabatic_overlaps)
    true_energies = np.array(true_energies)
        
    return adiabatic_energies, adiabatic_wavefunctions, adiabatic_probabilities, adiabatic_overlaps, true_energies

# --------------------------------------------------------------------------------------------------------------------------------------------

def multistep_adiabatic_evolution(N, M, V, dt, mu_V_ratio_steps, J_V_ratio_steps, step_times):
    
    num_steps = len(step_times)
    results = []
    initial_state = None
    accumulated_time = 0

    for i in range(num_steps):
        total_time = step_times[i]
        total_steps = int(total_time / dt)
        times = np.linspace(0, total_time, total_steps)
        mu_V_ratio_routine = np.linspace(mu_V_ratio_steps[i][0], mu_V_ratio_steps[i][-1], total_steps) # simple linear interpolation
        J_V_ratio_routine = np.linspace(J_V_ratio_steps[i][0], J_V_ratio_steps[i][-1], total_steps) # simple linear interpolation

        adiabatic_energies, adiabatic_wavefunctions, adiabatic_probabilities, adiabatic_overlaps, true_energies = \
            simulate_adiabatic_evolution(N, M, V, mu_V_ratio_routine, J_V_ratio_routine, times, dt=dt, initial_state=initial_state)
        
        results.append({
            "times": times + accumulated_time,
            "adiabatic_probabilities": adiabatic_probabilities,
            "adiabatic_wavefunctions": adiabatic_wavefunctions,
            "adiabatic_energies": adiabatic_energies,
            "adiabatic_overlaps": adiabatic_overlaps,
            "true_energies": true_energies
        })
        
        initial_state = adiabatic_wavefunctions[-1]
        accumulated_time += total_time

    return results

# --------------------------------------------------------------------------------------------------------------------------------------------

def plot_multistep_adiabatic_evolution_results(results, N, M, J_V_ratio_steps, step_times):
    # Plotting adiabatic state probabilities
    fig, ax = plt.subplots()
    colors = get_cmap("gist_rainbow", M**N)
    
    for index in range(M**N):
        for step_count, step_result in enumerate(results):
            times = step_result["times"]
            probabilities = step_result["adiabatic_probabilities"][:, index]
            color = "k" if index == 0 else colors(index)
            label = "Ground State" if (index == 0)&(step_count == len(results)-1) else ""
            
            ax.plot(times, probabilities, color=color, label=label)
    
    # Add vertical lines indicating the boundaries between steps
    accumulated_time = 0
    for time in step_times[:-1]: 
        accumulated_time += time
        ax.axvline(accumulated_time, linestyle="--", color="k")
    
    ax.grid()
    ax.set_title(f"Adiabatic Probabilities: $N={N}$, $M={M}$, $V<0$, $(J/|V|)_f = {J_V_ratio_steps[-1][-1]}$")
    ax.set_xlabel("Time [$t/|V|$]")
    ax.set_ylabel("Probability")
    ax.legend()

    # Plotting adiabatic energies vs true energies
    fig, ax = plt.subplots()
    cmap = get_cmap("gist_rainbow", M**N)
    
    
    for idx in range(len(results)):
        for idx2 in range(M**N):
            ax.plot(results[idx]["times"], results[idx]["true_energies"][:, idx2], color=cmap(idx2))
        
        if idx == len(results)-1:
            ax.plot(results[idx]["times"], results[idx]["adiabatic_energies"], '--k', label="Adiabatic Energy")
        else:
            ax.plot(results[idx]["times"], results[idx]["adiabatic_energies"], '--k')
        
        ax.legend(loc="upper center")
        ax.set_title(f"Energies: $N={N}$, $M={M}$, $V<0$, $(J/|V|)_f = {J_V_ratio_steps[-1][-1]}$")
        ax.set_xlabel("Time [$t/|V|$]")
        ax.set_ylabel("Energy [$E/|V|$]")
    
    accumulated_time = 0
    for time in step_times[:-1]: 
        accumulated_time += time
        ax.axvline(accumulated_time, linestyle="--", color="k")

    fig, ax = plt.subplots()
    
    for idx in range(len(results)):
        for idx2 in range(M**N):
            if idx2 == 0:
                continue
            ax.plot(results[idx]["times"], np.real(results[idx]["adiabatic_overlaps"][:, idx2]), '.', color=cmap(idx2))
        
        if idx == len(results)-1:
            ax.plot(results[idx]["times"], np.real(results[idx]["adiabatic_overlaps"][:, 0]), '.k', label="Ground State")
        else:
            ax.plot(results[idx]["times"], np.real(results[idx]["adiabatic_overlaps"][:, 0]), '.k')
        
        ax.legend(loc="lower center")
        ax.set_title(f"$\\Re[<\\psi_i|\\psi_a>]$: $N={N}$, $M={M}$, $V<0$, $(J/|V|)_f = {J_V_ratio_steps[-1][-1]}$")
        ax.set_xlabel("Time [$t/|V|$]")
        ax.set_ylabel("Real Overlap")
        
    accumulated_time = 0
    for time in step_times[:-1]: 
        accumulated_time += time
        ax.axvline(accumulated_time, linestyle="--", color="k")
    
    # Plotting imaginary part of overlaps
    fig, ax = plt.subplots()
    
    for idx in range(len(results)):
        for idx2 in range(M**N):
            if idx2 == 0:
                continue
            ax.plot(results[idx]["times"], np.imag(results[idx]["adiabatic_overlaps"][:, idx2]), ".", color=cmap(idx2))
        
        if idx == len(results)-1:
            ax.plot(results[idx]["times"], np.imag(results[idx]["adiabatic_overlaps"][:, 0]), '.k', label="Ground State")
        else:
            ax.plot(results[idx]["times"], np.imag(results[idx]["adiabatic_overlaps"][:, 0]), '.k')
        
        ax.legend(loc="lower center")
        ax.set_title(f"$\\Im[<\\psi_i|\\psi_a>]$: $N={N}$, $M={M}$, $V<0$, $(J/|V|)_f = {J_V_ratio_steps[-1][-1]}$")
        ax.set_xlabel("Time [$t/|V|$]")
        ax.set_ylabel("Imaginary Overlap")
        
    accumulated_time = 0
    for time in step_times[:-1]: 
        accumulated_time += time
        ax.axvline(accumulated_time, linestyle="--", color="k")
