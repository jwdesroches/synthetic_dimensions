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

def exact_diagonalize(H, use_sparse = False, k = 1, verbose=False):
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

def construct_rescaled_hamiltonian(N, M, V, mu_V_ratio, J_V_ratio):
    """
    Constructs a rescaled Hamiltonian matrix for a quantum system with N sites and M states per site, 
    incorporating chemical potential, tunneling, and interaction terms. The Hamiltonian is normalized 
    by the absolute value of V to produce H_tilde. Uses open boundary conditions.

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

def simulate_time_evolution(N, M, V, mu_V_ratio_routine, J_V_ratio_routine, times, dt, initial_state=None):
    """
    Simulates the unitary time evolution of a quantum system by evolving a wavefunction under a time-dependent Hamiltonian,
    governed by J/V(t) and mu/V(t). Tracks the evolution of wavefunctions, energies, overlaps, and probabilities throughout 
    the process.

    Parameters:
    N (int): Number of sites in the system.
    M (int): Number of states per site.
    V (float): Interaction strength.
    mu_V_ratio_routine (list or np.ndarray): Time-dependent values of the chemical potential ratio (mu/V).
    J_V_ratio_routine (list or np.ndarray): Time-dependent values of the tunneling ratio (J/V).
    times (list or np.ndarray): Discrete time steps over which the evolution is simulated.
    dt (float, optional): Time step size for evolution.
    initial_state (np.ndarray, optional): Initial wavefunction as a column vector. If None, uses the ground state of the initial Hamiltonian.

    Returns:
    tuple: A tuple containing:
        - energies (list): Energies of the evolved state at each time step.
        - energy_diff (np.ndarray): Difference between time evolved energies and the ground state energy at each time step.
        - time_evolved_wavefunctions (list): Wavefunctions evolved over the simulation.
        - state_probabilities (np.ndarray): Probabilities of projection onto each eigenstate at each time step.
        - state_overlaps (np.ndarray): Overlaps of the evolved state with each instantaneous eigenstate at each time step.
        - true_energies (np.ndarray): Eigenvalues of the instantaneous Hamiltonian at each time step.
        - energy_gaps (np.ndarray): Energy gaps between each eigenvalue and the ground state energy at each time step.
    """
    
    n_excited_states = M**N
    initial_hamiltonian = construct_rescaled_hamiltonian(N, M, V, mu_V_ratio_routine[0], J_V_ratio_routine[0])
    
    if initial_state is None:
        _, eigenvectors_0 = exact_diagonalize(initial_hamiltonian)
        psi_0 = eigenvectors_0[0]
    else:
        psi_0 = initial_state

    energies = []
    time_evolved_wavefunctions = []
    state_probabilities = []
    state_overlaps = []
    true_energies = []

    psi = psi_0.copy()
    for index, t in enumerate(times):
        instantaneous_hamiltonian = construct_rescaled_hamiltonian(N, M, V, mu_V_ratio_routine[index], J_V_ratio_routine[index])
        
        eigenvalues, eigenvectors = exact_diagonalize(instantaneous_hamiltonian)
        true_energies.append(eigenvalues)
        
        psi = evolve_wavefunction(psi, instantaneous_hamiltonian, dt)
        psi = psi / np.linalg.norm(psi)  
        
        time_evolved_wavefunctions.append(psi)
        
        energy = np.real(np.conj(psi).T @ instantaneous_hamiltonian @ psi)
        energies.append(energy)
        
        overlap = [np.dot(np.conj(eigenvectors[i]).T, psi) for i in range(n_excited_states)] 
        probability = [np.abs(np.conj(eigenvectors[i]).T @ psi)**2 for i in range(n_excited_states)]
               
        state_probabilities.append(probability)
        state_overlaps.append(overlap)

    energies = np.array(energies)
    time_evolved_wavefunctions = np.array(time_evolved_wavefunctions)
    state_probabilities = np.array(state_probabilities)
    state_overlaps = np.array(state_overlaps)
    true_energies = np.array(true_energies)
        
    return energies, time_evolved_wavefunctions, state_probabilities, state_overlaps, true_energies
# --------------------------------------------------------------------------------------------------------------------------------------------

def plot_time_evolution(N, M, results, times, J_V_ratio_routine, mu_V_ratio_routine, time_array = None, plot_probability = True, plot_gap = True, plot_overlaps = True, plot_sigma = True):
    # to do: add documentation
    # tl;dr takes results from simulate_time_evolution() and and makes plots
    
    energies, time_evolved_wavefunctions, state_probabilities, state_overlaps, true_energies = results
    energies = energies * 1/N
    true_energies = true_energies * 1/N
    colors = get_cmap("gist_rainbow", M**N)
    
    if plot_probability == True:
        fig, ax = plt.subplots()
        for index in range(M**N):
            if index == 0:
                ax.plot(times, state_probabilities[:,index], color = "k", label = "Ground State")
            elif index == 1:
                ax.plot(times, state_probabilities[:,index], color = colors(index), label = "1st Excited State")
            elif index == 2:
                ax.plot(times, state_probabilities[:,index], color = colors(index), label = "2nd Excited State") 
            else: 
                ax.plot(times, state_probabilities[:,index], color = colors(index))
        ax.set_ylim(-0.1,1.1)
        ax.legend(loc = "center left")
        ax.set_title(f"State Probabilities: $N={N}$, $M={M}$, $V<0$, $(J/|V|)_f = {J_V_ratio_routine[-1]}$")
        ax.set_xlabel("Time [$t/|V|$]")
        ax.set_ylabel("State Probability")
        ax.grid()
        fig.tight_layout()
        if time_array is not None:
            accumulated_time = 0
            for time in time_array:
                ax.axvline(accumulated_time, color = "k", linestyle = "--")
                accumulated_time += time
    
    if plot_gap == True:
        fig, ax = plt.subplots()
        for index in range(M**N):
            ax.plot(times, true_energies[:,index]-true_energies[:,0], color = colors(index))   
        ax.plot(times, energies-true_energies[:,0], color = "k", label = "Time Evolved State")
        ax.legend(loc = "upper center")
        ax.set_title(f"Scaled Energy Gap: $N={N}$, $M={M}$, $V<0$, $(J/|V|)_f = {J_V_ratio_routine[-1]}$")
        ax.set_xlabel("Time [$t/|V|$]")
        ax.set_ylabel("Scaled Energy [$E/N|V| = \epsilon/|V|$]")
        fig.tight_layout()
        if time_array is not None:
            accumulated_time = 0
            for time in time_array:
                ax.axvline(accumulated_time, color = "k", linestyle = "--")
                accumulated_time += time

    if plot_overlaps == True:
        fig, (ax1,ax2) = plt.subplots(nrows = 2, sharex=True)
        for index in range(M**N):
            if index == 0:
                ax1.plot(times, np.real(state_overlaps[:,0]), '.', color = "k")
                ax2.plot(times, np.imag(state_overlaps[:,0]), '.', color = "k")
            else:    
                ax1.plot(times, np.real(state_overlaps[:,index]), '.', color = colors(index))
                ax2.plot(times, np.imag(state_overlaps[:,index]), '.', color = colors(index))
            ax1.set_ylabel("$\Re$ Component")
            ax2.set_ylabel("$\Im$ Component")
        ax2.set_xlabel("Time [$t/|V|$]")
        fig.suptitle(f"State Overlap: $N={N}$, $M={M}$, $V<0$, $(J/|V|)_f = {J_V_ratio_routine[-1]}$")
        fig.tight_layout()
        if time_array is not None:
            accumulated_time = 0
            for time in time_array:
                ax.axvline(accumulated_time, color = "k", linestyle = "--")
                accumulated_time += time

    if plot_sigma == True:
        fig, ax = plt.subplots()
        states, _ = enumerate_states(N, M)
        sigmas = []
        for wavefunction in time_evolved_wavefunctions:
            sigmas += [sigma_ij(0, 1, ground_state_wavefunction = wavefunction, states = states, N=N, M=M)/M]
        ax.plot(times, sigmas, "-k")
        ax.set_title(f"Time Evolved $\sigma$: $N={N}$, $M={M}$, $V<0$, $(J/|V|)_f = {J_V_ratio_routine[-1]}$")
        ax.set_ylabel("$\sigma^{01}/M$")
        ax.set_xlabel("Time [$t/|V|$]")
        ax.grid()
        if time_array is not None:
            accumulated_time = 0
            for time in time_array:
                ax1.axvline(accumulated_time, color = "k", linestyle = "--")
                ax2.axvline(accumulated_time, color = "k", linestyle = "--")
                accumulated_time += time

# --------------------------------------------------------------------------------------------------------------------------------------------

def make_linear_stepped_routines(J_V_ratios, mu_V_ratios, time_array, dt):
    # to do: documentation
    # tl;dr makes linear stepped routine (start here, end here for each step linearly)
    
    if len(J_V_ratios) != len(mu_V_ratios) or len(J_V_ratios) != len(time_array):
        raise ValueError("The length of J_V_ratios, mu_V_ratios, and time_array must be equal.")
    
    num_steps = len(time_array)
    
    times = []
    start_time = 0
    for t in time_array:
    
        time_array = np.linspace(start_time, start_time + t, num = int(t / dt))
        times.append(time_array)
        start_time += t
        
    concatenated_times = np.concatenate(times)

    J_V_ratio_steps = [np.linspace(J_V_ratios[i][0], J_V_ratios[i][1], int(time_array[i] / dt)) for i in range(num_steps)]
    mu_V_ratio_steps = [np.linspace(mu_V_ratios[i][0], mu_V_ratios[i][1], int(time_array[i] / dt)) for i in range(num_steps)]

    J_V_ratio_routine = np.concatenate(J_V_ratio_steps)
    mu_V_ratio_routine = np.concatenate(mu_V_ratio_steps)

    return concatenated_times, J_V_ratio_routine, mu_V_ratio_routine

# --------------------------------------------------------------------------------------------------------------------------------------------