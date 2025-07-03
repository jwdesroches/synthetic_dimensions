# --------------------------------------------------------------------------------------------------------------------------------------------
# imports
# --------------------------------------------------------------------------------------------------------------------------------------------

import numpy as np
from scipy.sparse.linalg import eigsh
from scipy.linalg import expm
from scipy.linalg import eigh
from scipy.optimize import minimize
import sys
import os

parent_dir = os.path.abspath(os.path.join(os.getcwd(), os.pardir))
data_folder_path = os.path.abspath(os.path.join(os.getcwd(), os.pardir, "data_folder"))
sys.path.append(parent_dir)
sys.path.append(data_folder_path) 

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

def old_exact_diagonalize(H, use_sparse = False, k = 1):
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

        return eigenvalues, [eigenvectors[:, i] for i in range(H.shape[0])]
    
# --------------------------------------------------------------------------------------------------------------------------------------------

def exact_diagonalize(H, use_sparse = False, k = 1):
    """
    Diagonalizes a Hermitian matrix using scipy's `eigh()` method. Faster than numpy's `eigh()` method.

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
        eigenvalues, eigenvectors = eigh(H)
        return eigenvalues, [eigenvectors[:, i] for i in range(H.shape[0])]
    
# --------------------------------------------------------------------------------------------------------------------------------------------

def old_construct_hamiltonian(N, M, J, V):
    """
    Constructs the J-V Hamiltonian matrix.

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

def sigma_ij(i, j, wavefunction, states, N, M):
    """
    Computes the sigma value, representing the difference in synthetic dimension space between sites i and j, 
    weighted by the wavefunction. Serves as an order parameter for the quantum string-gas transition.

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
                        sigma += abs(m - n) * np.linalg.norm(wavefunction[k])**2
                    else:
                        sigma += 0
                else:
                    sigma += 0
    
    return sigma

# --------------------------------------------------------------------------------------------------------------------------------------------

def sigma_ij_operator(i, j, states, N, M):
    """placeholder definition"""
    dim = M**N 
    Sigma_ij = np.zeros((dim, dim))

    for k in range(dim):
        for m in range(M):
            for n in range(M):
                if states[k][i] == m and states[k][j] == n:
                    Sigma_ij[k, k] += abs(m - n)  

    return Sigma_ij

# --------------------------------------------------------------------------------------------------------------------------------------------

def calculate_specific_heat(beta, energy_eigenvalues):
    """placeholder definition"""
    E0 = np.min(energy_eigenvalues)
    E_shifted = energy_eigenvalues - E0

    exp_factors = np.exp(-beta * E_shifted)
    Z_tilde = np.sum(exp_factors)

    avg_E_shifted = np.sum(E_shifted * exp_factors) / Z_tilde
    avg_E2_shifted = np.sum(E_shifted**2 * exp_factors) / Z_tilde

    avg_E = E0 + avg_E_shifted
    avg_E2 = E0**2 + 2 * E0 * avg_E_shifted + avg_E2_shifted

    Cv = (avg_E2 - avg_E**2) * beta**2
    return Cv

# --------------------------------------------------------------------------------------------------------------------------------------------

def calculate_average_energy(beta, energy_eigenvalues):
    """placeholder definition"""
    E0 = np.min(energy_eigenvalues)
    E_shifted = energy_eigenvalues - E0

    exp_factors = np.exp(-beta * E_shifted)
    Z_tilde = np.sum(exp_factors)

    avg_E_shifted = np.sum(E_shifted * exp_factors) / Z_tilde
    avg_E = E0 + avg_E_shifted
    
    return avg_E

# --------------------------------------------------------------------------------------------------------------------------------------------

def calculate_finite_temperature_expectation_value(operator, beta, energy_eigenvalues, energy_eigenstates):
    """placeholder definition"""    
    energy_shift = np.min(energy_eigenvalues)
    shifted_energies = energy_eigenvalues - energy_shift

    weights = np.exp(-beta * shifted_energies)
    normalization_factor = np.sum(weights)
    
    expectation_value = 0
    for i, psi in enumerate(energy_eigenstates):
        Ai = psi.T.conj() @ operator @ psi
        expectation_value += weights[i] * Ai

    return expectation_value / normalization_factor

# --------------------------------------------------------------------------------------------------------------------------------------------

def construct_ground_state_manifold(eigenvalues, eigenvectors, epsilon = 1e-9):
    """Placeholder definition."""
    ground_state_energy = eigenvalues[0]
    ground_state_manifold = []
    for i in range(len(eigenvalues)):
        if ground_state_energy - epsilon <= eigenvalues[i] <= ground_state_energy + epsilon:
            ground_state_manifold += [eigenvectors[i]]
    return ground_state_manifold

# --------------------------------------------------------------------------------------------------------------------------------------------

def calculate_ground_state_manifold_overlap(state, ground_state_manifold):
    """Placeholder definition."""
    projector = np.zeros((len(ground_state_manifold[0]), len(ground_state_manifold[0])), dtype=complex)
    for psi in ground_state_manifold:
        psi = psi.reshape(-1, 1)    
        projector += psi @ psi.T.conj()

    state = state.reshape(-1,1)
    ground_state_manifold_overlap = (state.T.conj() @ projector @ state)[0][0]
    
    if np.imag(ground_state_manifold_overlap) > 1e-9:
        print("Imaginary component for ground state manifold overlap is non-zero! Check code!")
    else:
        ground_state_manifold_overlap = np.real(ground_state_manifold_overlap)

    return ground_state_manifold_overlap

# --------------------------------------------------------------------------------------------------------------------------------------------

def construct_hamiltonian(N, M, V, mu, J, theta = 0, boundary_conditions = "OBC", chemical_potential_loc = 0):
    """Placeholder definition."""
    dim = M**N
    H = np.zeros((dim, dim), dtype=np.complex128)

    # Precompute powers of M for faster state-to-index conversion
    M_powers = np.array([M**i for i in range(N)])

    def index_to_state(index):
        return np.array([(index // M_powers[i]) % M for i in range(N-1, -1, -1)])
    
    # Helper function to convert a state representation (array of states) back to an index
    def state_to_index(state):
        return np.dot(state, M_powers[::-1])

    # Apply the chemical potential term
    for alpha in range(dim):
        state = index_to_state(alpha)
        for j in range(N):
            if state[j] == chemical_potential_loc:
                H[alpha, alpha] -= mu
                    
    # Apply the tunneling term
    for alpha in range(dim):
        state = index_to_state(alpha)
        for j in range(N):
            for n in range(M):
                if state[j] == n:
                    if n == 0:
                        if boundary_conditions == "PBC":
                            new_state = state.copy()
                            new_state[j] = M - 1
                            beta = state_to_index(new_state)
                            H[alpha, beta] -= J
                            H[beta, alpha] -= J
                        elif boundary_conditions == "OBC":
                            pass
                        
                    else:
                        new_state = state.copy()
                        new_state[j] = n - 1
                        beta = state_to_index(new_state)
                        
                        if n == 1:
                            H[alpha, beta] -= J*np.exp(1j*theta)
                            H[beta, alpha] -= J*np.exp(-1j*theta)
                            
                        else:
                            H[alpha, beta] -= J
                            H[beta, alpha] -= J  

    # Apply the interaction term
    for alpha in range(dim):
        state = index_to_state(alpha)
        for i in range(N - 1):
            j = i + 1
            for n in range(M):
                if n == 0:
                    if boundary_conditions == "PBC":
                        if state[i] == 0 and state[j] == M - 1:
                            new_state = state.copy()
                            new_state[i], new_state[j] = M - 1, 0
                            beta = state_to_index(new_state)
                            H[alpha, beta] += V
                            H[beta, alpha] += V  
                    elif boundary_conditions == "OBC":
                        pass
                else:
                    if state[i] == n and state[j] == n - 1:
                        new_state = state.copy()
                        new_state[i], new_state[j] = n - 1, n
                        beta = state_to_index(new_state)
                        H[alpha, beta] += V
                        H[beta, alpha] += V  
        
    return H

# ------------------------------------------------------------------------------------------------------------------------------------------

def construct_rescaled_hamiltonian(N, M, V, mu_V_ratio, J_V_ratio, theta = 0, boundary_conditions = "OBC", chemical_potential_loc = 0):
    """Placeholder definition."""
    mu = mu_V_ratio * abs(V)
    J = J_V_ratio * abs(V)

    H = construct_hamiltonian(N, M, V, mu, J, theta, boundary_conditions, chemical_potential_loc)
    H_tilde = H/np.abs(V)
    return H_tilde

# ------------------------------------------------------------------------------------------------------------------------------------------

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

def simulate_hamiltonian_time_evolution(hamiltonians, times, initial_state=None):
    """
    Simulates the time evolution of a quantum state under a series of time-dependent Hamiltonians.
    
    The function evolves an initial state (by default, the ground state of the first Hamiltonian) 
    through a sequence of Hamiltonians defined at discrete times. At each time step, the wavefunction 
    is evolved using the unitary time evolution operator (via the `evolve_wavefunction` function). 
    Additionally, the function records the expectation energy, the evolved wavefunctions, the probabilities 
    for each instantaneous eigenstate, the complex overlaps with those eigenstates, and the eigenenergies 
    (true energies) of the Hamiltonians.
    
    Parameters:
    hamiltonians (list or array-like): A sequence of Hamiltonian matrices representing the system at each time step.
    times (array-like): Array of time values corresponding to each Hamiltonian.
    initial_state (np.ndarray, optional): The initial quantum state as a column vector. If not provided, 
                                            the function uses the ground state of the first Hamiltonian.
    
    Returns:
    energies (np.ndarray): Array of expectation values of the energy at each time step.
    time_evolved_wavefunctions (np.ndarray): Array of the wavefunctions after evolution at each time step.
    state_probabilities (np.ndarray): Array containing the probability of the evolved state projecting onto 
                                      each instantaneous eigenstate at every time step.
    state_overlaps (np.ndarray): Array of complex overlaps between the evolved state and the instantaneous eigenstates.
    true_energies (np.ndarray): Array of eigenenergies of the instantaneous Hamiltonians at each time step.
    """
    
    n_excited_states = len(hamiltonians[0])
    initial_hamiltonian = hamiltonians[0]
    
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
    ground_state_manifold_overlaps = []

    psi = psi_0.copy()
    for idx, instantaneous_hamiltonian in enumerate(hamiltonians):
        
        if idx > 1:
            dt = times[idx] - times[idx - 1]
        else:
            dt = times[idx]
            
        eigenvalues, eigenvectors = exact_diagonalize(instantaneous_hamiltonian)
        ground_state_manifold = construct_ground_state_manifold(eigenvalues, eigenvectors)
        true_energies.append(eigenvalues) 
                
        psi = evolve_wavefunction(psi, instantaneous_hamiltonian, dt)
        psi = psi / np.linalg.norm(psi)  
        
        time_evolved_wavefunctions.append(psi)

        energy = np.real(np.conj(psi).T @ instantaneous_hamiltonian @ psi)
        energies.append(energy)
        
        ground_state_manifold_overlap = calculate_ground_state_manifold_overlap(psi, ground_state_manifold)
        overlap = [np.dot(np.conj(eigenvectors[i]).T, psi) for i in range(n_excited_states)] 
        probability = [np.abs(overlap[i])**2 for i in range(n_excited_states)]
        
        ground_state_manifold_overlaps.append(ground_state_manifold_overlap)
        state_probabilities.append(probability)
        state_overlaps.append(overlap)

    energies = np.array(energies)
    time_evolved_wavefunctions = np.array(time_evolved_wavefunctions)
    state_probabilities = np.array(state_probabilities)
    state_overlaps = np.array(state_overlaps)
    true_energies = np.array(true_energies)
    ground_state_manifold_overlaps = np.array(ground_state_manifold_overlaps)
        
    return energies, time_evolved_wavefunctions, state_probabilities, state_overlaps, true_energies, ground_state_manifold_overlaps


# --------------------------------------------------------------------------------------------------------------------------------------------

def old_create_optimal_piecewise_linear_paths(N, M, T, dt, V, J_V_init, J_V_final, mu_V_init, mu_V_final, num_control_points, alpha = 2, initial_state = None, chemical_potential_loc = 0):
    """
    Constructs optimized piecewise linear paths for the control parameters J/|V| and μ/|V|.
    
    This function sets up and solves an optimization problem in which both the intermediate control values 
    and the corresponding times are optimized. A dense time grid is generated for evaluation, and the objective 
    function includes penalties for rapid (non-adiabatic) changes, negative μ/|V| values, and lack of smoothness.
    A non-linear (quadratic) initial guess for the intermediate times is used to bias the evolution towards 
    spending more time near the end, where μ/|V| is small.
    
    Parameters:
    N (int): Number of particles or spins in the system.
    M (int): Local Hilbert space dimension.
    T (float): Total evolution time.
    dt (float): Time step for the dense evaluation grid.
    V (float): Interaction strength or scaling parameter in the Hamiltonian.
    J_V_init (float): Initial value of the ratio J/|V|.
    J_V_final (float): Final value of the ratio J/|V|.
    mu_V_init (float): Initial value of the ratio μ/|V|.
    mu_V_final (float): Final value of the ratio μ/|V|.
    num_control_points (int): Total number of control points (including the fixed endpoints).
    
    Returns:
    times_dense (np.ndarray): Dense time grid used for evaluation.
    J_V_path (np.ndarray): Optimized J/|V| path evaluated on the dense time grid.
    mu_V_path (np.ndarray): Optimized μ/|V| path evaluated on the dense time grid.
    obj_value (float): Final objective value from the optimization.
    opt_params (np.ndarray): Optimized parameter vector containing intermediate control values and times.
    t_control_opt (np.ndarray): Full set of optimized control times, including the endpoints.
    J_control_opt (np.ndarray): Optimized control values for J/|V| including the endpoints.
    mu_control_opt (np.ndarray): Optimized control values for μ/|V| including the endpoints.
    """
    
    # Dense time grid for evaluation.
    times_dense = np.arange(0, T + dt, dt)
    
    # Number of control points and free (intermediate) points.
    n_points = num_control_points
    n_int = n_points - 2

    # Initial guesses for the control values (linearly spaced between endpoints).
    J_initial_guess = np.linspace(J_V_init, J_V_final, n_points)[1:-1]
    mu_initial_guess = np.linspace(mu_V_init, mu_V_final, n_points)[1:-1]
    
    # Non-linear initial guess for intermediate times: more time allocated toward the end.
    t_initial_guess = T * (np.linspace(0, 1, n_points)[1:-1] ** alpha)

    # Combine intermediate control values and times into one vector.
    x0 = np.concatenate((J_initial_guess, mu_initial_guess, t_initial_guess))
    
    # A small buffer to ensure strict ordering of times.
    eps = 1e-3

    # Constraints for the intermediate times: they must be strictly between 0 and T and in ascending order.
    cons = []
    cons.append({'type': 'ineq', 'fun': lambda x: x[2*n_int] - eps})
    for i in range(1, n_int):
        cons.append({
            'type': 'ineq',
            'fun': lambda x, i=i: x[2*n_int + i] - x[2*n_int + i - 1] - eps
        })
    cons.append({'type': 'ineq', 'fun': lambda x: T - x[2*n_int + n_int - 1] - eps})
    
    # Weights for the penalty terms.
    lambda_adiabatic = 0.25       
    lambda_smooth_J = 0.1 
    lambda_smooth_mu = 0.1
    
    def objective(x):
        # Unpack the optimization vector.
        J_int = x[:n_int]
        mu_int = x[n_int:2*n_int]
        t_int = x[2*n_int:3*n_int]
        
        # Reconstruct full control arrays including fixed endpoints.
        J_control = np.concatenate(([J_V_init], J_int, [J_V_final]))
        mu_control = np.concatenate(([mu_V_init], mu_int, [mu_V_final]))
        t_control = np.concatenate(([0.0], t_int, [T]))
        
        # Build dense paths using linear interpolation.
        J_path_dense = np.interp(times_dense, t_control, J_control)
        mu_path_dense = np.interp(times_dense, t_control, mu_control)
        
        # Penalty for any negative μ/|V| values.
        penalty = np.sum(np.abs(np.minimum(0, mu_path_dense)))
        
        # Smoothness penalty using discrete second differences.
        smoothness_penalty_J = lambda_smooth_J * np.sum(np.diff(J_control, 2)**2)
        smoothness_penalty_mu = lambda_smooth_mu * np.sum(np.diff(mu_control, 2)**2)
        smoothness_penalty = smoothness_penalty_J + smoothness_penalty_mu
        
        # Construct Hamiltonians at each point in the dense time grid.
        hamiltonians = []
        for i, t in enumerate(times_dense):
            ham = construct_rescaled_hamiltonian(N, M, V, mu_V_ratio=mu_path_dense[i], J_V_ratio=J_path_dense[i], chemical_potential_loc = chemical_potential_loc)
            hamiltonians.append(ham)
        
        # Adiabaticity penalty: discourages rapid changes in the Hamiltonian.
        adiabatic_penalty = 0.0
        for i in range(len(times_dense) - 1):
            # Finite difference approximation for the derivative of the control parameters.
            dJ = J_path_dense[i+1] - J_path_dense[i]
            dmu = mu_path_dense[i+1] - mu_path_dense[i]
            dH_norm = np.sqrt(dJ**2 + dmu**2)
            # Compute the energy gap using the instantaneous Hamiltonian.
            energies, _ = exact_diagonalize(hamiltonians[i])
            gap = energies[1] - energies[0]
            adiabatic_penalty += (dH_norm**2 / gap**2) * (times_dense[i+1] - times_dense[i])
        adiabatic_penalty *= lambda_adiabatic
        
        # Simulate the time evolution and compute the ground state infidelity.
        _, _, _, _, _, calculate_ground_state_manifold_overlaps = simulate_hamiltonian_time_evolution(hamiltonians, times_dense, initial_state = initial_state)
        ground_state_fidelity = calculate_ground_state_manifold_overlaps[-1]
        ground_state_infidelity = 1 - ground_state_fidelity
        
        return ground_state_infidelity + penalty + smoothness_penalty + adiabatic_penalty

    # Optimize the control parameters using SLSQP to enforce the constraints.
    result = minimize(objective, x0, method='SLSQP', constraints=cons)
    opt_params = result.x

    # Extract the optimized intermediate values.
    J_int_opt = opt_params[:n_int]
    mu_int_opt = opt_params[n_int:2*n_int]
    t_int_opt = opt_params[2*n_int:3*n_int]

    # Construct full control arrays including endpoints.
    J_control_opt = np.concatenate(([J_V_init], J_int_opt, [J_V_final]))
    mu_control_opt = np.concatenate(([mu_V_init], mu_int_opt, [mu_V_final]))
    t_control_opt = np.concatenate(([0.0], t_int_opt, [T]))

    # Generate the optimized dense paths.
    J_V_path = np.interp(times_dense, t_control_opt, J_control_opt)
    mu_V_path = np.interp(times_dense, t_control_opt, mu_control_opt)
    
    obj_value = result.fun
    return (times_dense, J_V_path, mu_V_path, obj_value, opt_params, t_control_opt, J_control_opt, mu_control_opt)

# --------------------------------------------------------------------------------------------------------------------------------------------

def create_optimal_piecewise_linear_paths(N, M, T, dt, V, J_V_init, J_V_final, mu_V_init, mu_V_final, num_control_points, initial_guess=None):

    times_dense = np.arange(0, T + dt, dt)
    n_points = num_control_points
    n_int = n_points - 2

    # If initial_guess provided, use it; else use linear interpolation
    if initial_guess is not None:
        x0 = initial_guess
        # Optionally, validate length of initial_guess matches expected
        expected_len = 3 * n_int
        if len(x0) != expected_len:
            raise ValueError(f"initial_guess length {len(x0)} does not match expected {expected_len}")
    else:
        J_initial_guess = np.linspace(J_V_init, J_V_final, n_points)[1:-1]
        mu_initial_guess = np.linspace(mu_V_init, mu_V_final, n_points)[1:-1]
        t_initial_guess = T * np.linspace(0, 1, n_points)[1:-1]
        x0 = np.concatenate((J_initial_guess, mu_initial_guess, t_initial_guess))

    eps = 1e-3
    cons = []
    cons.append({'type': 'ineq', 'fun': lambda x: x[2*n_int] - eps})
    for i in range(1, n_int):
        cons.append({'type': 'ineq', 'fun': lambda x, i=i: x[2*n_int + i] - x[2*n_int + i - 1] - eps})  
    cons.append({'type': 'ineq', 'fun': lambda x: T - x[2*n_int + n_int - 1] - eps}) 

    # mu must decrease monotonically
    for i in range(1, n_int):
        cons.append({'type': 'ineq', 'fun': lambda x, i=i: x[n_int + i - 1] - x[n_int + i] - eps})

    loop_weight = 0.1

    def objective(x):
        J_int = x[:n_int]
        mu_int = x[n_int:2*n_int]
        t_int = x[2*n_int:3*n_int]

        J_control = np.concatenate(([J_V_init], J_int, [J_V_final]))
        mu_control = np.concatenate(([mu_V_init], mu_int, [mu_V_final]))
        t_control = np.concatenate(([0.0], t_int, [T]))

        J_path_dense = np.interp(times_dense, t_control, J_control)
        mu_path_dense = np.interp(times_dense, t_control, mu_control)

        negative_mu_penalty = np.sum(np.abs(np.minimum(0, mu_path_dense)))
        negative_J_penalty = np.sum(np.abs(np.minimum(0, J_path_dense)))

        delta_Js = np.diff(J_control)
        delta_mus = np.diff(mu_control)
        path_length = np.sum(np.sqrt(delta_Js**2 + delta_mus**2))
        straight_line_distance = np.sqrt((J_V_final - J_V_init)**2 + (mu_V_final - mu_V_init)**2)
        loop_penalty = loop_weight * (path_length - straight_line_distance)

        hamiltonians = []
        for i, t in enumerate(times_dense):
            mu = np.abs(V) * mu_path_dense[i]
            J = np.abs(V) * J_path_dense[i]
            ham = construct_hamiltonian(N, M, V, mu, J)
            hamiltonians.append(ham)

        _, _, _, _, _, calculate_ground_state_manifold_overlaps = simulate_hamiltonian_time_evolution(hamiltonians, times_dense)
        ground_state_fidelity = calculate_ground_state_manifold_overlaps[-1]
        ground_state_infidelity = 1 - ground_state_fidelity

        return ground_state_infidelity + negative_J_penalty + negative_mu_penalty + loop_penalty

    result = minimize(objective, x0, method='SLSQP', constraints=cons, options={'maxiter': 1000, 'ftol': 1e-9, 'disp': True})

    #print(result.message)
    print("Success:", result.success)

    opt_params = result.x
    J_int_opt = opt_params[:n_int]
    mu_int_opt = opt_params[n_int:2*n_int]
    t_int_opt = opt_params[2*n_int:3*n_int]

    J_control_opt = np.concatenate(([J_V_init], J_int_opt, [J_V_final]))
    mu_control_opt = np.concatenate(([mu_V_init], mu_int_opt, [mu_V_final]))
    t_control_opt = np.concatenate(([0.0], t_int_opt, [T]))

    J_V_path = np.interp(times_dense, t_control_opt, J_control_opt)
    mu_V_path = np.interp(times_dense, t_control_opt, mu_control_opt)

    obj_value = result.fun
    return (times_dense, J_V_path, mu_V_path, obj_value, opt_params, t_control_opt, J_control_opt, mu_control_opt)

# --------------------------------------------------------------------------------------------------------------------------------------------