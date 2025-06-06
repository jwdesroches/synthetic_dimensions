# --------------------------------------------------------------------------------------------------------------------------------------------
# imports
# --------------------------------------------------------------------------------------------------------------------------------------------

import numpy as np
from scipy.sparse.linalg import eigsh
from scipy.linalg import expm
from scipy.optimize import minimize

# --------------------------------------------------------------------------------------------------------------------------------------------
# functions
# --------------------------------------------------------------------------------------------------------------------------------------------

def exact_diagonalize(H, use_sparse = False, k = 1):
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
    
# ------------------------------------------------------------------------------------------------------------------------------------------

def construct_rescaled_hamiltonian(N, M, V, mu_V_ratio, J_V_ratio, theta = 0, boundary_conditions = "OBC"):
    """
    Constructs a rescaled J-V-mu Hamiltonian matrix with N sites and M states per site, incorporating 
    chemical potential, tunneling, and interaction terms. The Hamiltonian is normalized 
    by the absolute value of V to produce H_tilde. Can specify either OBC or PBC. 

    Parameters:
    N (int): Number of sites in the system.
    M (int): Number of states per site.
    V (float): Interaction strength.
    mu_V_ratio (float): Ratio of the chemical potential (mu) to the interaction strength (V).
    J_V_ratio (float): Ratio of the tunneling parameter (J) to the interaction strength (V).
    theta (float): The phase to apply to the tunneling term between n = 0 and n = 1. Defaults to 0.
    boundary_conditions (string, optional): Which boundary conditions to use. Either "OBC" for 
                                            open boundary conditions or PBC for periodic boundary 
                                            conditions. Defaults to OBC.

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
                    
    # Rescale H to H_tilde by dividing by |V|
    H_tilde = H / abs(V)
        
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

def construct_ground_state_manifold(eigenvalues, eigenvectors, epsilon = 1e-9):
    ground_state_energy = eigenvalues[0]
    ground_state_manifold = []
    for i in range(len(eigenvalues)):
        if ground_state_energy - epsilon <= eigenvalues[i] <= ground_state_energy + epsilon:
            ground_state_manifold += [eigenvectors[i]]
    return ground_state_manifold

# --------------------------------------------------------------------------------------------------------------------------------------------

def calculate_ground_state_manifold_overlap(state, ground_state_manifold):
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

def new_create_optimal_piecewise_linear_paths(N, M, T, dt, V, J_V_init, J_V_final, mu_V_init, mu_V_final, num_control_points, initial_guess=None):
    import numpy as np
    from scipy.optimize import minimize

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
    cons.append({'type': 'ineq', 'fun': lambda x: x[2*n_int] - eps})  # t0 > 0
    for i in range(1, n_int):
        cons.append({'type': 'ineq', 'fun': lambda x, i=i: x[2*n_int + i] - x[2*n_int + i - 1] - eps})  # ascending times
    cons.append({'type': 'ineq', 'fun': lambda x: T - x[2*n_int + n_int - 1] - eps})  # last time < T

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
            ham = construct_rescaled_hamiltonian(N, M, V,
                                                 mu_V_ratio=mu_path_dense[i],
                                                 J_V_ratio=J_path_dense[i])
            hamiltonians.append(ham)

        _, _, _, _, _, calculate_ground_state_manifold_overlaps = simulate_hamiltonian_time_evolution(hamiltonians, times_dense)
        ground_state_fidelity = calculate_ground_state_manifold_overlaps[-1]
        ground_state_infidelity = 1 - ground_state_fidelity

        return ground_state_infidelity + negative_J_penalty + negative_mu_penalty + loop_penalty

    result = minimize(
        objective,
        x0,
        method='SLSQP',
        constraints=cons,
        options={
            'maxiter': 1000,
            'ftol': 1e-9,
            'disp': True
        }
    )

    print(result.message)
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