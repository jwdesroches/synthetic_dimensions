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