# --------------------------------------------------------------------------------------------------------------------------------------------
# imports
# --------------------------------------------------------------------------------------------------------------------------------------------

import numpy as np
import matplotlib.pyplot as plt

# --------------------------------------------------------------------------------------------------------------------------------------------
# functions
# --------------------------------------------------------------------------------------------------------------------------------------------

def enumerate_states(N, M):
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

def initialize_hamiltonian(N, M):
    """
    Initialize a zero matrix for the Hamiltonian of size M^N x M^N.
    
    Inputs:
    N (int): Number of lattice sites.
    M (int): Number of synthetic levels.
    
    Returns:
    H (np.ndarray): Initialized Hamiltonian matrix of size M^N x M^N.
    """
    
    num_states = M**N
    H = np.zeros((num_states, num_states), dtype=np.complex128)
    return H

# --------------------------------------------------------------------------------------------------------------------------------------------

def get_new_state(state, site, new_level):
    """
    Generate a new state by changing the level of a particle at a given lattice site.
    
    Inputs:
    state (list): Current state configuration represented as a list of integers.
    site (int): Index of the lattice site to change.
    new_level (int): New synthetic level for the lattice site.
    
    Returns:
    new_state (list): New state configuration after modifying the specified lattice site.
    """
    
    new_state = state.copy()
    new_state[site] = new_level
    return new_state

# --------------------------------------------------------------------------------------------------------------------------------------------

def get_double_new_state(state, site1, level1, site2, level2):
    """
    Generate a new state by changing the levels of particles at two lattice sites.
    
    Inputs:
    state (list): Current state configuration represented as a list of integers.
    site1 (int): Index of the first lattice site to change.
    level1 (int): New synthetic level for the first lattice site.
    site2 (int): Index of the second lattice site to change.
    level2 (int): New synthetic level for the second lattice site.
    
    Returns:
    new_state (list): New state configuration after modifying the specified lattice sites.
    """
    
    new_state = state.copy()
    new_state[site1] = level1
    new_state[site2] = level2
    return new_state

# --------------------------------------------------------------------------------------------------------------------------------------------

def state_index(state, all_states):
    """
    Get the index of a given state in the list of all states.
    
    Inputs:
    state (list): State whose index is to be found.
    all_states (list): List of all states.
    
    Returns:
    index (int): Index of the state in the list of all_states.
    """
    
    return all_states.index(state)

# --------------------------------------------------------------------------------------------------------------------------------------------

def old_construct_hamiltonian(N, M, J, V):
    """
    Construct the Hamiltonian matrix including tunneling and interaction terms.
    
    Inputs:
    N (int): Number of lattice sites.
    M (int): Number of synthetic levels.
    t (float): Tunneling parameter.
    V (float): Interaction strength.
    
    Returns:
    H (np.ndarray): Constructed Hamiltonian matrix of size M^N x M^N.
    formatted_states (list): List of formatted state strings for easy visualization.
    """
    
    states, formatted_states = enumerate_states(N, M)
    H = initialize_hamiltonian(N, M)
    
    for state in states:
        current_index = state_index(state, states)
        
        for i in range(N):  # Loop over lattice sites
            for n in range(1, M):  # Loop over synthetic levels
                # -J * (c_{n-1,j}^\dagger c_{n,j} + h.c.) term (tunneling term)
                if state[i] == n:
                    new_state = get_new_state(state, i, n-1)
                    new_index = state_index(new_state, states)
                    H[current_index, new_index] -= J  # Adding the tunneling element with -J
                    H[new_index, current_index] -= J  # Hermitian conjugate term
                
                # V * c_{n-1,j}^\dagger c_{n,j} c_{n,i}^\dagger c_{n-1,i} term (interaction term)
                for j in range(N):
                    if i != j and state[i] == n and state[j] == n-1:
                        new_state = get_double_new_state(state, i, n-1, j, n)
                        new_index = state_index(new_state, states)
                        H[current_index, new_index] += V  # Adding the interaction term
                        H[new_index, current_index] += V  # Hermitian conjugate term

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

def generate_J_V_ratio_plot(N, M, min_J_V_ratio, max_J_V_ratio, num_points=100, n_excited_states=2, include_degeneracy_plot=False):
    """
    Generates a plot (or plots, if include_degeneracy_plot is True) that plots the epsilon values of n_excited_states (and always
    the ground state) versus an array of J_V_ratios. If include_degeneracy_plot is True, then the differences between the levels 
    are also plotted, to check where the degeneracies are broken. 

    Inputs:
    N (float): The number of real lattice sites. 
    M (float): The number of synthetic levels per real lattice site.
    min_J_V_ratio (float): The lowest J/V ratio that will be used in the plot.
    max_J_V_ratio (float): The highest J/V ratio that will be used in the plot.
    num_points (float): The number of J/V ratios that will be used in the plot betwen min_J_V_ratio and max_J_V_ratio.
    n_excited_states (float): The number of excited states that will be included in the plots.
    include_degeneracy_plot (bool): Whether or not to include the differences between the levels.
    
    Returns:
    None (None)
    
    """
    
    V = 1
    J_V_ratios = np.linspace(min_J_V_ratio, max_J_V_ratio, num_points)

    energies = {f"E_{i}": [] for i in range(n_excited_states + 1)}
    states = {f"$\psi_{i}": [] for i in range(n_excited_states + 1)}

    for J_V_ratio in J_V_ratios:
        J = J_V_ratio * V
        H, formatted_states = construct_hamiltonian(N, M, J, V)
        V_matrix, D_matrix = exact_diagonalize(H)
        eigenvalues = np.diag(D_matrix)
        eigenvectors = [V_matrix[:, col_idx] for col_idx in range(V_matrix.shape[1])]
        
        for j in range(n_excited_states + 1):
            var_name = f"E_{j}"
            state_name = f"$\psi_{j}"
            energies[var_name].append(eigenvalues[j])
            states[state_name].append(eigenvectors[j])

    eps = {f"eps_{i}": [E / N for E in energies[f"E_{i}"]] for i in range(n_excited_states + 1)}

    if include_degeneracy_plot:
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7, 5), gridspec_kw={'height_ratios': [2.5, 1]}, sharex=True)
    else: 
        fig, ax1= plt.subplots(figsize = (7,5))
        
    for i in range(n_excited_states + 1):
        var_name = f"eps_{i}"
        ax1.plot(J_V_ratios, eps[var_name], '.', label=f"$\\epsilon_{i}$")

    flat_line_eps = round(eps["eps_0"][len(eps["eps_0"]) // 2], 5)
    ax1.axhline(flat_line_eps, linestyle="--", color="k", label=f"$\\epsilon^*=${flat_line_eps}")
    
    ax1.legend(loc="lower center", ncols=n_excited_states + 2, fancybox=True)
    ax1.set_title(f"{N} Real Sites and {M} Synthetic Sites")
    ax1.set_ylabel("$\\epsilon$ [E/N]")
    ax1.grid()

    if include_degeneracy_plot:
        delta_eps = {f"eps{i+1}-eps_{i}": [eps[f"eps_{i+1}"][j] - eps[f"eps_{i}"][j] for j in range(len(eps[f"eps_{i+1}"]))] for i in range(n_excited_states)}

        for i in range(n_excited_states):
            var_name = f"eps{i+1}-eps_{i}"
            ax2.plot(J_V_ratios, delta_eps[var_name], '.', label=f"$\\epsilon_{i+1}-\\epsilon_{i}$")
        
        ax2.legend(loc="upper center", ncols=n_excited_states)
        ax2.set_xlabel("J/V Ratio")
        ax2.set_ylabel("$\\Delta \\epsilon$ [E/N]")
        ax2.grid()
    else:
        ax1.set_xlabel("J/V Ratio")

    fig.tight_layout()
    plt.show()
        
    return None
    
# --------------------------------------------------------------------------------------------------------------------------------------------
