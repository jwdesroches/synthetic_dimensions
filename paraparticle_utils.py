# --------------------------------------------------------------------------------------------------------------------------------------------
# imports
# --------------------------------------------------------------------------------------------------------------------------------------------

from synth_dim_model import *
import numpy as np
from itertools import product

# --------------------------------------------------------------------------------------------------------------------------------------------
# definitions
# --------------------------------------------------------------------------------------------------------------------------------------------

# identity for single site
id3 = np.eye(3)

# define x^{(±)}_{i,alpha}
x_matrices = {
    ('+', 1): np.array([[0, 0, 0], [0, 0, 0], [0, 1, 0]]),
    ('-', 1): np.array([[0, 0, 0], [0, 0, 1], [0, 0, 0]]),
    ('+', 2): np.array([[0, 1, 0], [0, 0, 0], [0, 0, 0]]),
    ('-', 2): np.array([[0, 0, 0], [1, 0, 0], [0, 0, 0]]),
}

# define T^{(±)}_{k,alpha,beta}
T_matrices = {
    ('+', 1, 1): np.array([[0, 0, 0], [0, 1, 0], [0, 0, -1]]),
    ('-', 1, 1): np.array([[0, 0, 0], [0, 1, 0], [0, 0, -1]]),
    ('+', 1, 2): np.array([[0, 0, 0], [0, 0, 0], [-1, 0, 0]]),
    ('-', 1, 2): np.array([[0, 0, -1], [0, 0, 0], [0, 0, 0]]),
    ('+', 2, 1): np.array([[0, 0, -1], [0, 0, 0], [0, 0, 0]]),
    ('-', 2, 1): np.array([[0, 0, 0], [0, 0, 0], [-1, 0, 0]]),
    ('+', 2, 2): np.array([[-1, 0, 0], [0, 1, 0], [0, 0, 0]]),
    ('-', 2, 2): np.array([[-1, 0, 0], [0, 1, 0], [0, 0, 0]]),
}

# --------------------------------------------------------------------------------------------------------------------------------------------
# functions
# --------------------------------------------------------------------------------------------------------------------------------------------

def build_paraparticle_operator(N, site_index, alpha, sign):
    assert 1 <= site_index <= N
    assert alpha in [1, 2]
    assert sign in ['+', '-']

    total_op = np.zeros((3**N, 3**N))

    # Generate all beta chains of length (site_index - 1)
    for beta_chain in product([1, 2], repeat=site_index - 1):
        op_list = []

        # Construct the x^{(sign)}_{i,beta_{i-1}} part
        beta_prev = beta_chain[-1] if beta_chain else alpha
        x_op = x_matrices[(sign, beta_prev)]
        op_list.append(x_op)

        # Build T^{(sign)} chain in reverse site order
        for k in reversed(range(site_index - 1)):
            beta_k = beta_chain[k]
            beta_k_minus_1 = alpha if k == 0 else beta_chain[k - 1]
            T_k = T_matrices[(sign, beta_k_minus_1, beta_k)]
            op_list.insert(0, T_k)  # prepend

        # Pad identity operators
        full_op_list = (
            [id3] * (site_index - len(op_list)) + op_list + [id3] * (N - site_index)
        )

        # Compute full Kronecker product
        kron_op = full_op_list[0]
        for m in range(1, len(full_op_list)):
            kron_op = np.kron(kron_op, full_op_list[m])

        total_op += kron_op

    return total_op

# --------------------------------------------------------------------------------------------------------------------------------------------

def construct_site_paraparticle_number_operator(N, site_index):
    """Placeholder definition."""
    assert 1 <= site_index <= N
    n_i = np.zeros((3**N, 3**N))

    for alpha in [1, 2]:
        psi_plus = build_paraparticle_operator(N, site_index, alpha, '+')
        psi_minus = build_paraparticle_operator(N, site_index, alpha, '-')
        n_i += psi_plus @ psi_minus

    return n_i

# --------------------------------------------------------------------------------------------------------------------------------------------

def construct_total_paraparticle_number_operator(N):
    """Placeholder definition."""
    total_n = np.zeros((3**N, 3**N))
    for i in range(1, N + 1):
        total_n += construct_site_paraparticle_number_operator(N, i)
    return total_n

# --------------------------------------------------------------------------------------------------------------------------------------------

def exact_diagonalize_with_total_paraparticle_number_symmetry(hamiltonian, total_paraparticle_number_operator):
    """Placeholder definition."""
    # Diagonalize the symmetry operator
    eigvals_symm, V_N = np.linalg.eigh(total_paraparticle_number_operator)

    # Rotate Hamiltonian into symmetry eigenbasis
    number_basis_ham = V_N.conj().T @ hamiltonian @ V_N

    # Identify unique symmetry sectors
    unique_sectors = np.unique(np.round(eigvals_symm, decimals=10))  # rounded for numerical stability
    sector_indices = {
        sector: np.where(np.isclose(eigvals_symm, sector, atol=1e-10))[0]
        for sector in unique_sectors
    }

    # Diagonalize each symmetry sector
    sector_eigenvalues = {}
    sector_eigenvectors = {}
    for sector, indices in sector_indices.items():
        block = number_basis_ham[np.ix_(indices, indices)]
        e_vals, e_vecs = np.linalg.eigh(block)
        sector_eigenvalues[sector] = e_vals
        sector_eigenvectors[sector] = e_vecs

    # Construct full eigenvectors in symmetry basis
    dim = hamiltonian.shape[0]
    full_symm_evecs = np.zeros((dim, dim), dtype=complex)
    for sector, indices in sector_indices.items():
        block_evecs = sector_eigenvectors[sector]
        idx = np.ix_(indices, indices)
        full_symm_evecs[idx] = block_evecs

    # Rotate back to original basis
    full_eigenvectors = V_N @ full_symm_evecs

    # Gather eigenvalues and eigenvectors
    full_eigenvalues = np.concatenate([sector_eigenvalues[sector] for sector in unique_sectors])
    eigenvectors = [full_eigenvectors[:, i] for i in range(full_eigenvectors.shape[1])]

    # Compute total paraparticle number in each eigenstate
    total_ns = [
        np.real_if_close(vec.conj().T @ total_paraparticle_number_operator @ vec)
        for vec in eigenvectors
    ]

    return full_eigenvalues, eigenvectors, np.array(total_ns)

# --------------------------------------------------------------------------------------------------------------------------------------------