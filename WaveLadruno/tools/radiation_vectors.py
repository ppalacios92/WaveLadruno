import numpy as np

def create_angular_grid(n_theta=300, n_phi=300):
    """
    Crea malla angular para patrones de radiación.
    
    Parameters
    ----------
    n_theta, n_phi : int
        Resolución de la malla angular
        
    Returns
    -------
    theta_grid, phi_grid : ndarray
        Grillas de coordenadas angulares
    """
    theta = np.linspace(0, np.pi, n_theta)
    phi = np.linspace(0, 2*np.pi, n_phi)
    theta_grid, phi_grid = np.meshgrid(theta, phi)
    return theta_grid, phi_grid

def get_spherical_unit_vectors(theta_grid, phi_grid):
    """
    Calcula vectores esféricos unitarios.
    
    Parameters
    ----------
    theta_grid, phi_grid : ndarray
        Grillas de coordenadas angulares
        
    Returns
    -------
    gamma, r_hat, theta_hat, phi_hat : ndarray
        Vectores unitarios esféricos
    """
    gamma = np.array([
        np.sin(theta_grid)*np.cos(phi_grid),
        np.sin(theta_grid)*np.sin(phi_grid),
        np.cos(theta_grid)
    ])
    
    r_hat = gamma
    theta_hat = np.array([
        np.cos(theta_grid)*np.cos(phi_grid),
        np.cos(theta_grid)*np.sin(phi_grid),
        -np.sin(theta_grid)
    ])
    phi_hat = np.array([
        -np.sin(phi_grid),
        np.cos(phi_grid),
        np.zeros_like(phi_grid)
    ])
    
    return gamma, r_hat, theta_hat, phi_hat

def A_N(gamma):
    """
    Patrón tensorial A_N (near-field).
    
    Parameters
    ----------
    gamma : ndarray
        Vector unitario radial
        
    Returns
    -------
    A : ndarray
        Tensor del patrón A_N
    """
    delta = np.eye(3)
    shape = gamma.shape[1:]
    A = np.zeros((3,3,3) + shape)
    
    for n in range(3):
        for p in range(3):
            for q in range(3):
                A[n,p,q] = (15*gamma[n]*gamma[p]*gamma[q]
                            - 3*(gamma[n]*delta[p,q] + gamma[p]*delta[n,q] + gamma[q]*delta[n,p]))
    return A

def A_IP(gamma):
    """
    Patrón tensorial A_IP (intermediate P-field).
    
    Parameters
    ----------
    gamma : ndarray
        Vector unitario radial
        
    Returns
    -------
    A : ndarray
        Tensor del patrón A_IP
    """
    delta = np.eye(3)
    shape = gamma.shape[1:]
    A = np.zeros((3,3,3) + shape)
    
    for n in range(3):
        for p in range(3):
            for q in range(3):
                A[n,p,q] = (6*gamma[n]*gamma[p]*gamma[q]
                            - (gamma[n]*delta[p,q] + gamma[p]*delta[n,q] + gamma[q]*delta[n,p]))
    return A

def A_IS(gamma):
    """
    Patrón tensorial A_IS (intermediate S-field).
    
    Parameters
    ----------
    gamma : ndarray
        Vector unitario radial
        
    Returns
    -------
    A : ndarray
        Tensor del patrón A_IS
    """
    delta = np.eye(3)
    shape = gamma.shape[1:]
    A = np.zeros((3,3,3) + shape)
    
    for n in range(3):
        for p in range(3):
            for q in range(3):
                A[n,p,q] = -(6*gamma[n]*gamma[p]*gamma[q]
                             - gamma[n]*delta[p,q] - gamma[p]*delta[n,q] - 2*gamma[q]*delta[n,p])
    return A

def A_FP(gamma):
    """
    Patrón tensorial A_FP (far-field P-wave).
    
    Parameters
    ----------
    gamma : ndarray
        Vector unitario radial
        
    Returns
    -------
    A : ndarray
        Tensor del patrón A_FP
    """
    shape = gamma.shape[1:]
    A = np.zeros((3,3,3) + shape)
    
    for n in range(3):
        for p in range(3):
            for q in range(3):
                A[n,p,q] = gamma[n]*gamma[p]*gamma[q]
    return A

def A_FS(gamma):
    """
    Patrón tensorial A_FS (far-field S-wave).
    
    Parameters
    ----------
    gamma : ndarray
        Vector unitario radial
        
    Returns
    -------
    A : ndarray
        Tensor del patrón A_FS
    """
    delta = np.eye(3)
    shape = gamma.shape[1:]
    A = np.zeros((3,3,3) + shape)
    
    for n in range(3):
        for p in range(3):
            for q in range(3):
                A[n,p,q] = -(gamma[n]*gamma[p] - delta[n,p]) * gamma[q]
    return A

def A_TOTAL(gamma):
    """
    Patrón tensorial total (suma de todos).
    
    Parameters
    ----------
    gamma : ndarray
        Vector unitario radial
        
    Returns
    -------
    A : ndarray
        Tensor del patrón total
    """
    return A_N(gamma) + A_IP(gamma) + A_IS(gamma) + A_FP(gamma) + A_FS(gamma)

def get_all_patterns(gamma):
    """
    Obtiene todos los patrones tensoriales.
    
    Parameters
    ----------
    gamma : ndarray
        Vector unitario radial
        
    Returns
    -------
    patterns : dict
        Diccionario con todos los patrones
    """
    patterns = {
        "AN": A_N(gamma),
        "AIP": A_IP(gamma),
        "AIS": A_IS(gamma),
        "AFP": A_FP(gamma),
        "AFS": A_FS(gamma),
        "ATOTAL": A_TOTAL(gamma)
    }
    return patterns

def contract_tensor(A, M):
    """
    Contracción tensorial A_{npq} M_{pq}.
    
    Parameters
    ----------
    A : ndarray
        Tensor patrón de forma (3,3,3,...)
    M : ndarray
        Tensor momento de forma (3,3)
        
    Returns
    -------
    out : ndarray
        Vector resultante de la contracción
    """
    out = np.zeros((3,) + A.shape[3:])
    for n in range(3):
        for p in range(3):
            for q in range(3):
                out[n] += A[n,p,q] * M[p,q]
    return out