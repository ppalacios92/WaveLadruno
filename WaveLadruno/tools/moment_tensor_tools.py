import numpy as np

def moment_tensor_from_strike_dip_rake(strike, dip, rake, M0=1.0):
    """
    Calcula el tensor de momento sísmico a partir de strike, dip y rake.

    Parámetros:
    - strike: ángulo de rumbo (°)
    - dip: ángulo de buzamiento (°)
    - rake: ángulo de deslizamiento (°)
    - M0: magnitud escalar del momento sísmico (Nm, opcional)

    Retorna:
    - M: tensor simétrico 3x3 (numpy array)
    """
    # Convertir ángulos a radianes
    phi = np.deg2rad(strike)  # φ (strike)
    delta = np.deg2rad(dip)   # δ (dip)
    lam = np.deg2rad(rake)    # λ (rake)

    # Componentes del tensor según Aki & Richards, Eq. (10.7.11)
    M11 = -M0 * (np.sin(delta) * np.cos(lam) * np.sin(2*phi) +
                 np.sin(2*delta) * np.sin(lam) * np.sin(phi)**2)

    M12 =  M0 * (np.sin(delta) * np.cos(lam) * np.cos(2*phi) +
                 0.5 * np.sin(2*delta) * np.sin(lam) * np.sin(2*phi))

    M13 = -M0 * (np.cos(delta) * np.cos(lam) * np.cos(phi) +
                 np.cos(2*delta) * np.sin(lam) * np.sin(phi))

    M22 =  M0 * (np.sin(delta) * np.cos(lam) * np.sin(2*phi) -
                 np.sin(2*delta) * np.sin(lam) * np.cos(phi)**2)

    M23 = -M0 * (np.cos(delta) * np.cos(lam) * np.sin(phi) -
                 np.cos(2*delta) * np.sin(lam) * np.cos(phi))

    M33 =  M0 * np.sin(2*delta) * np.sin(lam)

    # Tensor de momento simétrico 3x3
    M = np.array([[M11, M12, M13],
                  [M12, M22, M23],
                  [M13, M23, M33]])

    return M
