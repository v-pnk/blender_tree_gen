import math
import numpy as np

def norm_vec(vec: np.ndarray) -> np.ndarray:
    """
    Normalize a vector

    Parameters
    ----------
    vec : np.ndarray
        The vector to normalize

    Returns
    -------
    np.ndarray
        The normalized vector
    """

    return vec / np.linalg.norm(vec)


def get_rotmat_xz(x_ang: float, z_ang: float) -> np.ndarray:
    """
    Generate a rotation matrix from the given angles

    Parameters
    ----------
    x_ang : float
        Angle around the branch X-axis
    z_ang : float
        Angle around the branch Z-axis

    Returns
    -------
    np.ndarray
        Rotation matrix
    """

    x_ang_rad = math.radians(x_ang)
    z_ang_rad = math.radians(z_ang)

    x_rot = np.array([
        [1, 0, 0],
        [0, math.cos(x_ang_rad), -math.sin(x_ang_rad)],
        [0, math.sin(x_ang_rad), math.cos(x_ang_rad)]])

    z_rot = np.array([
        [math.cos(z_ang_rad), -math.sin(z_ang_rad), 0],
        [math.sin(z_ang_rad), math.cos(z_ang_rad), 0],
        [0, 0, 1]])

    rotmat = z_rot @ x_rot

    return rotmat
