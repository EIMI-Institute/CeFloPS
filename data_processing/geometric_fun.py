import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from skimage.measure import EllipseModel
import logging
from CeFloPS.logger_config import setup_logger
from numba import jit
logger = setup_logger(__name__, level=logging.WARNING)

PI = 3.14159265359


def get_area_from_ring(ring, plane_position, plane_normal):
    """get_area_from_ring Function to get area from ring by fitting an ellipse

    Args:
        ring (list): ringshaped pointcloud
        plane_position (list): plane positional vector
        plane_normal (list): plane normal vector

    Returns:
        int: area of the ring assuming an ellipsis shape
    """
    assert len(ring) > 0, "ring should contain points"
    points_to_fit = project_3d_points_to_2d(ring, plane_position, plane_normal)
    # print(len(points_to_fit))
    res = fit_ellipse(points_to_fit)
    if res == None:
        return np.inf
    xc, yc, a, b, theta = fit_ellipse(points_to_fit)
    area = a * b * PI
    return area


def get_center_ellipsis(ring, plane_position, plane_normal):
    points_to_fit = project_3d_points_to_2d(ring, plane_position, plane_normal)
    xc, yc, a, b, theta = fit_ellipse(points_to_fit)
    return translate_2d_to_3d([[xc, yc]], plane_position, plane_normal)


def plot_ellipse(xc, yc, a, b, theta, points=[], ax=None):
    """plot_ellipse Function to plot an ellipsis given the parameters and points

    Args:
        xc (_type_): _description_
        yc (_type_): _description_
        a (_type_): _description_
        b (_type_): _description_
        theta (_type_): _description_
        ax (_type_, optional): _description_. Defaults to None.
    """
    t = np.linspace(0, 2 * np.pi, 100)
    Ell = np.array([a * np.cos(t), b * np.sin(t)])
    R = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    Ell_rot = R.dot(Ell)

    if ax is None:
        ax = plt.gca()

    ax.plot(xc + Ell_rot[0, :], yc + Ell_rot[1, :], "r--", label="Fitted Ellipse")
    ax.plot(points[:, 0], points[:, 1], "g.", label="Points")
    ax.set_aspect("equal")
    ax.legend()
    plt.show()


def fit_ellipse(points):
    """fit_ellipse Function to fit an ellipsis for given points

    Args:
        points (_type_): Points in 2D coordinate space

    Returns:
        _type_: ellipse centre coordinates, axislenghts and theta
    """
    # use ellipsismodel of skimage
    ellipse = EllipseModel()
    try:
        ellipse.estimate(points)
    except Exception as e:
        logger.warning(e, points)
        return None
    # print(len(points))
    #  retrieve model parameters
    if ellipse.params != None:
        xc, yc, a, b, theta = ellipse.params
    else:
        return None
    return xc, yc, a, b, theta


def project_3d_points_to_2d(points, plane_point, plane_normal):
    """
    Projekt 3D-Punkte auf eine 2D-Ebene.

    :param points: numpy array (n, 3) of 3D-points
    :param plane_point: numpy array (3,) with plane point
    :param plane_normal: numpy array (3,) with normal of the plane
    :return: numpy array (n, 2) with 2D-Projection of points on plane
    """
    # normal vector of plane
    plane_normal = plane_normal / np.linalg.norm(plane_normal)

    # directional vector for plane
    if plane_normal[0] != 0 or plane_normal[1] != 0:
        u = np.array([-plane_normal[1], plane_normal[0], 0])
    else:
        u = np.array([1, 0, 0])

    u = u / np.linalg.norm(u)

    # find second directional vector
    v = np.cross(plane_normal, u)
    v = v / np.linalg.norm(v)

    projected_points = []

    for point in points:
        # translate point
        relative_point = point - plane_point

        # project onto basevectors
        p_u = np.dot(relative_point, u)
        p_v = np.dot(relative_point, v)

        projected_points.append([p_u, p_v])

    return np.array(projected_points)


def translate_2d_to_3d(points_2d, plane_point, plane_normal):
    """
    Translate 2D points back to 3D coordinates on the plane.

    :param points_2d: numpy array (n, 2) of 2D points
    :param plane_point: numpy array (3,) with a reference point on the plane
    :param plane_normal: numpy array (3,) with the normal of the plane
    :return: numpy array (n, 3) with 3D coordinates on the plane
    """
    # normalize plane normal
    plane_normal = plane_normal / np.linalg.norm(plane_normal)

    # compute a directional vector u for the plane
    if plane_normal[0] != 0 or plane_normal[1] != 0:
        u = np.array([-plane_normal[1], plane_normal[0], 0])
    else:
        u = np.array([1, 0, 0])

    u = u / np.linalg.norm(u)

    # find a second directional vector v
    v = np.cross(plane_normal, u)
    v = v / np.linalg.norm(v)

    # translate each 2D point back to 3D
    translated_points = [
        plane_point + point_2d[0] * u + point_2d[1] * v for point_2d in points_2d
    ]

    return np.array(translated_points)


def tilt_vector(v, angles):
    """tilt_vector Rotate a vector along the x,y, or z axis by a specified angle

    Args:
        v (tuple): vector to rotate
        angles (tuple): angles to rotate around x,y,z axis

    Returns:
        np array: rotated vector
    """
    assert len(angles) == 3 and len(v) == 3
    # convert angles from degrees to radians
    angles = np.radians(angles)

    # define the rotation matrix for each axis
    Rx = np.array(
        [
            [1, 0, 0],
            [0, np.cos(angles[0]), -np.sin(angles[0])],
            [0, np.sin(angles[0]), np.cos(angles[0])],
        ]
    )
    Ry = np.array(
        [
            [np.cos(angles[1]), 0, np.sin(angles[1])],
            [0, 1, 0],
            [-np.sin(angles[1]), 0, np.cos(angles[1])],
        ]
    )
    Rz = np.array(
        [
            [np.cos(angles[2]), -np.sin(angles[2]), 0],
            [np.sin(angles[2]), np.cos(angles[2]), 0],
            [0, 0, 1],
        ]
    )

    # obtain rotated vector by applying all rotation matrices sequentially
    tilted_vector = Rx @ Ry @ Rz @ v

    return tilted_vector


def plot_vector(vec):
    """plot_vector Method to plot vectors

    Args:
        vec (_type_): _description_
    """
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    # start points  (0, 0, 0) and direction of the vector as two separate lists: [x, y, z]
    ax.quiver([0], [0], [0], [vec[0]], [vec[1]], [vec[2]], color=["r"])

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")

    # axis limits
    ax.set_xlim([-0, 10])
    ax.set_ylim([-0, 10])
    ax.set_zlim([-0, 10])

    plt.show()
@jit(nopython=True)
def random_point_on_circle(center, normal, radius):
    """
    Generate a random point on a circle with a given center, normal, and radius.

    Parameters:
    center (tuple): Coordinates of the center of the circle (x, y, z).
    normal (tuple): Normal vector defining the orientation of the plane of the circle.
    radius (float): Radius of the circle.

    Returns:
    tuple: Coordinates of a random point on the circle.
    """
    # Normalize normal in case of it not being normalized already
    normal_normalized = normal / np.linalg.norm(normal)

    # Find two perpendicular vectors in the plane of the circle
    in_plane = np.empty(3, dtype=np.float64)

    if np.all(normal[:2] == 0):
        in_plane[0] = 0.0
        in_plane[1] = 1.0
        in_plane[2] = 0.0
    else:
        # Use the z-axis as a reference
        in_plane = np.cross(
            normal_normalized, np.array([0.0, 0.0, 1.0], dtype=np.float64)
        )

        # If normal is aligned with z-axis, choose y-axis as default
        if np.linalg.norm(in_plane) < 1e-10:
            in_plane = np.array([1.0, 0.0, 0.0], dtype=np.float64)

    in_plane /= np.linalg.norm(in_plane)

    cross_vec = np.cross(normal_normalized, in_plane)
    cross_vec /= np.linalg.norm(cross_vec)

    # Random angle to use for random point on circle
    theta = np.random.uniform(0, 2 * np.pi)

    # Get point on the circle
    circle_point = np.empty(3, dtype=np.float64)
    circle_point[0] = (
        center[0]
        + radius * np.cos(theta) * in_plane[0]
        + radius * np.sin(theta) * cross_vec[0]
    )
    circle_point[1] = (
        center[1]
        + radius * np.cos(theta) * in_plane[1]
        + radius * np.sin(theta) * cross_vec[1]
    )
    circle_point[2] = (
        center[2]
        + radius * np.cos(theta) * in_plane[2]
        + radius * np.sin(theta) * cross_vec[2]
    )

    return circle_point


@jit(nopython=True)
def closest_point_on_circle(A, B, r, N):
    """
    Calculate the closest point on a circle with radius r centered at A,
    and lying in the plane defined by the normal N, to a point B.
    """
    AB = B - A
    N_normalized = N / np.linalg.norm(N)

    projection_to_plane = AB - np.dot(AB, N_normalized) * N_normalized

    if np.linalg.norm(projection_to_plane) < 1e-10:
        # random value if exaxtly under centre: find two perpendicular vectors in the plane of the circle
        raise ValueError(
            "Point B lies directly above or below the center A, making the closest point determination ambiguous."
        )

    unit_projection = projection_to_plane / np.linalg.norm(projection_to_plane)
    closest_point_on_circle = A + r * unit_projection

    return closest_point_on_circle
