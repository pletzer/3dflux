import numpy as np
from itertools import combinations

def plane_from_points(ra, rb, rc):
    """
    Return (n, d) such that n·x + d = 0 defines the plane
    """
    n = np.cross(rb - ra, rc - ra)
    n = n / np.linalg.norm(n)
    d = -np.dot(n, ra)
    return n, d


def intersect_segment_plane(p0, p1, n, d, tol=1e-12):
    """
    Intersect segment p(t) = p0 + t (p1 - p0), t ∈ [0,1]
    with plane n·x + d = 0.
    Returns point or None.
    """
    v = p1 - p0
    denom = np.dot(n, v)

    if abs(denom) < tol:
        return None  # parallel

    t = -(np.dot(n, p0) + d) / denom
    if 0.0 <= t <= 1.0:
        return p0 + t * v

    return None


def cell_edges(x0, x1, y0, y1, z0, z1):
    """
    Return the 12 edges of a hexahedral cell as (p0, p1)
    """
    corners = np.array([
        [x0, y0, z0],
        [x1, y0, z0],
        [x0, y1, z0],
        [x1, y1, z0],
        [x0, y0, z1],
        [x1, y0, z1],
        [x0, y1, z1],
        [x1, y1, z1],
    ])

    # Edges = pairs of corners differing in exactly one coordinate
    edges = []
    for a, b in combinations(corners, 2):
        if np.count_nonzero(np.abs(a - b)) == 1:
            edges.append((a, b))
    return edges


def physical_to_parametric(p, x0, x1, y0, y1, z0, z1):
    """
    Map physical point p to (u, v, w) ∈ [0,1]^3 in the cell
    """
    u = (p[0] - x0) / (x1 - x0)
    v = (p[1] - y0) / (y1 - y0)
    w = (p[2] - z0) / (z1 - z0)
    return np.array([u, v, w])


def plane_grid_intersections(x, y, z, ra, rb, rc, tol=1e-10):
    """
    Main function.

    Returns:
        dict mapping (i, j, k) -> list of parametric intersection points
    """
    ra, rb, rc = map(np.asarray, (ra, rb, rc))
    n, d = plane_from_points(ra, rb, rc)

    result = {}

    for i in range(len(x) - 1):
        for j in range(len(y) - 1):
            for k in range(len(z) - 1):

                x0, x1 = x[i], x[i+1]
                y0, y1 = y[j], y[j+1]
                z0, z1 = z[k], z[k+1]

                points = []

                for p0, p1 in cell_edges(x0, x1, y0, y1, z0, z1):
                    p = intersect_segment_plane(p0, p1, n, d)
                    if p is not None:
                        points.append(p)

                if not points:
                    continue

                # Deduplicate points
                uniq = []
                for p in points:
                    if not any(np.linalg.norm(p - q) < tol for q in uniq):
                        uniq.append(p)

                # Convert to parametric coordinates
                param = [
                    physical_to_parametric(p, x0, x1, y0, y1, z0, z1)
                    for p in uniq
                ]

                result[(i, j, k)] = param

    return result

###############################################################################

def test1():

    # define the rectilinear grid
    x = np.linspace(0, 1, 6)
    y = np.linspace(0, 1, 5)
    z = np.linspace(0, 1, 4)

    # define the plan using 3 points
    ra = np.array([0.2, 0.2, 0.2])
    rb = np.array([0.8, 0.2, 0.2])
    rc = np.array([0.2, 0.8, 0.8])

    # compute the intersections
    intersections = plane_grid_intersections(x, y, z, ra, rb, rc)

    for cell, pts in intersections.items():
        print(cell)
        for p in pts:
            print("  ", p)

if __name__ == '__main__':
    test1()