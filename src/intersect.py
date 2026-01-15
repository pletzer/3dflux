import numpy as np
from itertools import combinations

def parametric_to_physical(p, i, j, k, x, y, z):
    u, v, w = p
    return np.array([
        x[i] + u * (x[i+1] - x[i]),
        y[j] + v * (y[j+1] - y[j]),
        z[k] + w * (z[k+1] - z[k]),
    ])


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


def triangulate_plane_cell_intersections(intersections, plane_normal, tol=1e-12):
    """
    Triangulate plane–cell intersection polygons.

    Parameters
    ----------
    intersections : dict
        {(i,j,k): [p0, p1, ...]} where p are 3D parametric points
    plane_normal : array-like, shape (3,)
        Normal vector of the plane (orientation reference)
    tol : float
        Numerical tolerance

    Returns
    -------
    dict
        {(i,j,k): [(a,b,c), (d,e,f), ...]} where a,b,c are 3D parametric points
    """

    plane_normal = np.asarray(plane_normal)
    plane_normal = plane_normal / np.linalg.norm(plane_normal)

    triangles = {}

    for cell, pts in intersections.items():
        pts = np.asarray(pts)

        if len(pts) < 3:
            continue

        # ------------------------------------------------------------------
        # Compute centroid
        # ------------------------------------------------------------------
        centroid = pts.mean(axis=0)

        # ------------------------------------------------------------------
        # Construct local 2D basis in the plane
        # ------------------------------------------------------------------
        # Pick a vector not parallel to the normal
        ref = np.array([1.0, 0.0, 0.0])
        if abs(np.dot(ref, plane_normal)) > 0.9:
            ref = np.array([0.0, 1.0, 0.0])

        e1 = np.cross(plane_normal, ref)
        e1 /= np.linalg.norm(e1)
        e2 = np.cross(plane_normal, e1)

        # ------------------------------------------------------------------
        # Project points into 2D plane coordinates
        # ------------------------------------------------------------------
        proj = np.array([
            [np.dot(p - centroid, e1), np.dot(p - centroid, e2)]
            for p in pts
        ])

        # ------------------------------------------------------------------
        # Sort vertices counterclockwise
        # ------------------------------------------------------------------
        angles = np.arctan2(proj[:,1], proj[:,0])
        order = np.argsort(angles)
        ordered_pts = pts[order]

        # ------------------------------------------------------------------
        # Fan triangulation
        # ------------------------------------------------------------------
        cell_tris = []
        p0 = ordered_pts[0]

        for i in range(1, len(ordered_pts) - 1):
            a = p0
            b = ordered_pts[i]
            c = ordered_pts[i+1]

            # ------------------------------------------------------------------
            # Enforce consistent orientation
            # ------------------------------------------------------------------
            n_tri = np.cross(b - a, c - a)
            if np.dot(n_tri, plane_normal) < 0:
                b, c = c, b

            cell_tris.append((a, b, c))

        triangles[cell] = cell_tris

    return triangles


def write_plane_triangles_vtk(
    filename,
    triangles,
    x, y, z,
    ascii=True
):
    """
    Write triangulated plane–grid intersections to a VTK PolyData file.

    Parameters
    ----------
    filename : str
        Output VTK filename (e.g. 'slice.vtk')
    triangles : dict
        {(i,j,k): [(a,b,c), ...]} in parametric coordinates
    x, y, z : array-like
        Grid coordinate arrays
    ascii : bool
        Write ASCII VTK (recommended for debugging)
    """

    points = []
    polys = []

    point_map = {}   # maps tuple(x,y,z) -> global point index

    def get_point_id(p):
        key = tuple(np.round(p, 12))  # tolerance-based hashing
        if key not in point_map:
            point_map[key] = len(points)
            points.append(p)
        return point_map[key]

    # ------------------------------------------------------------------
    # Collect points and triangles
    # ------------------------------------------------------------------
    for (i, j, k), tris in triangles.items():
        for a, b, c in tris:
            pa = parametric_to_physical(a, i, j, k, x, y, z)
            pb = parametric_to_physical(b, i, j, k, x, y, z)
            pc = parametric_to_physical(c, i, j, k, x, y, z)

            ia = get_point_id(pa)
            ib = get_point_id(pb)
            ic = get_point_id(pc)

            polys.append((ia, ib, ic))

    # ------------------------------------------------------------------
    # Write legacy VTK PolyData
    # ------------------------------------------------------------------
    with open(filename, "w") as f:
        f.write("# vtk DataFile Version 3.0\n")
        f.write("Plane-grid intersection triangulation\n")
        f.write("ASCII\n" if ascii else "BINARY\n")
        f.write("DATASET POLYDATA\n")

        # Points
        f.write(f"POINTS {len(points)} float\n")
        for p in points:
            f.write(f"{p[0]} {p[1]} {p[2]}\n")

        # Polygons
        f.write(f"POLYGONS {len(polys)} {4 * len(polys)}\n")
        for ia, ib, ic in polys:
            f.write(f"3 {ia} {ib} {ic}\n")

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

    print("intersection points")
    for cell, pts in intersections.items():
        print(cell)
        for p in pts:
            print(f"  ", p)

    plane_normal = np.cross(rb - ra, rc - ra)
    plane_normal /= np.sqrt(plane_normal.dot(plane_normal))
    cell_triangles = triangulate_plane_cell_intersections(intersections, plane_normal)

    print("cell triangles")
    for cell, triangles in cell_triangles.items():
        print(cell)
        for triangle in triangles:
            print("  ", triangle)

    write_plane_triangles_vtk("triangles.vtk", cell_triangles,
        x, y, z, ascii=True)

if __name__ == '__main__':
    test1()