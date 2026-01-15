import numpy as np
from itertools import combinations

def _intersect_segment_plane(p0, p1, n, d, tol=1e-12):
    v = p1 - p0
    denom = np.dot(n, v)
    if abs(denom) < tol:
        return None
    t = -(np.dot(n, p0) + d) / denom
    if 0.0 <= t <= 1.0:
        return p0 + t * v
    return None


def _cell_edges(x0, x1, y0, y1, z0, z1):
    corners = np.array([
        [x0, y0, z0], [x1, y0, z0],
        [x0, y1, z0], [x1, y1, z0],
        [x0, y0, z1], [x1, y0, z1],
        [x0, y1, z1], [x1, y1, z1],
    ])

    edges = []
    for a, b in combinations(corners, 2):
        if np.count_nonzero(np.abs(a - b)) == 1:
            edges.append((a, b))
    return edges


def _physical_to_parametric(p, x0, x1, y0, y1, z0, z1):
    return np.array([
        (p[0] - x0) / (x1 - x0),
        (p[1] - y0) / (y1 - y0),
        (p[2] - z0) / (z1 - z0),
    ])


class PlaneGridIntersect:
    """
    Plane–rectilinear grid intersection and triangulation.
    """

    def __init__(self, xaxis, yaxis, zaxis, ra, normal):
        self.x = np.asarray(xaxis, dtype=float)
        self.y = np.asarray(yaxis, dtype=float)
        self.z = np.asarray(zaxis, dtype=float)

        self.ra = np.asarray(ra, dtype=float)
        self.normal = np.asarray(normal, dtype=float)
        self.normal /= np.linalg.norm(self.normal)

        # Plane equation n·x + d = 0
        self.d = -np.dot(self.normal, self.ra)

        self._intersections = None
        self._triangles = None

    def get_triangles(self):
        """
        Return {(i,j,k): [(a,b,c), ...]} in parametric coordinates.
        """

        if self._triangles is not None:
            return self._triangles

        if self._intersections is None:
            self._compute_intersections()

        triangles = {}

        for cell, pts in self._intersections.items():
            pts = np.asarray(pts)
            if len(pts) < 3:
                continue

            centroid = pts.mean(axis=0)

            # Local 2D basis
            ref = np.array([1.0, 0.0, 0.0])
            if abs(np.dot(ref, self.normal)) > 0.9:
                ref = np.array([0.0, 1.0, 0.0])

            e1 = np.cross(self.normal, ref)
            e1 /= np.linalg.norm(e1)
            e2 = np.cross(self.normal, e1)

            proj = np.array([
                [np.dot(p - centroid, e1), np.dot(p - centroid, e2)]
                for p in pts
            ])

            angles = np.arctan2(proj[:, 1], proj[:, 0])
            order = np.argsort(angles)
            pts = pts[order]

            tris = []
            p0 = pts[0]
            for i in range(1, len(pts) - 1):
                a, b, c = p0, pts[i], pts[i+1]
                if np.dot(np.cross(b - a, c - a), self.normal) < 0:
                    b, c = c, b
                tris.append((a, b, c))

            triangles[cell] = tris

        self._triangles = triangles
        return triangles


    def save_grid(self, file_name):
        with open(file_name, "w") as f:
            f.write("# vtk DataFile Version 3.0\n")
            f.write("Rectilinear grid\n")
            f.write("ASCII\n")
            f.write("DATASET RECTILINEAR_GRID\n")
            f.write(f"DIMENSIONS {len(self.x)} {len(self.y)} {len(self.z)}\n")

            f.write(f"X_COORDINATES {len(self.x)} float\n")
            for v in self.x:
                f.write(f"{v}\n")

            f.write(f"Y_COORDINATES {len(self.y)} float\n")
            for v in self.y:
                f.write(f"{v}\n")

            f.write(f"Z_COORDINATES {len(self.z)} float\n")
            for v in self.z:
                f.write(f"{v}\n")

    def save_triangles(self, file_name):
        tris = self.get_triangles()

        points = []
        polys = []
        cell_ids = []

        point_map = {}

        def pid(p):
            key = tuple(np.round(p, 12))
            if key not in point_map:
                point_map[key] = len(points)
                points.append(p)
            return point_map[key]

        for (i, j, k), tlist in tris.items():
            for a, b, c in tlist:
                pa = self._param_to_phys(a, i, j, k)
                pb = self._param_to_phys(b, i, j, k)
                pc = self._param_to_phys(c, i, j, k)

                ia, ib, ic = pid(pa), pid(pb), pid(pc)
                polys.append((ia, ib, ic))
                cell_ids.append((i, j, k))

        with open(file_name, "w") as f:
            f.write("# vtk DataFile Version 3.0\n")
            f.write("Plane slice\n")
            f.write("ASCII\n")
            f.write("DATASET POLYDATA\n")

            f.write(f"POINTS {len(points)} float\n")
            for p in points:
                f.write(f"{p[0]} {p[1]} {p[2]}\n")

            f.write(f"POLYGONS {len(polys)} {4 * len(polys)}\n")
            for a, b, c in polys:
                f.write(f"3 {a} {b} {c}\n")

            f.write(f"CELL_DATA {len(polys)}\n")
            f.write("SCALARS cell_i int 1\nLOOKUP_TABLE default\n")
            for i, _, _ in cell_ids:
                f.write(f"{i}\n")

            f.write("SCALARS cell_j int 1\nLOOKUP_TABLE default\n")
            for _, j, _ in cell_ids:
                f.write(f"{j}\n")

            f.write("SCALARS cell_k int 1\nLOOKUP_TABLE default\n")
            for _, _, k in cell_ids:
                f.write(f"{k}\n")


    def _compute_intersections(self, tol=1e-10):
        intersections = {}

        for i in range(len(self.x) - 1):
            for j in range(len(self.y) - 1):
                for k in range(len(self.z) - 1):

                    x0, x1 = self.x[i], self.x[i+1]
                    y0, y1 = self.y[j], self.y[j+1]
                    z0, z1 = self.z[k], self.z[k+1]

                    pts = []
                    for p0, p1 in _cell_edges(x0, x1, y0, y1, z0, z1):
                        p = _intersect_segment_plane(
                            p0, p1, self.normal, self.d
                        )
                        if p is not None:
                            pts.append(p)

                    if not pts:
                        continue

                    # Deduplicate
                    uniq = []
                    for p in pts:
                        if not any(np.linalg.norm(p - q) < tol for q in uniq):
                            uniq.append(p)

                    param = [
                        _physical_to_parametric(p, x0, x1, y0, y1, z0, z1)
                        for p in uniq
                    ]

                    intersections[(i, j, k)] = param

        self._intersections = intersections

    def _param_to_phys(self, p, i, j, k):
        u, v, w = p
        return np.array([
            self.x[i] + u * (self.x[i+1] - self.x[i]),
            self.y[j] + v * (self.y[j+1] - self.y[j]),
            self.z[k] + w * (self.z[k+1] - self.z[k]),
        ])


###############################################################################

def test1():

    # define the rectilinear grid
    x = np.linspace(0, 1, 6)
    y = np.linspace(0, 1, 5)
    z = np.linspace(0, 1, 4)**2 # non uniform

    # define the plan using one point ad a normal
    ra = np.array([0.2, 0.3, 0.7])
    normal = np.array([0.8, 0.2, 0.2])

    # compute the intersections

    pg = PlaneGridIntersect(x, y, z, ra, normal)
    pg.save_grid("grid.vtk")
    pg.save_triangles("triangles.vtk")

if __name__ == '__main__':
    test1()