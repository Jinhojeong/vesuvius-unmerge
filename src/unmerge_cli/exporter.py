"""OBJ export for Khartes (global volume coordinates, X Y Z vertex order).

Mesh/pointcloud writers adapted from the qc-export prototype used in the
65-site validation (valid Wavefront OBJ, marching-cubes path)."""
import os
import numpy as np

try:
    from skimage.measure import marching_cubes
except ImportError:
    marching_cubes = None


def write_mesh(verts, faces, path, color=None):
    with open(path, "w") as fh:
        fh.write("# unmerge-cli export\n")
        fh.write(f"# Vertices: {len(verts)}, Faces: {len(faces)}\n")
        for v in verts:
            if color is not None:
                fh.write(f"v {v[0]:.3f} {v[1]:.3f} {v[2]:.3f} "
                         f"{color[0]:.3f} {color[1]:.3f} {color[2]:.3f}\n")
            else:
                fh.write(f"v {v[0]:.3f} {v[1]:.3f} {v[2]:.3f}\n")
        for f in faces:
            fh.write(f"f {f[0]+1} {f[1]+1} {f[2]+1}\n")


def write_pointcloud(coords, path, color=(1.0, 0.2, 0.2)):
    with open(path, "w") as fh:
        fh.write("# unmerge-cli point cloud export\n")
        fh.write(f"# Points: {len(coords)}\n")
        for p in coords:
            fh.write(f"v {p[0]:.3f} {p[1]:.3f} {p[2]:.3f} "
                     f"{color[0]:.3f} {color[1]:.3f} {color[2]:.3f}\n")
        fh.write("p " + " ".join(str(i + 1) for i in range(len(coords))) + "\n")


def export_sheet(mask, origin_zyx, path, color):
    """Mesh a binary crop mask into global coords; pointcloud fallback."""
    if marching_cubes is not None and mask.sum() >= 20:
        try:
            verts, faces, _, _ = marching_cubes(mask.astype(np.float32),
                                                level=0.5)
            verts += np.array(origin_zyx, np.float32)
            write_mesh(verts[:, [2, 1, 0]], faces, path, color=color)
            return len(verts), len(faces)
        except Exception:
            pass
    coords = np.argwhere(mask) + np.array(origin_zyx)
    write_pointcloud(coords[:, [2, 1, 0]].astype(np.float32), path,
                     color=color)
    return int(mask.sum()), 0


def export_flag(decision, p0_zyx, out_dir, tag):
    """Octahedron marker at the click point; filename carries the reason."""
    z, y, x = [float(v) for v in p0_zyx]
    r = 3.0
    verts = np.array([[x - r, y, z], [x + r, y, z], [x, y - r, z],
                      [x, y + r, z], [x, y, z - r], [x, y, z + r]])
    faces = np.array([[0, 2, 4], [2, 1, 4], [1, 3, 4], [3, 0, 4],
                      [2, 0, 5], [1, 2, 5], [3, 1, 5], [0, 3, 5]])
    color = (1.0, 0.6, 0.1) if decision.startswith("DISCONNECTED") \
        else (1.0, 0.1, 0.1)
    path = os.path.join(out_dir, f"{decision}_{tag}.obj")
    write_mesh(verts, faces, path, color=color)
    return path
