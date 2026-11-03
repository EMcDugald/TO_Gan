"""
v1029_binned_cond_utils.py

Shared helpers for the v1029 "binned TO spec" pipeline (data generation,
bucket diagnostics, training-data visualizer, and later the sampler).

The condition vector replaces exact coordinates with coarse cells measured
relative to each part's OWN extent:

  load_cell_onehot      k^3   one-hot: which cell holds the load point
  bc_cells_multihot     k^3   multi-hot: which cells hold ANY fixed BC node
  vf_bin_onehot         n_vf  one-hot: volume-fraction quantile bin
  load_dir_class_onehot 6     one-hot: dominant axis and sign of the load
  manuf_label           1     +1 positive / -1 negative
  manuf_has_label       1     1 (all parts are labeled in v1029)

BC DOFs are omitted. Every entry is 0/1 except the +/-1 label, so a TO spec
is fully described by a short "bucket key", e.g.

  64x32x16|L5|B0-1-4-5|V2|D-z

and two parts share a neighborhood exactly when they share that key. The
manufacturability label is NOT part of the key.

Coordinate convention (same as the existing data): BC/load coords are in
[0,1] normalized by max(Nx,Ny,Nz); voxel index = round(coord*(max_dim-1)),
clipped to the part. A voxel with index i along an axis of length N has
relative position (i + 0.5)/N, and its cell along that axis is
floor(k * (i + 0.5)/N). So cell j along an axis covers voxel coordinates
[j*N/k, (j+1)*N/k], which is also how the plots draw the cell boxes.

Cell id = bx*k*k + by*k + bz (x slowest). Readable name "x1y0z1".

Slice names are chosen so the existing decode_condition_overlay() in
v0720_neural_diff_trainer_fixed.py finds none of its "fine" slices (it only
draws markers for bc_points/load_point/load_dir), so the unchanged trainer
just plots in-training samples without markers.
"""

import json
import numpy as np

DIR_CLASSES = ["+x", "-x", "+y", "-y", "+z", "-z"]


# -------------------------------------------------------------------------
# geometry -> cells
# -------------------------------------------------------------------------

def coord_to_voxel_index(coord01, shape):
    """[0,1] max-dim-normalized coords (..., 3) -> integer voxel indices."""
    shape = np.asarray(shape, dtype=np.int64)
    c = float(shape.max())
    idx = np.rint(np.asarray(coord01, dtype=np.float64) * (c - 1.0)).astype(np.int64)
    return np.clip(idx, 0, shape - 1)


def voxel_index_to_cell(idx, shape, k):
    """Integer voxel indices (..., 3) -> per-axis cell indices in [0, k-1]."""
    shape = np.asarray(shape, dtype=np.float64)
    rel = (np.asarray(idx, dtype=np.float64) + 0.5) / shape
    return np.clip(np.floor(rel * k).astype(np.int64), 0, k - 1)


def cell_id(bxyz, k):
    bxyz = np.asarray(bxyz, dtype=np.int64)
    return bxyz[..., 0] * k * k + bxyz[..., 1] * k + bxyz[..., 2]


def cell_xyz(cid, k):
    cid = int(cid)
    return (cid // (k * k), (cid // k) % k, cid % k)


def cell_name(cid, k):
    bx, by, bz = cell_xyz(cid, k)
    return f"x{bx}y{by}z{bz}"


def point_cell(coord01, shape, k):
    return int(cell_id(voxel_index_to_cell(coord_to_voxel_index(coord01, shape), shape, k), k))


def points_cells(coords01, shape, k):
    coords01 = np.asarray(coords01, dtype=np.float64).reshape(-1, 3)
    if len(coords01) == 0:
        return []
    ids = cell_id(voxel_index_to_cell(coord_to_voxel_index(coords01, shape), shape, k), k)
    return sorted(set(int(v) for v in np.atleast_1d(ids)))


def load_dir_class(vec):
    """Dominant axis and sign -> class index into DIR_CLASSES; -1 if zero."""
    v = np.asarray(vec, dtype=np.float64).reshape(3)
    if not np.all(np.isfinite(v)) or np.max(np.abs(v)) == 0.0:
        return -1
    a = int(np.argmax(np.abs(v)))
    return 2 * a + (0 if v[a] > 0 else 1)


# -------------------------------------------------------------------------
# volume fraction bins
# -------------------------------------------------------------------------

def vf_quantile_edges(vfs, n_bins):
    """n_bins+1 edges from quantiles; interior edges decide the bin."""
    qs = np.linspace(0.0, 1.0, n_bins + 1)
    edges = np.quantile(np.asarray(vfs, dtype=np.float64), qs)
    for i in range(1, len(edges)):
        if edges[i] <= edges[i - 1]:
            edges[i] = edges[i - 1] + 1e-9
    return edges


def vf_bin(vf, edges):
    """Bin by interior edges, so values outside the training range clamp."""
    interior = np.asarray(edges, dtype=np.float64)[1:-1]
    return int(np.searchsorted(interior, float(vf), side="right"))


# -------------------------------------------------------------------------
# condition vector + bucket key
# -------------------------------------------------------------------------

def cond_layout(k, n_vf_bins):
    """Slice table, identical for every part."""
    n_cells = k ** 3
    slices, pos = {}, 0
    for name, n in [("load_cell_onehot", n_cells), ("bc_cells_multihot", n_cells),
                    ("vf_bin_onehot", n_vf_bins), ("load_dir_class_onehot", len(DIR_CLASSES)),
                    ("manuf_label", 1), ("manuf_has_label", 1)]:
        slices[name] = (pos, pos + n)
        pos += n
    return slices, pos


def make_bucket_key(shape, load_c, bc_cs, vf_b, dir_c):
    s = f"{int(shape[0])}x{int(shape[1])}x{int(shape[2])}"
    b = "-".join(str(c) for c in bc_cs) if bc_cs else "none"
    d = DIR_CLASSES[dir_c] if dir_c >= 0 else "zero"
    return f"{s}|L{load_c}|B{b}|V{vf_b}|D{d}"


def encode_binned(shape, bc_points, load_point, load_vec, vf, label_pm1,
                  k, vf_edges):
    """Returns (cond_vec float32, decoded dict incl. bucket_key)."""
    n_vf = len(vf_edges) - 1
    slices, dim = cond_layout(k, n_vf)
    cond = np.zeros(dim, dtype=np.float32)

    load_c = point_cell(load_point, shape, k)
    bc_cs = points_cells(bc_points, shape, k)
    vf_b = vf_bin(vf, vf_edges)
    dir_c = load_dir_class(load_vec)

    cond[slices["load_cell_onehot"][0] + load_c] = 1.0
    for c in bc_cs:
        cond[slices["bc_cells_multihot"][0] + c] = 1.0
    cond[slices["vf_bin_onehot"][0] + vf_b] = 1.0
    if dir_c >= 0:
        cond[slices["load_dir_class_onehot"][0] + dir_c] = 1.0
    cond[slices["manuf_label"][0]] = float(label_pm1)
    cond[slices["manuf_has_label"][0]] = 1.0

    decoded = {
        "load_cell": int(load_c),
        "bc_cells": [int(c) for c in bc_cs],
        "vf_bin": int(vf_b),
        "load_dir_class": int(dir_c),
        "bucket_key": make_bucket_key(shape, load_c, bc_cs, vf_b, dir_c),
    }
    return cond, decoded


def decode_binned(cond_vec, slices):
    """Inverse of encode_binned for plotting a condition with no part."""
    cond_vec = np.asarray(cond_vec, dtype=np.float32)

    def seg(name):
        a, b = slices[name]
        return cond_vec[a:b]

    load = seg("load_cell_onehot")
    dirs = seg("load_dir_class_onehot")
    return {
        "load_cell": int(np.argmax(load)) if load.max() > 0.5 else -1,
        "bc_cells": [int(i) for i in np.nonzero(seg("bc_cells_multihot") > 0.5)[0]],
        "vf_bin": int(np.argmax(seg("vf_bin_onehot"))),
        "load_dir_class": int(np.argmax(dirs)) if dirs.max() > 0.5 else -1,
        "manuf_label": float(seg("manuf_label")[0]),
        "manuf_has_label": float(seg("manuf_has_label")[0]),
    }


def read_binned_meta(meta):
    """Pull the v1029 fields out of a loaded _meta.npz."""
    return {
        "k_bins": int(meta["k_bins"]),
        "n_vf_bins": int(meta["n_vf_bins"]),
        "vf_edges": np.asarray(meta["vf_edges"], dtype=np.float64),
        "cond_slices": {k_: tuple(v) for k_, v in json.loads(str(meta["cond_slices_json"])).items()},
        "dir_classes": json.loads(str(meta["dir_classes_json"])),
        "manuf_threshold_value": float(meta["manuf_threshold_value"]),
    }


def vf_bin_label(b, edges):
    return f"VF bin {b} [{edges[b]:.3f}, {edges[b + 1]:.3f}]"


# -------------------------------------------------------------------------
# plotting
# -------------------------------------------------------------------------

def _box_faces(x0, x1, y0, y1, z0, z1):
    v = np.array([[x0, y0, z0], [x1, y0, z0], [x1, y1, z0], [x0, y1, z0],
                  [x0, y0, z1], [x1, y0, z1], [x1, y1, z1], [x0, y1, z1]])
    return [[v[0], v[1], v[2], v[3]], [v[4], v[5], v[6], v[7]],
            [v[0], v[1], v[5], v[4]], [v[2], v[3], v[7], v[6]],
            [v[1], v[2], v[6], v[5]], [v[0], v[3], v[7], v[4]]]


def draw_cell_box(ax, cid, shape, k, color, alpha=0.10, lw=1.4):
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    nx, ny, nz = [float(s) for s in shape]
    bx, by, bz = cell_xyz(cid, k)
    faces = _box_faces(bx * nx / k, (bx + 1) * nx / k,
                       by * ny / k, (by + 1) * ny / k,
                       bz * nz / k, (bz + 1) * nz / k)
    pc = Poly3DCollection(faces, facecolors=color, edgecolors=color,
                          linewidths=lw, alpha=alpha)
    ax.add_collection3d(pc)


def plot_part_binned(ax, vox, shape, k, load_cell=None, bc_cells=(),
                     exact_bc=None, exact_load=None, load_vec=None,
                     elev=25, azim=35, title=""):
    """One 3D panel: voxels, BC cells (blue boxes), load cell (red box),
    and, when given, the exact BC points / load point as small markers."""
    nx, ny, nz = [int(s) for s in shape]
    c = float(max(nx, ny, nz))
    if vox is not None:
        ax.voxels(np.asarray(vox) > 0, facecolors="0.75", edgecolor="k",
                  linewidth=0.1, alpha=0.55)
    for cid in bc_cells:
        draw_cell_box(ax, cid, shape, k, "tab:blue", alpha=0.08)
    if load_cell is not None and load_cell >= 0:
        draw_cell_box(ax, load_cell, shape, k, "tab:red", alpha=0.12, lw=2.0)
    if exact_bc is not None and len(exact_bc) > 0:
        p = coord_to_voxel_index(np.asarray(exact_bc).reshape(-1, 3), shape) + 0.5
        ax.scatter(p[:, 0], p[:, 1], p[:, 2], c="navy", s=6, depthshade=False)
    if exact_load is not None:
        p = coord_to_voxel_index(np.asarray(exact_load).reshape(3), shape) + 0.5
        ax.scatter([p[0]], [p[1]], [p[2]], c="red", s=40, marker="^",
                   edgecolors="white", linewidths=0.6, depthshade=False)
        if load_vec is not None and np.max(np.abs(load_vec)) > 0:
            lv = np.asarray(load_vec, dtype=np.float64)
            ax.quiver(p[0], p[1], p[2], lv[0], lv[1], lv[2], color="red",
                      linewidth=1.8, length=c * 0.18, normalize=True)
    ax.set_xlim(0, nx); ax.set_ylim(0, ny); ax.set_zlim(0, nz)
    ax.set_box_aspect((nx, ny, nz))
    ax.view_init(elev=elev, azim=azim)
    ax.set_axis_off()
    if title:
        ax.set_title(title, fontsize=8)
