"""
v1029b_keyed_cond_utils.py

Condition vectors built from a chosen KEY: a short list of coarse TO-spec
fields. Shape is always part of the bucket key (it reaches the model through
the domain mask) but never part of the condition vector. Each field becomes a
one-hot (bcpat: multi-hot) block; the manufacturability slot [label, 1] is
appended last.

Fields (k = cells per axis, n_vf = VF bins):
  load    load cell                                     one-hot k^3
  vf      VF quantile bin                               one-hot n_vf
  bccen   cell holding the centroid of the BC points    one-hot k^3
  ldist   grid steps from load cell to nearest BC cell  one-hot 3(k-1)+1
  bcpat   full BC cell pattern                          multi-hot k^3
  dir     load-direction class (dominant axis, sign)    one-hot 6

Example: key "load+vf" at k=2, 4 VF bins -> 8 + 4 + 2 = 14 dims, bucket key
"64x32x16|L5|V2".

All fields are always computed and stored in sample_info (so plots can show
everything), but only the key's fields enter the condition and the bucket.
"""

import numpy as np

from v1029_binned_cond_utils import (
    DIR_CLASSES, cell_xyz, point_cell, points_cells, load_dir_class, vf_bin,
)

ALL_FIELDS = ["load", "vf", "bccen", "ldist", "bcpat", "dir"]


def parse_key(key_str):
    fields = [f.strip() for f in key_str.replace(",", "+").split("+") if f.strip()]
    fields = [f for f in fields if f != "shape"]   # shape is implicit
    bad = [f for f in fields if f not in ALL_FIELDS]
    if bad:
        raise ValueError(f"Unknown key field(s) {bad}; choose from {ALL_FIELDS}")
    if not fields:
        raise ValueError("Key needs at least one field besides shape")
    return fields


def field_size(field, k, n_vf):
    return {"load": k ** 3, "vf": n_vf, "bccen": k ** 3, "ldist": 3 * (k - 1) + 1,
            "bcpat": k ** 3, "dir": len(DIR_CLASSES)}[field]


def keyed_layout(fields, k, n_vf):
    slices, pos = {}, 0
    for f in fields:
        n = field_size(f, k, n_vf)
        slices[f"{f}_onehot" if f != "bcpat" else "bcpat_multihot"] = (pos, pos + n)
        pos += n
    slices["manuf_label"] = (pos, pos + 1)
    slices["manuf_has_label"] = (pos + 1, pos + 2)
    return slices, pos + 2


def all_field_values(shape, bc_points, load_point, load_vec, vf, k, vf_edges):
    bc = np.asarray(bc_points, dtype=np.float64).reshape(-1, 3)
    load_c = point_cell(load_point, shape, k)
    bc_cs = points_cells(bc, shape, k)
    lx = np.array(cell_xyz(load_c, k))
    ldist = min(int(np.abs(lx - np.array(cell_xyz(c, k))).sum()) for c in bc_cs) if bc_cs else 0
    return {
        "load": int(load_c),
        "vf": int(vf_bin(vf, vf_edges)),
        "bccen": int(point_cell(bc.mean(axis=0), shape, k)) if len(bc) else 0,
        "ldist": int(ldist),
        "bcpat": [int(c) for c in bc_cs],
        "dir": int(load_dir_class(load_vec)),
    }


def field_token(f, v):
    if f == "load":
        return f"L{v}"
    if f == "vf":
        return f"V{v}"
    if f == "bccen":
        return f"C{v}"
    if f == "ldist":
        return f"d{v}"
    if f == "bcpat":
        return "B" + ("-".join(str(c) for c in v) if v else "none")
    if f == "dir":
        return "D" + (DIR_CLASSES[v] if v >= 0 else "zero")
    raise ValueError(f)


def keyed_bucket_key(shape, fields, values):
    s = f"{int(shape[0])}x{int(shape[1])}x{int(shape[2])}"
    return "|".join([s] + [field_token(f, values[f]) for f in fields])


def encode_keyed(fields, values, label_pm1, k, n_vf):
    slices, dim = keyed_layout(fields, k, n_vf)
    cond = np.zeros(dim, dtype=np.float32)
    for f in fields:
        name = f"{f}_onehot" if f != "bcpat" else "bcpat_multihot"
        a = slices[name][0]
        if f == "bcpat":
            for c in values["bcpat"]:
                cond[a + c] = 1.0
        elif values[f] >= 0:
            cond[a + values[f]] = 1.0
    cond[slices["manuf_label"][0]] = float(label_pm1)
    cond[slices["manuf_has_label"][0]] = 1.0
    return cond


def decode_keyed(cond_vec, slices, fields):
    out = {}
    cond_vec = np.asarray(cond_vec, dtype=np.float32)
    for f in fields:
        name = f"{f}_onehot" if f != "bcpat" else "bcpat_multihot"
        a, b = slices[name]
        seg = cond_vec[a:b]
        if f == "bcpat":
            out[f] = [int(i) for i in np.nonzero(seg > 0.5)[0]]
        else:
            out[f] = int(np.argmax(seg)) if seg.max() > 0.5 else -1
    out["manuf_label"] = float(cond_vec[slices["manuf_label"][0]])
    out["manuf_has_label"] = float(cond_vec[slices["manuf_has_label"][0]])
    return out
