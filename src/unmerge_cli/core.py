"""Core mechanism: ray-order seeds + anisotropic Dirichlet Laplacian solve.

Validation (Dataset059, 2026-07): relative-order identity 92.1% over 40
sites / 144 rays; in the packaged tool, 25% auto-split at 98.4% observed
ray accuracy on GT-quality masks, and all 19 true-weld sites on raw model
predictions refused with zero false splits. See README."""
import numpy as np
from scipy import ndimage as ndi
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import spsolve

SPAN, STEP = 7.0, 0.5
K = int(2 * SPAN / STEP) + 1
HALF = 32
SIG_D, LAM, GAM = 1.0, 2.0, 6.0
TAU = 0.10
MIN_SEED_RAYS, MIN_EVAL_RAYS = 5, 2
SEED_RING = (6, 12)
EVAL_RING = 4
OFFS = [(a, b, c) for a in (-1, 0, 1) for b in (-1, 0, 1) for c in (-1, 0, 1)
        if (a, b, c) != (0, 0, 0)]


def runs(profile):
    """Contiguous on-runs of a boolean ray profile: (centre, start, end)."""
    out = []
    i = 0
    while i < len(profile):
        if profile[i]:
            j = i
            while j < len(profile) and profile[j]:
                j += 1
            out.append(((i + j - 1) / 2.0, i, j - 1))
            i = j
        else:
            i += 1
    return out


def normals_of(mask):
    """Per-voxel sheet normals from the smoothed signed EDT gradient."""
    d_out = ndi.distance_transform_edt(~mask).astype(np.float32)
    d_in = ndi.distance_transform_edt(mask).astype(np.float32)
    sm = ndi.gaussian_filter(d_out - d_in, 1.0)
    n = np.stack(np.gradient(sm), 0).astype(np.float32)
    n /= (np.sqrt((n ** 2).sum(0)) + 1e-6)
    return n


def adlrw_solve(mask_crop, nrm_crop, seed_a, seed_b):
    """Dirichlet solve on the crop's voxel graph; conductance suppressed for
    steps along the local normal (the bridge direction). Returns (idx, phi)
    or (None, None)."""
    vox = np.argwhere(mask_crop)
    if len(vox) < 200:
        return None, None
    idx = -np.ones(mask_crop.shape, np.int64)
    idx[tuple(vox.T)] = np.arange(len(vox))
    sa = seed_a[tuple(vox.T)]
    sb = seed_b[tuple(vox.T)]
    fixed = sa | sb
    if fixed.sum() == 0 or (~fixed).sum() == 0:
        return None, None
    phi_fixed = np.zeros(len(vox))
    phi_fixed[sa] = 1.0
    phi_fixed[sb] = -1.0
    free = ~fixed
    fmap = -np.ones(len(vox), np.int64)
    fmap[free] = np.arange(free.sum())
    nvox = nrm_crop[:, vox[:, 0], vox[:, 1], vox[:, 2]].T
    rows, cols, ws = [], [], []
    bvec = np.zeros(free.sum())
    diag = np.zeros(free.sum())
    for dz, dy, dx in OFFS:
        q = vox + np.array([dz, dy, dx])
        inb = ((q >= 0) & (q < np.array(mask_crop.shape))).all(1)
        qi = np.full(len(vox), -1, np.int64)
        qi[inb] = idx[q[inb, 0], q[inb, 1], q[inb, 2]]
        ok = qi >= 0
        pi = np.arange(len(vox))[ok]
        qj = qi[ok]
        step = np.array([dz, dy, dx], float)
        d2 = step @ step
        ncos = np.abs((nvox[pi] * nvox[qj]).sum(1)).clip(0, 1)
        along = (nvox[pi] @ step) ** 2 / d2
        w = np.exp(-d2 / (2 * SIG_D ** 2) - LAM * (1 - ncos) ** 2 - GAM * along)
        fp = fmap[pi]
        isfree = fp >= 0
        fp = fp[isfree]
        qj2 = qj[isfree]
        w2 = w[isfree]
        da = np.zeros(free.sum())
        np.add.at(da, fp, w2)
        diag += da
        qfree = fmap[qj2]
        m = qfree >= 0
        rows.append(fp[m])
        cols.append(qfree[m])
        ws.append(-w2[m])
        ba = np.zeros(free.sum())
        np.add.at(ba, fp[~m], w2[~m] * phi_fixed[qj2[~m]])
        bvec += ba
    rows = np.concatenate(rows)
    cols = np.concatenate(cols)
    ws = np.concatenate(ws)
    nf = free.sum()
    lap = coo_matrix((np.concatenate([ws, diag]),
                      (np.concatenate([rows, np.arange(nf)]),
                       np.concatenate([cols, np.arange(nf)]))),
                     shape=(nf, nf)).tocsr()
    try:
        pf = spsolve(lap, bvec)
    except Exception:
        return None, None
    phi = phi_fixed.copy()
    phi[free] = pf
    return idx, phi


def extract_site(mask, nrm, p0):
    """Seeds + eval rays around p0. Returns dict with crop, seeds, eval pairs,
    normal frame — or a decision string when the site is not processable."""
    z, y, x = p0
    if min(z, y, x) < HALF or z >= mask.shape[0] - HALF or \
            y >= mask.shape[1] - HALF or x >= mask.shape[2] - HALF:
        return "OUT_OF_BOUNDS"
    nv = nrm[:, z, y, x]
    offs = np.linspace(-SPAN, SPAN, K)
    prof = ndi.map_coordinates(mask.astype(np.float32),
                               (np.array(p0, float)[None, :] +
                                offs[:, None] * nv[None, :]).T,
                               order=0, mode="constant")
    rc = runs(prof > 0.5)
    if len(rc) == 0:
        return "NOT_ON_SHEET"
    crop = mask[z - HALF:z + HALF, y - HALF:y + HALF, x - HALF:x + HALF]
    cn = nrm[:, z - HALF:z + HALF, y - HALF:y + HALF, x - HALF:x + HALF]
    centre = np.array([HALF] * 3, float)
    a = np.array([1.0, 0, 0]) if abs(nv[0]) < 0.9 else np.array([0, 1.0, 0])
    e1 = np.cross(nv, a)
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(nv, e1)
    seed_a = np.zeros_like(crop)
    seed_b = np.zeros_like(crop)
    eval_pairs = []
    n_seed = 0
    for du in range(-SEED_RING[1], SEED_RING[1] + 1, 2):
        for dv in range(-SEED_RING[1], SEED_RING[1] + 1, 2):
            o = centre + du * e1 + dv * e2
            cc = (o[None, :] + offs[:, None] * nv[None, :]).T
            gl = ndi.map_coordinates(crop.astype(np.float32), cc, order=0,
                                     mode="constant")
            rr = runs(gl > 0.5)
            if len(rr) != 2:
                continue
            ring = max(abs(du), abs(dv))
            ps = []
            okray = True
            for r_ in rr:
                pr = np.round(o + (r_[0] * STEP - SPAN) * nv).astype(int)
                if ((pr < 0) | (pr >= 2 * HALF)).any() or \
                        not crop[pr[0], pr[1], pr[2]]:
                    okray = False
                    break
                ps.append(pr)
            if not okray:
                continue
            if SEED_RING[0] <= ring <= SEED_RING[1]:
                for r_, sm in zip(rr, (seed_a, seed_b)):
                    for kk in range(r_[1], r_[2] + 1):
                        pr = np.round(o + (kk * STEP - SPAN) * nv).astype(int)
                        if ((pr >= 0) & (pr < 2 * HALF)).all() and \
                                crop[pr[0], pr[1], pr[2]]:
                            sm[pr[0], pr[1], pr[2]] = True
                n_seed += 1
            elif ring <= EVAL_RING:
                eval_pairs.append(ps)
    if (seed_a & seed_b).any():
        seed_b &= ~seed_a
    return {"crop": crop, "normals": cn, "seed_a": seed_a, "seed_b": seed_b,
            "n_seed_rays": n_seed, "eval_pairs": eval_pairs, "nv": nv}


def decide(site, tau=TAU):
    """Run the solve and the tau gate. Returns (decision, extras)."""
    if site["n_seed_rays"] < MIN_SEED_RAYS:
        return "WELD_FLAG_STARVED_PERIPHERY", {}
    idx, phi = adlrw_solve(site["crop"], site["normals"],
                           site["seed_a"], site["seed_b"])
    if phi is None:
        return "WELD_FLAG_SOLVE_FAILED", {}
    if np.isnan(phi).any():
        return "DISCONNECTED_GRAPH_FLAG", {"nan_voxels": int(np.isnan(phi).sum())}
    margins = []
    for p1, p2 in site["eval_pairs"]:
        # seed-adjacent eval rays inflate margins (rounding collisions)
        if site["seed_a"][tuple(p1)] or site["seed_b"][tuple(p1)] or \
                site["seed_a"][tuple(p2)] or site["seed_b"][tuple(p2)]:
            continue
        i1 = idx[tuple(p1)]
        i2 = idx[tuple(p2)]
        if i1 >= 0 and i2 >= 0:
            margins.append(float(phi[i1] - phi[i2]))
    if len(margins) < MIN_EVAL_RAYS:
        return "WELD_FLAG_NO_EVAL_RAYS", {}
    med = float(np.median(margins))
    if med < tau:
        return "WELD_FLAG_LOW_MARGIN", {"median_margin": med, "margins": margins}
    return "SPLIT", {"idx": idx, "phi": phi, "median_margin": med,
                     "margins": margins}
