"""unmerge-cli entry point.

  unmerge --volume mask.tif --click X Y Z --out out/
  unmerge --volume mask.tif --candidates points.json --out out/

Clicks are given in X Y Z (Khartes convention); volumes are Z-Y-X arrays.
Volumes: .tif/.tiff binary or probability (thresholded at --thr), or .npy.
"""
import argparse
import json
import os

import numpy as np
import tifffile
from scipy import ndimage as ndi

from . import core
from .exporter import export_sheet, export_flag


def load_volume(path, thr):
    if path.endswith(".npy"):
        vol = np.load(path)
    elif path.endswith((".tif", ".tiff")):
        vol = tifffile.imread(path)
    else:
        raise SystemExit(f"unsupported volume format: {path} (.tif/.npy)")
    if vol.dtype.kind == "f":
        return vol.astype(np.float32) > thr
    return vol > 0


def process_point(mask, nrm, p0, out_dir, tag, tau):
    rep = {"click_xyz": [int(p0[2]), int(p0[1]), int(p0[0])],
           "point_zyx": [int(v) for v in p0], "tau": tau}
    site = core.extract_site(mask, nrm, p0)
    if isinstance(site, str):
        rep["decision"] = site
        return rep
    decision, extras = core.decide(site, tau=tau)
    rep["decision"] = decision
    rep["seed_rays"] = site["n_seed_rays"]
    if "median_margin" in extras:
        rep["median_margin"] = round(extras["median_margin"], 4)
    if decision != "SPLIT":
        rep["obj"] = export_flag(decision, p0, out_dir, tag)
        return rep
    idx, phi = extras["idx"], extras["phi"]
    crop = site["crop"]
    vox = np.argwhere(crop)
    lab, _ = ndi.label(crop, structure=np.ones((3, 3, 3)))
    comp = lab == lab[core.HALF, core.HALF, core.HALF]
    phiv = np.full(crop.shape, np.nan, np.float32)
    phiv[tuple(vox.T)] = phi
    origin = tuple(int(v) - core.HALF for v in p0)
    out = {}
    for name, m, color in (("A", comp & (phiv > 0), (0.1, 0.5, 1.0)),
                           ("B", comp & (phiv < 0), (1.0, 0.5, 0.1))):
        path = os.path.join(out_dir, f"Sheet_{name}_{tag}.obj")
        nv_, nf_ = export_sheet(m, origin, path, color)
        out[name] = {"obj": path, "verts": nv_, "faces": nf_,
                     "vox": int(m.sum())}
    rep["sheets"] = out
    return rep


def main():
    ap = argparse.ArgumentParser(prog="unmerge", description=__doc__)
    ap.add_argument("--volume", required=True,
                    help="mask or probability volume (.tif/.npy, Z-Y-X)")
    ap.add_argument("--click", nargs=3, type=int, metavar=("X", "Y", "Z"),
                    help="contact point in Khartes X Y Z")
    ap.add_argument("--candidates",
                    help="JSON list of [X, Y, Z] points (e.g. from qc192)")
    ap.add_argument("--out", default="./unmerge_out")
    ap.add_argument("--tau", type=float, default=core.TAU,
                    help="split-acceptance margin threshold (default 0.10)")
    ap.add_argument("--thr", type=float, default=0.6,
                    help="binarization threshold for probability volumes")
    args = ap.parse_args()
    if not args.click and not args.candidates:
        ap.error("provide --click X Y Z or --candidates points.json")
    os.makedirs(args.out, exist_ok=True)
    mask = load_volume(args.volume, args.thr)
    nrm = core.normals_of(mask)
    points = []
    if args.click:
        x, y, z = args.click
        points.append((z, y, x))
    if args.candidates:
        for x, y, z in json.load(open(args.candidates)):
            points.append((int(z), int(y), int(x)))
    reports = []
    for i, p0 in enumerate(points):
        tag = f"x{p0[2]}y{p0[1]}z{p0[0]}"
        rep = process_point(mask, nrm, p0, args.out, tag, args.tau)
        reports.append(rep)
        print(f"[{i+1}/{len(points)}] {rep['decision']} at XYZ="
              f"{rep['click_xyz']}"
              + (f" margin {rep['median_margin']}" if "median_margin" in rep
                 else ""))
    with open(os.path.join(args.out, "report.json"), "w") as fh:
        json.dump(reports, fh, indent=1)
    print(f"report: {os.path.join(args.out, 'report.json')}")


if __name__ == "__main__":
    main()
