#!/usr/bin/env python3
"""Audit MR-LBM turbulent jet RMS/Reynolds profiles without rescaling their amplitude.

Place this file alongside post/jet/validation/statistics_io.py and execute from
any working directory using --data and --output.  Uses the same Hussein et al.
digitizations present in the MR-LBM repository's plotting scripts.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from statistics_io import fit_virtual_origin, load_radial_profiles

# Experimental digitizations already used by the original jet plotting scripts.
HUSSEIN_MEAN_ETA = np.array([
    0, .010599057494345044, .019815647572123837, .030875562697158625,
    .04009215277493742, .04838709790833872, .056221185411050346,
    .06405527291376191, .07281104051935197, .08479260058876421,
    .09400919066654304, .10737325331102249, .11658984338880131,
    .12949308356109202, .14147464363050433, .15345620369991658,
    .16866359139165193, .18248847650832015, .19723500656936588,
    .21290321673329005, .22672810184995826, .2447004595333271,
])
HUSSEIN_MEAN = np.array([
    .9986970883174464, .988273596045439, .9622149647712097,
    .9205211944947582, .8657980588782979, .8136807963298389,
    .7563517876453765, .7042345250969178, .6469055164124551,
    .5635178764537632, .514006536676201, .4332247200826162,
    .38371338030505414, .32899019498569865, .2690553629361291,
    .2247556698887811, .16742676061010808, .11791512261517773,
    .08143319788051967, .05276854442960405, .03452768146806465,
    .024104189196056974,
])
HUSSEIN_UY_RMS_ETA = np.array([
    0, .01, .02, .03, .04, .05, .06, .07, .08, .09, .10, .11,
    .12, .13, .14, .15, .16, .17, .18, .19, .20, .21, .22, .235,
])
HUSSEIN_UY_RMS = np.array([
    .278, .282, .286, .286, .283, .276, .265, .252,
    .235, .216, .196, .176, .155, .136, .118, .101,
    .084, .069, .056, .045, .036, .029, .023, .018,
])
HUSSEIN_UR_RMS_ETA = np.array([
    0, .015, .03, .045, .06, .075, .09, .105, .12, .135,
    .15, .165, .18, .195, .21, .225, .24, .26, .28, .3,
])
HUSSEIN_UR_RMS = np.array([
    .235, .224, .216, .207, .197, .185, .174, .162,
    .148, .132, .114, .096, .079, .063, .049, .037,
    .027, .017, .011, .007,
])
HUSSEIN_STRESS_ETA = np.array([
    0, .01, .02, .03, .04, .05, .06, .07, .08, .09,
    .105, .12, .135, .15, .17, .19, .21, .23,
])
HUSSEIN_STRESS = np.array([
    0, .006, .011, .016, .020, .022, .022, .021,
    .019, .017, .014, .011, .009, .007, .0045, .0025,
    .0010, .0002,
])


def halfwidth(r: np.ndarray, vals: np.ndarray, center: float) -> float:
    if not np.isfinite(center) or center <= 0:
        return float("nan")
    f = vals / center
    indices = np.flatnonzero(np.isfinite(f) & (f <= .5))
    indices = indices[indices > 0]
    if not len(indices):
        return float("nan")
    i = int(indices[0])
    if not np.isfinite(f[i - 1]) or f[i] == f[i - 1]:
        return float("nan")
    return float(r[i-1] + (.5-f[i-1])*(r[i]-r[i-1])/(f[i]-f[i-1]))


def save_overlay(
    out: Path, name: str, profile: np.ndarray, p, stations: list[float],
    exp_eta: np.ndarray, exp_value: np.ndarray, exp_half_eta: float,
    ylabel: str, coord: str, y0: float, *, ylims=None,
):
    fig, ax = plt.subplots(figsize=(7.5, 5.2))
    diameter = float(p.metadata['D'])
    uc = p.uy[:, 0]
    radial = np.arange(p.uy.shape[1], dtype=float)
    for j, y_over_d in enumerate(stations):
        iy = int(round(y_over_d * diameter))
        if iy < 1 or iy >= p.uy.shape[0] or not np.isfinite(uc[iy]) or uc[iy] <= 0:
            print(f"Skipping slice y/D={y_over_d:g}")
            continue
        hw = halfwidth(radial, p.uy[iy], uc[iy])
        denom = hw if coord == 'half' else (iy-y0)
        if not np.isfinite(denom) or denom <= 0:
            print(f"Skipping invalid length scale at y/D={y_over_d:g}")
            continue
        good = (p.count[iy] > 0) & np.isfinite(profile[iy])
        if not np.any(good):
            continue
        ax.plot(radial[good]/denom, profile[iy, good],
                marker=['o','s','^','D','v','<'][j % 6],
                linestyle='none', markersize=3, alpha=.8,
                label=rf'$y/D={y_over_d:g}$')
        if name.startswith('uy_rms'):
            diagnostic = float(np.nanmax(profile[iy, good]))
            print(f"  y/D={y_over_d:g}: Uc={uc[iy]:.7g}, r_half/D={hw/diameter:.5f}, "
                  f"r_half/(y-y0)={hw/(iy-y0):.5f}, peak Uy_rms/Uc={diagnostic:.5f}")
    exp_x = exp_eta/exp_half_eta if coord=='half' else exp_eta
    ax.plot(exp_x, exp_value, 'k+', markersize=7, label='Hussein et al. (1994)')
    ax.set_xlabel(r'$r/r_{1/2}$' if coord=='half' else r'$r/(y-y_0)$')
    ax.set_ylabel(ylabel)
    ax.set_xlim(0, 3.8 if coord=='half' else .33)
    if ylims is not None:
        ax.set_ylim(*ylims)
    ax.grid(alpha=.2)
    ax.legend(fontsize=8, ncol=2, frameon=False)
    fig.tight_layout()
    path=out/name
    fig.savefig(path, dpi=300)
    plt.close(fig)
    print('Saved:', path)


def audit_raw_variances(folder: Path, metadata: dict[str, str], stations: list[float]):
    """Check whether variance clipping in statistics_io concealed any negative FP32 moments."""
    shape = (int(metadata['NY']), int(metadata['NR_BINS']))
    elements = shape[0] * shape[1]
    uy_path = folder / 'radial_sum_uy.bin'
    dtype = '<f4' if uy_path.stat().st_size == elements * 4 else '<f8'
    raw = {
        name: np.fromfile(folder / f'radial_sum_{name}.bin', dtype=dtype).reshape(shape).astype(np.float64)
        for name in ('uy', 'uy2', 'ur', 'ur2')
    }
    counts = np.fromfile(folder / 'radial_count.bin', dtype='<u8').reshape(shape)
    print('Raw variance consistency (negative values may indicate FP32 accumulation/cancellation):')
    for station in stations:
        iy = int(round(station * float(metadata['D'])))
        if iy <= 0 or iy >= shape[0]:
            continue
        c = counts[iy].astype(np.float64)
        valid = c > 0
        if not np.any(valid):
            print(f'  y/D={station:g}: no samples')
            continue
        for component in ('uy', 'ur'):
            var = raw[f'{component}2'][iy,valid]/c[valid] - (raw[component][iy,valid]/c[valid])**2
            neg = np.count_nonzero(var < 0)
            print(f'  y/D={station:g}, {component}: negative variances={neg}/{var.size}; '
                  f'minimum={np.nanmin(var):.4e}; mean count={np.nanmean(c[valid]):.1f}')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--output', type=Path, default=Path('post/jet/validation/plots/diagnostics'))
    parser.add_argument('--fit-start', type=float, default=10)
    parser.add_argument('--fit-end', type=float, default=15)
    parser.add_argument('--slices', type=float, nargs='+', default=[10,11,12,13,14,15])
    parser.add_argument('--stress-sign', type=float, choices=(-1.,1.), default=1.)
    args=parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    p=load_radial_profiles(args.data)
    d=float(p.metadata['D'])
    uj=float(p.metadata['jet_velocity'])
    yc=np.arange(p.uy.shape[0], dtype=float)
    uc=p.uy[:,0]
    B,y0,_,_=fit_virtual_origin(yc,uc,d,uj,args.fit_start,args.fit_end)
    exp_half=halfwidth(HUSSEIN_MEAN_ETA,HUSSEIN_MEAN,1.0)
    print(f"Data directory: {args.data}")
    print(f"Metadata: Re={p.metadata.get('Re')}, D={d:g}, Ujet={uj:g}, sample_count={p.metadata.get('radial_sample_count')}")
    print(f"Centerline-decay fit: B={B:.6g}, virtual origin y0/D={y0/d:.6g}")
    print(f"Hussein mean halfwidth: r_half/(y-y0)={exp_half:.6g}")
    print("Reference peaks: Uy_rms/Uc=%.5f, Ur_rms/Uc=%.5f, <ur'uy'>/Uc²=%.5f" %
          (max(HUSSEIN_UY_RMS), max(HUSSEIN_UR_RMS), max(HUSSEIN_STRESS)))
    audit_raw_variances(args.data, p.metadata, args.slices)
    with np.errstate(invalid='ignore',divide='ignore'):
        uy_norm=p.uy/uc[:,None]
        uy_rms_norm=p.uy_rms/uc[:,None]
        ur_rms_norm=p.ur_rms/uc[:,None]
        stress_norm=args.stress_sign*p.reynolds_uruy/(uc[:,None]**2)
    for coord in ['eta','half']:
        save_overlay(args.output,f'mean_{coord}_diagnostic.png',uy_norm,p,args.slices,
                     HUSSEIN_MEAN_ETA,HUSSEIN_MEAN,exp_half,
                     r'$\langle u_y\rangle/U_c$',coord,y0,ylims=(0,1.1))
    for name,values,ex,ey,ylabel,ylim in [
        ('uy_rms',uy_rms_norm,HUSSEIN_UY_RMS_ETA,HUSSEIN_UY_RMS,r'$u_{y,\mathrm{rms}}/U_c$',(0,.35)),
        ('ur_rms',ur_rms_norm,HUSSEIN_UR_RMS_ETA,HUSSEIN_UR_RMS,r'$u_{r,\mathrm{rms}}/U_c$',(0,.35)),
        ('uruy',stress_norm,HUSSEIN_STRESS_ETA,HUSSEIN_STRESS,r'$\langle u_r\prime u_y\prime\rangle/U_c^2$',(-.005,.04)),
    ]:
        save_overlay(args.output, f'{name}_rhalf_diagnostic.png',values,p,args.slices,
                     ex,ey,exp_half,ylabel,'half',y0,ylims=ylim)
    print("Note: vertical values are NOT artificially scaled. The alternate x-axis only removes half-width differences.")

if __name__=='__main__':
    main()
