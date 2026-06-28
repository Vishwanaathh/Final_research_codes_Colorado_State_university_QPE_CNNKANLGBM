"""
build_qpe_dataset_T4x9x9.py
============================
Research-production dataset builder for QPE using:
  - NEXRAD Level-2 V06 files (downloaded via nexradaws from KMLB, 2025)
  - NASA GPM GMIN rain-gauge files (SFL network, Florida)

Output tensor shape: T_{4 x 9 x 9}
    ch 0  Z    at lowest  elevation (~0.5 deg)
    ch 1  ZDR  at lowest  elevation (~0.5 deg)
    ch 2  Z    at second-lowest elevation (~0.9 deg)
    ch 3  ZDR  at second-lowest elevation (~0.9 deg)

Target: mean rain-gauge rate (mm/h) within [t_scan_i, t_scan_{i+1}]

DROPS2.0 Inspired QC
--------------------------
  Step 1 -- Non-precipitation echo removal:
              Gates with rhoHV < RHOHV_THRESH (0.85) OR Z < Z_MIN_DBZ (5 dBZ)
              are masked as non-meteorological.  The union mask is applied to
              Z and ZDR in-place.  Any NaN gate in the 9x9 sample window
              causes that sample to be discarded.

  KDP is intentionally excluded:
    - KDP is unreliable below ~5 mm/h; the dataset minimum is 1 mm/h
    - Z and ZDR alone fully parameterise the R(Z, ZDR) relation used by
      the downstream CNN-KAN-LightGBM pipeline

Quality filters:
  - Per-interval mean rate < MIN_RATE_MMH (1 mm/h) => excluded.
  - Any NaN in 9x9 window => excluded.

Checkpoint / resume system
---------------------------
  Progress is saved every CHECKPOINT_EVERY (50) scan intervals so that a
  stopped run can be resumed without losing work.

  Layout inside output_dir/checkpoints/:
    {i:06d}.npz    accepted samples from the batch ending at interval i
                   contains arrays: X (M,4,9,9), y (M,), meta (JSON string)
    {i:06d}.done   zero-byte marker for batches with no accepted samples
                   records that those intervals were processed

  On startup: finds the highest completed interval index, loads all existing
  checkpoint data, then resumes the loop from the next unprocessed interval.

  After a successful final save: checkpoints/ directory is deleted.

  To force a full re-run from scratch:
    rm -rf <output_dir>/checkpoints

Train/val split: 80/20 by scan-time (no temporal leakage).

Usage
-----
    python build_qpe_dataset_T4x9x9.py \\
        --radar_dir   "dataset/rainiest month" \\
        --gauge_dir   rain_gauge/GMIN_SFL_2025 \\
        --output_dir  dataset/processed_T4 \\
        --elev_low    0.5 \\
        --elev_high   0.9 \\
        --elev_tol    0.2 \\
        --val_frac    0.2 \\
        --seed        42

Dependencies
------------
    pip install arm-pyart tqdm numpy pandas
"""

# ---------------------------------------------------------------------------
# Standard library
# ---------------------------------------------------------------------------
import argparse
import json
import logging
import math
import random
import re
import shutil
import sys
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Tuple

# ---------------------------------------------------------------------------
# Third-party
# ---------------------------------------------------------------------------
import numpy as np
import pandas as pd
from tqdm import tqdm

try:
    import pyart
except ImportError:
    sys.exit("[FATAL] arm-pyart not installed.  pip install arm-pyart")

# ---------------------------------------------------------------------------
# DROPS2.0 inspired QC Step 1 constants  (S-band / WSR-88D)
# ---------------------------------------------------------------------------
RHOHV_THRESH   = 0.85
Z_MIN_DBZ      = 5.0

# ---------------------------------------------------------------------------
# Dataset geometry / filtering
# ---------------------------------------------------------------------------
HALF_WIN      = 4
WIN_SIZE      = 2 * HALF_WIN + 1   # 9
N_CHANNELS    = 4
MIN_RATE_MMH  = 1.0
MAX_GAP_MIN   = 15.0

# ---------------------------------------------------------------------------
# Checkpoint settings
# ---------------------------------------------------------------------------
CHECKPOINT_EVERY = 50   # save to disk every this many scan intervals

# ---------------------------------------------------------------------------
# pyart field names
# ---------------------------------------------------------------------------
PYART_Z     = "reflectivity"
PYART_ZDR   = "differential_reflectivity"
PYART_RHOHV = "cross_correlation_ratio"

# KMLB radar (WSR-88D Melbourne, FL)
KMLB_LAT = 28.1133
KMLB_LON = -80.6542

CHANNEL_NAMES = ["Z_low", "ZDR_low", "Z_high", "ZDR_high"]


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def setup_logging(output_dir: Path) -> logging.Logger:
    log = logging.getLogger("qpe_T4")
    log.setLevel(logging.DEBUG)
    fmt = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(fmt)
    log.addHandler(ch)
    fh = logging.FileHandler(output_dir / "build_log.txt", mode="a")  # append
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt)
    log.addHandler(fh)
    return log


# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------

def _ckpt_dir(output_dir: Path) -> Path:
    d = output_dir / "checkpoints"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _find_resume_index(ckpt_dir: Path) -> int:
    """
    Return the highest completed scan-interval index found in checkpoint files.
    Returns -1 if no checkpoints exist (start from scratch).
    """
    max_idx = -1
    for f in ckpt_dir.iterdir():
        if f.suffix in (".npz", ".done"):
            try:
                idx = int(f.stem)
                if idx > max_idx:
                    max_idx = idx
            except ValueError:
                continue
    return max_idx


def _load_all_checkpoints(ckpt_dir: Path,
                           log: logging.Logger
                           ) -> Tuple[List[np.ndarray], List[float], List[dict]]:
    """
    Load every .npz checkpoint file and return merged (all_X, all_y, all_meta).
    .done files are skipped (they contain no sample data).
    """
    all_X:    List[np.ndarray] = []
    all_y:    List[float]      = []
    all_meta: List[dict]       = []

    npz_files = sorted(ckpt_dir.glob("*.npz"))
    if not npz_files:
        return all_X, all_y, all_meta

    log.info("  Loading %d checkpoint file(s) ...", len(npz_files))
    for f in npz_files:
        try:
            data     = np.load(f, allow_pickle=False)
            X_ck     = data["X"]
            y_ck     = data["y"]
            meta_ck  = json.loads(str(data["meta"]))
            all_X.extend(list(X_ck))
            all_y.extend(y_ck.tolist())
            all_meta.extend(meta_ck)
        except Exception as exc:
            log.warning("  Could not load checkpoint %s: %s", f.name, exc)

    log.info("  Recovered %d samples from checkpoints.", len(all_y))
    return all_X, all_y, all_meta


def _save_checkpoint(ckpt_dir:   Path,
                     interval_i: int,
                     batch_X:    List[np.ndarray],
                     batch_y:    List[float],
                     batch_meta: List[dict],
                     log:        logging.Logger) -> None:
    """
    Flush one batch to disk.
    Non-empty batch  -> compressed .npz file.
    Empty batch      -> zero-byte .done marker (records interval as processed).
    """
    stem = f"{interval_i:06d}"
    if batch_X:
        npz_path = ckpt_dir / f"{stem}.npz"
        X_arr    = np.stack(batch_X, axis=0).astype(np.float32)
        y_arr    = np.array(batch_y, dtype=np.float32)
        meta_arr = np.array(json.dumps(batch_meta))
        np.savez_compressed(npz_path, X=X_arr, y=y_arr, meta=meta_arr)
        log.debug("  Checkpoint saved: %s  (%d samples)", npz_path.name, len(batch_y))
    else:
        (ckpt_dir / f"{stem}.done").touch()
        log.debug("  Checkpoint marker: %s  (0 samples)", f"{stem}.done")


# ---------------------------------------------------------------------------
# Geodesy helpers
# ---------------------------------------------------------------------------

def haversine_m(lat1: float, lon1: float,
                lat2: float, lon2: float) -> float:
    R  = 6_371_000.0
    p1 = math.radians(lat1);  p2 = math.radians(lat2)
    dp = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)
    a  = (math.sin(dp / 2) ** 2
          + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2)
    return 2.0 * R * math.asin(math.sqrt(a))


def bearing_deg(lat1: float, lon1: float,
                lat2: float, lon2: float) -> float:
    p1 = math.radians(lat1);  p2 = math.radians(lat2)
    dl = math.radians(lon2 - lon1)
    x  = math.sin(dl) * math.cos(p2)
    y  = math.cos(p1) * math.sin(p2) - math.sin(p1) * math.cos(p2) * math.cos(dl)
    return (math.degrees(math.atan2(x, y)) + 360.0) % 360.0


# ---------------------------------------------------------------------------
# GMIN file parser
# ---------------------------------------------------------------------------

class GminStation:
    __slots__ = (
        "station_id", "tip_mm", "lat", "lon", "radar_id",
        "range_m_computed", "azimuth_deg_computed",
        "df", "total_accum_mm",
    )

    def __init__(self) -> None:
        for s in self.__slots__:
            setattr(self, s, None)


def _parse_gmin_header(line: str) -> dict:
    t = line.split()
    if len(t) < 12 or t[0] != "GMIN":
        raise ValueError(f"Bad GMIN header: {line!r}")
    return {
        "station_id": t[2] + t[3],
        "tip_mm"    : float(t[6]),
        "lat"       : float(t[7]),
        "lon"       : float(t[8]),
        "radar_id"  : t[9],
    }


def _parse_gmin_datarow(line: str) -> Optional[Tuple[datetime, float, int]]:
    t = line.split()
    if len(t) < 11:
        return None
    try:
        dt = datetime(int(t[0]), int(t[1]), int(t[2]),
                      int(t[4]), int(t[5]), int(t[6]))
        return dt, float(t[7]), int(t[10])
    except (ValueError, IndexError):
        return None


def load_gmin_file(fpath: Path) -> Optional[GminStation]:
    try:
        lines = fpath.read_text(errors="replace").splitlines()
    except OSError:
        return None
    if not lines:
        return None

    hdr_line = None
    hdr_idx  = 0
    for i, ln in enumerate(lines):
        if ln.strip().startswith("GMIN"):
            hdr_line = ln.strip()
            hdr_idx  = i
            break
    if hdr_line is None:
        return None

    try:
        meta = _parse_gmin_header(hdr_line)
    except ValueError:
        return None

    records: List[Tuple[datetime, float, int]] = []
    for ln in lines[hdr_idx + 1:]:
        s = ln.strip()
        if not s or s.startswith("#"):
            continue
        row = _parse_gmin_datarow(s)
        if row is not None:
            records.append(row)
    if not records:
        return None

    df = pd.DataFrame(records, columns=["datetime", "precip_mmh", "quality"])
    df = df.sort_values("datetime").reset_index(drop=True)

    # The precip column in GMIN is already a rain RATE in mm/h, not cumulative.
    # Positive values are measured rain rates. The -3.05 value is a dry baseline
    # offset (no rain). quality=1 marks dry/baseline rows; quality=0 or other
    # values mark actual rain measurements.
    rain_rate = np.where(df["precip_mmh"] > 0, df["precip_mmh"], 0.0)
    df["rain_rate_mmh"] = rain_rate.astype(np.float32)

    # Total accumulation: sum of rate * (1/60 h) for each 1-minute rain record
    total_accum = float(np.sum(
        np.where(df["precip_mmh"] > 0, df["precip_mmh"] / 60.0, 0.0)
    ))

    st                       = GminStation()
    st.station_id            = meta["station_id"]
    st.tip_mm                = meta["tip_mm"]
    st.lat                   = meta["lat"]
    st.lon                   = meta["lon"]
    st.radar_id              = meta["radar_id"]
    st.range_m_computed      = haversine_m(
        KMLB_LAT, KMLB_LON, meta["lat"], meta["lon"])
    st.azimuth_deg_computed  = bearing_deg(
        KMLB_LAT, KMLB_LON, meta["lat"], meta["lon"])
    st.df                    = df[["datetime", "rain_rate_mmh"]].copy()
    st.total_accum_mm        = total_accum
    return st


def load_all_gmin(gauge_dir: Path,
                  log: logging.Logger) -> List[GminStation]:
    files = sorted(gauge_dir.glob("*.gmin"))
    if not files:
        raise FileNotFoundError(f"No .gmin files found in {gauge_dir}")

    stations: List[GminStation] = []
    n_bad = n_site = 0
    for f in files:
        st = load_gmin_file(f)
        if st is None:
            log.warning("  Could not parse %s", f.name)
            n_bad += 1
            continue
        if st.radar_id != "KMLB":
            n_site += 1
            continue
        stations.append(st)
        log.debug("  Loaded %-10s  range=%.1fkm  az=%.1f deg  accum=%.1fmm  n=%d",
                  st.station_id, st.range_m_computed / 1e3,
                  st.azimuth_deg_computed, st.total_accum_mm, len(st.df))

    log.info("Gauges: total=%d  bad=%d  non-KMLB=%d  accepted=%d",
             len(files), n_bad, n_site, len(stations))
    return stations


# ---------------------------------------------------------------------------
# NEXRAD L2 inventory
# ---------------------------------------------------------------------------

_L2_RE = re.compile(
    r"^(?P<site>[A-Z]{4})(?P<date>\d{8})_(?P<time>\d{6})_V06(?:_MDM)?$"
)


def parse_l2_filename(name: str) -> Optional[Tuple[str, datetime]]:
    stem = Path(name).stem
    if "_MDM" in stem:
        return None
    m = _L2_RE.match(stem)
    if not m:
        return None
    ts = datetime.strptime(m.group("date") + m.group("time"), "%Y%m%d%H%M%S")
    return m.group("site"), ts


def build_l2_inventory(radar_dir: Path,
                       log: logging.Logger) -> List[Tuple[datetime, Path]]:
    entries: List[Tuple[datetime, Path]] = []
    for f in sorted(radar_dir.iterdir()):
        result = parse_l2_filename(f.name)
        if result is None:
            continue
        site, ts = result
        if site != "KMLB":
            continue
        entries.append((ts, f))
    entries.sort(key=lambda x: x[0])
    if not entries:
        raise FileNotFoundError(
            f"No KMLB NEXRAD L2 V06 files found in {radar_dir}")
    log.info("Found %d KMLB L2 scans  (%s  -->  %s)",
             len(entries),
             entries[0][0].strftime("%Y-%m-%d %H:%M:%S"),
             entries[-1][0].strftime("%Y-%m-%d %H:%M:%S"))
    return entries


# ---------------------------------------------------------------------------
# DROPS2.0 inspired  Step 1 -- non-met echo removal
# ---------------------------------------------------------------------------

def _drops2_nonmet_mask(Z_2d: np.ndarray,
                        rhohv_2d: np.ndarray) -> np.ndarray:
    """
    Return boolean mask (True = non-meteorological gate).
    Criteria: rhoHV < RHOHV_THRESH  OR  Z < Z_MIN_DBZ  OR  either is NaN.
    """
    bad_rhohv = ~np.isfinite(rhohv_2d) | (rhohv_2d < RHOHV_THRESH)
    bad_z     = ~np.isfinite(Z_2d)     | (Z_2d     < Z_MIN_DBZ)
    return bad_rhohv | bad_z


def _apply_nonmet_mask_to_sweep(radar,
                                sweep_idx: int,
                                mask_2d: np.ndarray) -> None:
    """Apply non-met mask to Z and ZDR in-place for one sweep."""
    sl = radar.get_slice(sweep_idx)
    for field_name in [PYART_Z, PYART_ZDR]:
        if field_name not in radar.fields:
            continue
        data  = radar.fields[field_name]["data"]
        dense = np.ma.filled(data[sl], fill_value=np.nan).copy()
        dense[mask_2d] = np.nan
        data[sl] = np.ma.masked_invalid(dense)


# ---------------------------------------------------------------------------
# Sweep cache
# ---------------------------------------------------------------------------

class SweepCache:
    """
    Holds DROPS2.0 inspired QC Step-1-filtered Z and ZDR arrays for two sweeps.
    All field arrays: shape (n_rays, n_gates), dtype float32, NaN = masked.
    """
    __slots__ = (
        "timestamp",
        "Z_low",  "ZDR_low",
        "Z_high", "ZDR_high",
        "az_low",  "rng_low",
        "az_high", "rng_high",
    )

    def __init__(self) -> None:
        for s in self.__slots__:
            setattr(self, s, None)


def _find_best_sweep(radar, target_elev: float,
                     elev_tol: float) -> Optional[int]:
    fixed = radar.fixed_angle["data"]
    diffs = np.abs(fixed - target_elev)
    best  = int(np.argmin(diffs))
    return best if diffs[best] <= elev_tol else None


def _extract_sweep_arrays(radar, sweep_idx: int
                          ) -> Tuple[np.ndarray, np.ndarray,
                                     np.ndarray, np.ndarray]:
    """Return (Z, ZDR, azimuths, ranges) for one sweep, all float32."""
    sl  = radar.get_slice(sweep_idx)
    az  = radar.azimuth["data"][sl].astype(np.float32)
    rng = radar.range["data"].astype(np.float32)

    def to_f32(name: str) -> np.ndarray:
        return np.ma.filled(
            radar.fields[name]["data"][sl], fill_value=np.nan
        ).astype(np.float32)

    return to_f32(PYART_Z), to_f32(PYART_ZDR), az, rng


def load_sweep_cache(fpath: Path, ts: datetime,
                     elev_low: float, elev_high: float,
                     elev_tol: float,
                     log: logging.Logger) -> Optional[SweepCache]:
    """
    Load one NEXRAD L2 scan, apply DROPS2.0 inspired QC Step 1, cache Z and ZDR.

    Steps:
      1. Read file with pyart.
      2. Verify Z, ZDR, rhoHV are present.
      3. Find low and high elevation sweeps.
      4. DROPS2.0 Step 1: mask non-met gates in Z and ZDR.
      5. Extract float32 arrays for both sweeps.

    Returns SweepCache or None on any failure.
    """
    try:
        radar = pyart.io.read_nexrad_archive(str(fpath))
    except Exception as exc:
        log.warning("  Cannot read %s: %s", fpath.name, exc)
        return None

    for req in [PYART_Z, PYART_ZDR, PYART_RHOHV]:
        if req not in radar.fields:
            log.debug("  %s: missing field %r -- skip", fpath.name, req)
            return None

    sw_low  = _find_best_sweep(radar, elev_low,  elev_tol)
    sw_high = _find_best_sweep(radar, elev_high, elev_tol)
    if sw_low is None:
        log.debug("  %s: no sweep near %.2f deg -- skip", fpath.name, elev_low)
        return None
    if sw_high is None:
        log.debug("  %s: no sweep near %.2f deg -- skip", fpath.name, elev_high)
        return None

    for sw_idx in [sw_low, sw_high]:
        sl       = radar.get_slice(sw_idx)
        Z_sw     = np.ma.filled(
            radar.fields[PYART_Z]["data"][sl],     np.nan).astype(np.float32)
        rhohv_sw = np.ma.filled(
            radar.fields[PYART_RHOHV]["data"][sl], np.nan).astype(np.float32)
        mask = _drops2_nonmet_mask(Z_sw, rhohv_sw)
        _apply_nonmet_mask_to_sweep(radar, sw_idx, mask)
        log.debug("  %s sw%d Step1: nonmet=%d/%d (%.1f%%)",
                  fpath.name, sw_idx,
                  int(mask.sum()), int(mask.size),
                  100.0 * mask.sum() / mask.size)

    Z_l,  ZDR_l,  az_l, rng_l = _extract_sweep_arrays(radar, sw_low)
    Z_h,  ZDR_h,  az_h, rng_h = _extract_sweep_arrays(radar, sw_high)

    cache           = SweepCache()
    cache.timestamp = ts
    cache.Z_low     = Z_l;   cache.ZDR_low  = ZDR_l
    cache.Z_high    = Z_h;   cache.ZDR_high = ZDR_h
    cache.az_low    = az_l;  cache.rng_low  = rng_l
    cache.az_high   = az_h;  cache.rng_high = rng_h
    return cache


# ---------------------------------------------------------------------------
# 9x9 window extraction
# ---------------------------------------------------------------------------

def _nearest_ray_gate(azimuths: np.ndarray, ranges: np.ndarray,
                      target_az: float,
                      target_range_m: float) -> Tuple[int, int]:
    az_diff  = np.abs(((azimuths - target_az) + 180.0) % 360.0 - 180.0)
    ray_idx  = int(np.argmin(az_diff))
    gate_idx = int(np.argmin(np.abs(ranges - target_range_m)))
    return ray_idx, gate_idx


def extract_window_2d(field: np.ndarray,
                      ray_idx: int, gate_idx: int,
                      half_win: int = HALF_WIN) -> Optional[np.ndarray]:
    """
    9x9 patch centred at (ray_idx, gate_idx).
    Rays wrap circularly. Gates do not wrap.
    Returns None if window extends beyond range boundary or contains any NaN.
    """
    n_rays, n_gates = field.shape
    if gate_idx < half_win or gate_idx + half_win + 1 > n_gates:
        return None
    ray_indices = (
        np.arange(ray_idx - half_win, ray_idx + half_win + 1) % n_rays
    )
    patch = field[ray_indices, :][
        :, (gate_idx - half_win):(gate_idx + half_win + 1)
    ]
    if np.any(~np.isfinite(patch)):
        return None
    return patch.astype(np.float32)


# ---------------------------------------------------------------------------
# Gauge query
# ---------------------------------------------------------------------------

def gauge_mean_rate(df: pd.DataFrame,
                    t_start: datetime,
                    t_end: datetime) -> Optional[float]:
    mask = (df["datetime"] >= t_start) & (df["datetime"] <= t_end)
    sub  = df.loc[mask, "rain_rate_mmh"]
    return float(sub.mean()) if not sub.empty else None


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def build_dataset(radar_dir:  Path,
                  gauge_dir:  Path,
                  output_dir: Path,
                  elev_low:   float,
                  elev_high:  float,
                  elev_tol:   float,
                  val_frac:   float,
                  seed:       int,
                  log:        logging.Logger) -> None:

    # 1 -- Load gauges
    log.info("=== Step 1: Loading GMIN gauge files ===")
    stations = load_all_gmin(gauge_dir, log)
    if not stations:
        raise RuntimeError("No usable KMLB gauge stations after QC.")

    # 2 -- Radar inventory
    log.info("=== Step 2: Building NEXRAD L2 scan inventory ===")
    scan_list = build_l2_inventory(radar_dir, log)

    # 3 -- Checkpoint resume logic
    ckpt_dir   = _ckpt_dir(output_dir)
    resume_idx = _find_resume_index(ckpt_dir)

    if resume_idx >= 0:
        log.info("=== RESUMING from checkpoint (last completed interval: %d) ===",
                 resume_idx)
        all_X, all_y, all_meta = _load_all_checkpoints(ckpt_dir, log)
        log.info("  Recovered %d samples. Skipping intervals 0..%d.",
                 len(all_y), resume_idx)
    else:
        log.info("=== No checkpoints found. Starting fresh. ===")
        all_X, all_y, all_meta = [], [], []

    start_i = resume_idx + 1

    # 4 -- Sample extraction
    log.info("=== Step 3: DROPS2.0 inspired Step1 (Z/rhoHV masking) + sample extraction ===")
    log.info("    Variables: Z, ZDR  |  Tensor: T_{4x9x9}  |  "
             "RHOHV_THRESH=%.2f  Z_MIN=%.1f dBZ",
             RHOHV_THRESH, Z_MIN_DBZ)
    log.info("    Processing intervals %d to %d  (%d remaining)",
             start_i, len(scan_list) - 2,
             max(0, len(scan_list) - 1 - start_i))
    log.info("    Checkpoint every %d intervals (~every 2-3 min)",
             CHECKPOINT_EVERY)

    cnt = dict(total=0, no_radar=0, bad_window=0,
               no_gauge=0, rate_filter=0, accepted=0)

    # In-progress batch buffer (flushed every CHECKPOINT_EVERY intervals)
    batch_X:    List[np.ndarray] = []
    batch_y:    List[float]      = []
    batch_meta: List[dict]       = []

    total_intervals = len(scan_list) - 1
    progress = tqdm(range(start_i, total_intervals),
                    desc="Scan intervals", unit="interval",
                    initial=start_i, total=total_intervals)

    for i in progress:
        ts_cur,  fpath_cur = scan_list[i]
        ts_next, _         = scan_list[i + 1]

        gap_min = (ts_next - ts_cur).total_seconds() / 60.0
        if gap_min <= MAX_GAP_MIN:
            cache = load_sweep_cache(
                fpath_cur, ts_cur, elev_low, elev_high, elev_tol, log)
            if cache is None:
                cnt["no_radar"] += len(stations)
            else:
                for st in stations:
                    cnt["total"] += 1

                    mean_rate = gauge_mean_rate(st.df, ts_cur, ts_next)
                    if mean_rate is None:
                        cnt["no_gauge"] += 1
                        continue
                    if mean_rate < MIN_RATE_MMH:
                        cnt["rate_filter"] += 1
                        continue

                    ray_l, gate_l = _nearest_ray_gate(
                        cache.az_low, cache.rng_low,
                        st.azimuth_deg_computed, st.range_m_computed)
                    ray_h, gate_h = _nearest_ray_gate(
                        cache.az_high, cache.rng_high,
                        st.azimuth_deg_computed, st.range_m_computed)

                    windows: List[np.ndarray] = []
                    ok = True
                    for (field, ri, gi) in [
                        (cache.Z_low,    ray_l, gate_l),
                        (cache.ZDR_low,  ray_l, gate_l),
                        (cache.Z_high,   ray_h, gate_h),
                        (cache.ZDR_high, ray_h, gate_h),
                    ]:
                        win = extract_window_2d(field, ri, gi)
                        if win is None:
                            ok = False
                            break
                        windows.append(win)

                    if not ok:
                        cnt["bad_window"] += 1
                        continue

                    batch_X.append(np.stack(windows, axis=0))   # (4, 9, 9)
                    batch_y.append(mean_rate)
                    batch_meta.append({
                        "scan_time"    : ts_cur.strftime("%Y-%m-%d %H:%M:%S"),
                        "next_time"    : ts_next.strftime("%Y-%m-%d %H:%M:%S"),
                        "gap_min"      : round(gap_min, 2),
                        "station_id"   : st.station_id,
                        "lat"          : st.lat,
                        "lon"          : st.lon,
                        "range_m"      : round(st.range_m_computed, 1),
                        "azimuth_deg"  : round(st.azimuth_deg_computed, 2),
                        "rain_rate_mmh": round(mean_rate, 4),
                    })
                    cnt["accepted"] += 1

        # ── Checkpoint flush ─────────────────────────────────────────────────
        is_last     = (i == total_intervals - 1)
        is_ckpt_due = ((i + 1) % CHECKPOINT_EVERY == 0) or is_last

        if is_ckpt_due:
            _save_checkpoint(ckpt_dir, i, batch_X, batch_y, batch_meta, log)
            all_X.extend(batch_X)
            all_y.extend(batch_y)
            all_meta.extend(batch_meta)
            batch_X.clear()
            batch_y.clear()
            batch_meta.clear()
            progress.set_postfix({
                "accepted": len(all_y),
                "radar_t" : ts_cur.strftime("%m-%d %H:%M"),
            })
            log.info("  [interval %d/%d]  radar=%s  accepted_total=%d",
                     i, total_intervals - 1,
                     ts_cur.strftime("%Y-%m-%d %H:%M:%S UTC"),
                     len(all_y))

    # 5 -- Summary
    log.info("=" * 62)
    log.info("Extraction complete.")
    log.info("  Candidates (interval x station) : %8d", cnt["total"])
    log.info("  Rejected -- missing/bad radar   : %8d", cnt["no_radar"])
    log.info("  Rejected -- NaN in 9x9 window   : %8d", cnt["bad_window"])
    log.info("  Rejected -- no gauge data       : %8d", cnt["no_gauge"])
    log.info("  Rejected -- rate < %.1f mm/h     : %8d",
             MIN_RATE_MMH, cnt["rate_filter"])
    log.info("  Accepted (this run)             : %8d", cnt["accepted"])
    log.info("  Total accepted (incl. resumed)  : %8d", len(all_y))
    log.info("=" * 62)

    if len(all_y) == 0:
        raise RuntimeError(
            "Zero samples accepted (including any resumed checkpoints).\n"
            "Common causes:\n"
            "  1. Wrong --radar_dir or --gauge_dir paths.\n"
            "  2. Elevation mismatch: try --elev_tol 0.3.\n"
            "  3. Missing ZDR or rhoHV fields in NEXRAD files.\n"
            "  4. All gauge rates below MIN_RATE_MMH.\n"
        )

    # 6 -- Build arrays
    X = np.stack(all_X, axis=0).astype(np.float32)   # (N, 4, 9, 9)
    y = np.array(all_y,  dtype=np.float32)            # (N,)
    log.info("Final array shapes: X=%s  y=%s", X.shape, y.shape)

    # 7 -- Train/val split by scan-time (no temporal leakage)
    unique_times = sorted({m["scan_time"] for m in all_meta})
    rng = random.Random(seed)
    rng.shuffle(unique_times)
    n_val       = max(1, int(round(len(unique_times) * val_frac)))
    val_times   = set(unique_times[:n_val])
    train_times = set(unique_times[n_val:])

    train_idx = [i for i, m in enumerate(all_meta)
                 if m["scan_time"] in train_times]
    val_idx   = [i for i, m in enumerate(all_meta)
                 if m["scan_time"] in val_times]

    X_train, y_train = X[train_idx], y[train_idx]
    X_val,   y_val   = X[val_idx],   y[val_idx]
    log.info("Train: %d samples (%d scan-times)  |  Val: %d samples (%d scan-times)",
             len(train_idx), len(train_times), len(val_idx), len(val_times))

    # 8 -- Channel statistics (train only)
    stats: dict = {}
    for c, name in enumerate(CHANNEL_NAMES):
        v = X_train[:, c, :, :].ravel()
        stats[name] = dict(
            mean=round(float(np.mean(v)),           6),
            std =round(float(np.std(v)),            6),
            min =round(float(np.min(v)),            6),
            max =round(float(np.max(v)),            6),
            p05 =round(float(np.percentile(v,  5)), 6),
            p95 =round(float(np.percentile(v, 95)), 6),
        )
    v = y_train
    stats["rain_rate_mmh"] = dict(
        mean=round(float(np.mean(v)),           6),
        std =round(float(np.std(v)),            6),
        min =round(float(np.min(v)),            6),
        max =round(float(np.max(v)),            6),
        p05 =round(float(np.percentile(v,  5)), 6),
        p95 =round(float(np.percentile(v, 95)), 6),
    )
    log.info("Channel statistics (train):")
    for name, s in stats.items():
        log.info("  %-20s  mean=%9.4f  std=%9.4f  min=%9.4f  max=%9.4f",
                 name, s["mean"], s["std"], s["min"], s["max"])

    # 9 -- Save final outputs
    output_dir.mkdir(parents=True, exist_ok=True)
    np.save(output_dir / "X_train.npy", X_train)
    np.save(output_dir / "y_train.npy", y_train)
    np.save(output_dir / "X_val.npy",   X_val)
    np.save(output_dir / "y_val.npy",   y_val)

    meta_df          = pd.DataFrame(all_meta)
    meta_df["split"] = ["train" if m["scan_time"] in train_times else "val"
                         for m in all_meta]
    meta_df[meta_df["split"] == "train"].to_csv(
        output_dir / "metadata_train.csv", index=False)
    meta_df[meta_df["split"] == "val"].to_csv(
        output_dir / "metadata_val.csv",   index=False)

    # 10 -- Dataset manifest
    stats["build_info"] = {
        "n_total"        : int(len(all_y)),
        "n_train"        : int(len(train_idx)),
        "n_val"          : int(len(val_idx)),
        "val_frac"       : val_frac,
        "seed"           : seed,
        "elev_low_deg"   : elev_low,
        "elev_high_deg"  : elev_high,
        "elev_tol_deg"   : elev_tol,
        "min_rate_mmh"   : MIN_RATE_MMH,
        "tensor_shape"   : [N_CHANNELS, WIN_SIZE, WIN_SIZE],
        "channel_order"  : CHANNEL_NAMES,
        "radar_station"  : "KMLB",
        "qc_algorithm"   : "DROPS2.0 inspired Step 1 (Chen & Chandrasekar, "
                           "J. Hydrometeorol. 2017) -- rhoHV/Z masking only",
        "variables"      : "Z, ZDR",
        "drops2_settings": {
            "rhohv_thresh": RHOHV_THRESH,
            "z_min_dbz"   : Z_MIN_DBZ,
        },
        "checkpoint_every": CHECKPOINT_EVERY,
        "model_pipeline" : (
            "CNN (spatial features) -> "
            "KAN (non-linear R regression) -> "
            "LightGBM (residual correction)"
        ),
    }
    with open(output_dir / "dataset_stats.json", "w") as f:
        json.dump(stats, f, indent=2)

    log.info("Final outputs saved to %s", output_dir)

    # 11 -- Clean up checkpoints after successful save
    try:
        shutil.rmtree(ckpt_dir)
        log.info("Checkpoint directory removed (clean run complete).")
    except Exception as exc:
        log.warning("Could not remove checkpoint dir: %s", exc)

    log.info("Done.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Build T_{4x9x9} QPE dataset from NEXRAD L2 + GMIN gauges. "
            "Variables: Z, ZDR. DROPS2.0 inspired  Step 1 QC. Checkpoint/resume support."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--radar_dir",  type=Path,
                   default=Path("../data/dataset/rainiest month"))
    p.add_argument("--gauge_dir",  type=Path,
                   default=Path("../data/dataset/rain_gauge"))
    p.add_argument("--output_dir", type=Path,
                   default=Path("../data/dataset/processed_T4"))
    p.add_argument("--elev_low",   type=float, default=0.5)
    p.add_argument("--elev_high",  type=float, default=0.9)
    p.add_argument("--elev_tol",   type=float, default=0.2)
    p.add_argument("--val_frac",   type=float, default=0.2)
    p.add_argument("--seed",       type=int,   default=42)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    log = setup_logging(args.output_dir)
    log.info("QPE Dataset Builder (T_{4x9x9} / Z+ZDR / checkpoint-resume) started.")
    log.info("Config: %s", vars(args))
    build_dataset(
        radar_dir  = args.radar_dir,
        gauge_dir  = args.gauge_dir,
        output_dir = args.output_dir,
        elev_low   = args.elev_low,
        elev_high  = args.elev_high,
        elev_tol   = args.elev_tol,
        val_frac   = args.val_frac,
        seed       = args.seed,
        log        = log,
    )


if __name__ == "__main__":
    main()