"""Tracklet generator
"""
import csv
import json
import math
import random
import re
import warnings
import logging
from pathlib import Path

import numpy as np
import torch
from scipy.ndimage import map_coordinates
from astropy.io import fits
from astropy.wcs import WCS
from torch.utils.data import Dataset

from config import IMAGE_SIZE

_NONSTANDARD_HEADER_PREFIX = (
    "The following header keyword is invalid or follows an unrecognized non-standard convention:"
)
warnings.filterwarnings(
    "ignore",
    message=r"The following header keyword is invalid or follows an unrecognized non-standard convention:.*",
    category=Warning,
)
_fits_card_logger = logging.getLogger("astropy.io.fits.card")
_fits_card_logger.addFilter(_IgnoreNonstandardDecamHeaderCards())

class _IgnoreNonstandardDecamHeaderCards(logging.Filter):
    def filter(self, record):
        return not record.getMessage().startswith(_NONSTANDARD_HEADER_PREFIX)


def _float(v):
    try:
        return float(v)
    except Exception:
        return None


def read_metadata(folder):
    p = folder / "metadata.csv"
    if not p.exists():
        return []
    with p.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def read_json(folder):
    p = folder / "tracklet.json"
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _frame_number(path):
    match = re.search(r"frame_(\d+)", path.name, flags=re.IGNORECASE)
    return int(match.group(1)) if match else 10**12


def raw_files(folder):
    """Find downloaded FITS/FITS-tile-compressed exposures, in numeric order."""
    r = folder / "raw"
    if not r.exists():
        return []
    files = [p for p in r.iterdir() if p.is_file() and (
        p.name.lower().endswith(".fits.fz") or p.name.lower().endswith(".fits")
    )]
    return sorted(files, key=lambda p: (_frame_number(p), p.name.lower()))


def _check_fits_signature(path):
    """Fail before Astropy can emit a huge warning storm on binary garbage."""
    with Path(path).open("rb") as f:
        first = f.read(80)
    if not first.startswith(b"SIMPLE  "):
        # A normal .fits.fz tile-compressed FITS file still starts with a FITS
        # primary header. It is not a gzip stream or a raw binary blob.
        raise ValueError(
            f"Not a standard FITS primary header: {path}; "
            f"first 16 bytes={first[:16]!r}. Expected a .fits.fz FITS file."
        )


def _fits_open(path, **kwargs):
    """Open FITS/FITS.FZ while converting malformed-card warning storms to one error."""
    _check_fits_signature(path)
    # Parse all HDU headers while warnings are captured; do not defer header
    # parsing until the caller iterates the HDUList, which could recreate a storm.
    kwargs["lazy_load_hdus"] = False
    # DECam archive products can contain non-standard long header cards that
    # Astropy warns about while still reading the image HDUs correctly. Do not
    # turn that warning into a hard rejection; later HDU/WCS checks determine
    # whether this exposure is actually usable.
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=r"The following header keyword is invalid or follows an unrecognized non-standard convention:.*",
            category=Warning,
        )
        warnings.filterwarnings(
            "ignore",
            message=r".*: Q$",
            category=Warning,
            module=r"astropy\.io\.fits\.card",
        )
        return fits.open(path, **kwargs)


def primary_product_type(path):
    """Return normalized primary-header product type (e.g. image, dqmask)."""
    with _fits_open(path, mode="readonly", memmap=True, lazy_load_hdus=False) as hdul:
        primary = hdul[0].header
        return {
            "prodtype": str(primary.get("PRODTYPE", "")).strip().lower(),
            "proctype": str(primary.get("PROCTYPE", "")).strip().lower(),
            "obstype": str(primary.get("OBSTYPE", "")).strip().lower(),
            "object": str(primary.get("OBJECT", "")).strip(),
        }


def require_science_image(path):
    """Reject DQ masks/weight maps before any image crop or injection is made."""
    product = primary_product_type(path)
    prodtype = product["prodtype"]
    if prodtype != "image":
        raise ValueError(
            f"not a science image: PRODTYPE={prodtype or '<missing>'!r} "
            f"(PROCTYPE={product['proctype'] or '<missing>'!r}, "
            f"OBJECT={product['object'] or '<missing>'!r}); expected PRODTYPE='image'"
        )
    if product["obstype"] and product["obstype"] != "object":
        raise ValueError(
            f"not an object exposure: OBSTYPE={product['obstype']!r}; expected 'object'"
        )
    return product


def list_image_hdus(path):
    """List 2-D image HDUs without loading pixel arrays.

    SCI extensions are preferred over generic/auxiliary image extensions.
    The HDU index is retained because labels must be checked against the
    exact image plane that will be cropped.
    """
    require_science_image(path)
    result = []
    with _fits_open(
        path, mode="readonly", memmap=True, lazy_load_hdus=False,
        do_not_scale_image_data=True
    ) as hdul:
        for i, hdu in enumerate(hdul):
            header = hdu.header
            naxis = int(header.get("NAXIS", 0) or 0)
            w = int(header.get("NAXIS1", 0) or 0)
            h = int(header.get("NAXIS2", 0) or 0)
            if naxis < 2 or w <= 1 or h <= 1:
                continue
            extname = str(header.get("EXTNAME", "PRIMARY" if i == 0 else "")).strip().upper()
            priority = 0 if extname in {"SCI", "SCIENCE"} else (1 if i == 0 else 2)
            result.append({"index": i, "width": w, "height": h,
                           "extname": extname or f"HDU{i}", "priority": priority,
                           "header": header.copy()})
    result.sort(key=lambda x: (x["priority"], x["index"]))
    if not result:
        raise ValueError(f"No 2-D image HDU found in FITS exposure: {path}")
    return result


def image_hdu(path):
    """Backward-compatible: return preferred 2-D image HDU index, width, height."""
    h = list_image_hdus(path)[0]
    return h["index"], h["width"], h["height"]


def read_crop(path, x0, y0, size, hdu_index=None, clip=True):
    """Read a pure, unpadded crop from .fits or tile-compressed .fits.fz.

    Uses Astropy's HDU.section interface so compressed FITS can decompress only
    the requested region where the format supports it. No resizing or padding.
    """
    if hdu_index is None:
        idx, w, h = image_hdu(path)
    else:
        candidates = {q["index"]: q for q in list_image_hdus(path)}
        if int(hdu_index) not in candidates:
            raise ValueError(f"Selected HDU {hdu_index} is not a 2-D image in {path}")
        idx = int(hdu_index)
        w, h = candidates[idx]["width"], candidates[idx]["height"]
    x0i, y0i = int(round(float(x0))), int(round(float(y0)))
    if abs(float(x0) - x0i) > 1e-6 or abs(float(y0) - y0i) > 1e-6:
        raise ValueError(f"Crop origin must be integer pixels, got ({x0},{y0}): {path}")
    if x0i < 0 or y0i < 0 or x0i + size > w or y0i + size > h:
        raise ValueError(
            f"Pure {size}x{size} crop outside image bounds "
            f"(image={w}x{h}, origin=({x0i},{y0i})): {path}"
        )

    with _fits_open(path, mode="readonly", memmap=True, lazy_load_hdus=True) as hdul:
        hdu = hdul[idx]
        if not hasattr(hdu, "section"):
            raise ValueError(f"HDU does not support safe section reads: {path} (HDU {idx})")
        crop = np.asarray(hdu.section[y0i:y0i + size, x0i:x0i + size], dtype=np.float32)

    if crop.shape != (size, size):
        raise ValueError(f"Unexpected crop shape {crop.shape}; expected {(size,size)}: {path}")
    crop = np.nan_to_num(crop, nan=0.0, posinf=0.0, neginf=0.0)
    med = float(np.median(crop))
    mad = float(np.median(np.abs(crop - med))) * 1.4826 + 1e-6
    crop = (crop - med) / mad  # defer clipping until after injection/reprojection
    return crop.astype(np.float32)


"""Synthetic moving-source supervision on real DECam FITS backgrounds.

Training targets are generated from the injected source coordinates. Existing DAD
asteroid x/y labels are deliberately not read or used by this dataset.
"""
import csv
import json
import math
import random
import re
import warnings
from pathlib import Path
from config import IMAGE_SIZE
import numpy as np
import torch
from scipy.ndimage import map_coordinates
from astropy.io import fits
from astropy.wcs import WCS
from torch.utils.data import Dataset


def _norm_key(k):
    return str(k or "").strip().lower().replace(" ", "_").replace("-", "_")


def _walk_dicts(obj):
    if isinstance(obj, dict):
        yield obj
        for v in obj.values():
            yield from _walk_dicts(v)
    elif isinstance(obj, list):
        for v in obj:
            yield from _walk_dicts(v)


def _lookup(d, names):
    normalized = {_norm_key(k): v for k, v in d.items()}
    for name in names:
        value = normalized.get(_norm_key(name))
        if value not in (None, ""):
            try:
                f = float(value)
                if math.isfinite(f):
                    return f
            except (TypeError, ValueError):
                continue
    return None


def _pixel_scale_arcsec(header):
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            w = WCS(header, relax=True).celestial
        if not w.has_celestial:
            return None
        # Approximate local pixel scale from the celestial WCS matrix.
        from astropy.wcs.utils import proj_plane_pixel_scales
        scales = np.asarray(proj_plane_pixel_scales(w), dtype=float) * 3600.0
        scales = scales[np.isfinite(scales) & (scales > 0)]
        return float(np.mean(scales)) if scales.size else None
    except Exception:
        return None


def _extract_motion(folder, rows, json_data, headers, cfg):
    """Estimate synthetic track velocity without reading asteroid x/y or RA/Dec.

    Uses explicitly pixel/frame velocity components when present. Otherwise it
    uses angular rate + position angle and the WCS pixel scale. Explicit
    *_arcsec_min fields are arcsec/min; generic archive `rate` is treated as
    arcsec/hour (the usual DECam moving-object catalog convention). If no
    trustworthy motion metadata exists, draw a conservative random velocity.
    """
    sources = list(_walk_dicts(json_data)) + list(rows)
    for d in sources:
        vx = _lookup(d, ["vx_px_frame", "vx_pix_frame", "velocity_x_px_frame", "vx_pixels_per_frame"])
        vy = _lookup(d, ["vy_px_frame", "vy_pix_frame", "velocity_y_px_frame", "vy_pixels_per_frame"])
        if vx is not None and vy is not None:
            mag = math.hypot(vx, vy)
            if 0.03 <= mag <= float(cfg.get("max_velocity_px_frame", 8.0)):
                return (vx, vy), "metadata_pixel_components"

    rate = pa = None
    unit = None
    for d in sources:
        # Do not read x/y, RA, or Dec fields: these labels are known to be unreliable.
        r_min = _lookup(d, ["rate_arcsec_min", "rate_as_min", "motion_rate_arcsec_min"])
        r_hour = _lookup(d, ["rate_arcsec_hour", "rate_as_hour", "rate_as_hr", "motion_rate_arcsec_hour"])
        generic = _lookup(d, ["rate", "motion_rate", "sky_rate"])
        p_angle = _lookup(d, ["pa_deg", "position_angle_deg", "position_angle", "pa"])
        if p_angle is None:
            continue
        if r_min is not None and r_min > 0:
            rate, unit, pa = r_min, "arcsec_min", p_angle
            break
        if r_hour is not None and r_hour > 0:
            rate, unit, pa = r_hour, "arcsec_hour", p_angle
            break
        if generic is not None and generic > 0:
            rate, unit, pa = generic, str(cfg.get("generic_rate_unit", "arcsec_hour")), p_angle
            break

    if rate is not None and pa is not None and headers:
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                w = WCS(headers[0], relax=True).celestial
            scale = _pixel_scale_arcsec(headers[0])
            if w.has_celestial and scale and scale > 0:
                mjds = [_lookup(r, ["mjdobs", "mjd_obs", "mjd", "mjd-obs"]) for r in rows]
                deltas_min = [abs((b-a) * 1440.0) for a, b in zip(mjds, mjds[1:])
                              if a is not None and b is not None and a != b]
                dt_min = float(np.median(deltas_min)) if deltas_min else float(cfg.get("fallback_cadence_min", 2.0))
                rate_as_min = rate if unit == "arcsec_min" else rate / 60.0
                speed = rate_as_min * dt_min / scale
                # Position angle is sky-relative; convert through local WCS axes.
                theta = math.radians(pa)
                dec0 = float(headers[0].get("CRVAL2", 0.0))
                cos_dec = max(1e-4, abs(math.cos(math.radians(dec0))))
                world_delta = np.array([
                    rate_as_min * dt_min * math.sin(theta) / (3600.0 * cos_dec),
                    rate_as_min * dt_min * math.cos(theta) / 3600.0,
                ], dtype=float)
                matrix = np.asarray(w.pixel_scale_matrix, dtype=float)
                if matrix.shape == (2, 2) and np.all(np.isfinite(matrix)) and abs(np.linalg.det(matrix)) > 1e-12:
                    vx, vy = np.linalg.solve(matrix, world_delta)
                    mag = math.hypot(float(vx), float(vy))
                    if 0.03 <= mag <= float(cfg.get("max_velocity_px_frame", 8.0)):
                        return (float(vx), float(vy)), f"metadata_rate_pa_wcs_{unit}"
        except Exception:
            pass

    # Fallback deliberately spans slow and faster movers, concentrated at low
    # pixel/frame speeds rather than the previous arbitrary 1-5 px/frame range.
    lo, hi = cfg.get("fallback_speed_px_frame_range", (0.08, 1.8))
    speed = math.exp(random.uniform(math.log(float(lo)), math.log(float(hi))))
    angle = random.uniform(0.0, 2.0 * math.pi)
    return (speed * math.cos(angle), speed * math.sin(angle)), "synthetic_fallback_realistic_range"


def _wcs_for_header(header):
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            w = WCS(header, relax=True).celestial
        return w if w.has_celestial else None
    except Exception:
        return None


def _sample_brightness(cfg, rng=None):
    """Sample a broad faint-to-bright range in robust-normalized image units."""
    lo, hi = cfg.get("injection_amplitude_range", (0.35, 8.0))
    lo, hi = max(float(lo), 1e-3), max(float(hi), float(lo) + 1e-3)
    # Log-uniform gives faint sources plenty of representation without starving
    # the brighter, easy-to-localize examples.
    rng = rng or random
    return math.exp(rng.uniform(math.log(lo), math.log(hi)))

def _ccd_text(value):
    if value is None:
        return ""
    text = str(value).strip().upper()
    # CSVs sometimes store a numeric CCD identifier as e.g. 29.0.
    if re.fullmatch(r"\d+\.0+", text):
        text = text.split(".", 1)[0]
    return re.sub(r"\s+", "", text)


def _row_ccd(row):
    """Return (metadata field, normalized identifier), prioritizing explicit CCS.

    In this dataset `ccs` is the detector/CCD selector supplied by metadata.
    Do not silently prefer another similarly named column when `ccs` exists.
    """
    aliases = (
        "ccs", "ccd", "ccd_name", "ccdname", "ccd_id", "ccdid", "ccd_num",
        "ccdnum", "ccd_number", "detector", "detector_name", "chip",
        "chip_name", "extname", "extension", "hdu_name"
    )
    normalized = {_norm_key(k): v for k, v in (row or {}).items()}
    for key in aliases:
        value = normalized.get(_norm_key(key))
        if value is not None and str(value).strip():
            return key, _ccd_text(value)
    return "", ""


def _header_values_by_key(h):
    """Preserve FITS header key provenance; numeric IMAGEID is not a CCD ID."""
    header = h.get("header", {})
    groups = {
        "ccs": ("CCS",),
        "explicit": ("CCDNUM", "CCD_NUM", "CCDID", "CCD", "DETECTOR", "CHIP"),
        "names": ("EXTNAME", "CCDNAME", "CHIPNAME", "DETECTOR_NAME"),
    }
    out = {name: [] for name in groups}
    for group, keys in groups.items():
        for key in keys:
            value = header.get(key)
            if value is not None and str(value).strip():
                out[group].append(_ccd_text(value))
    # EXTNAME can be carried outside the header dict by synthetic/test records.
    if not out["names"] and h.get("extname"):
        out["names"].append(_ccd_text(h["extname"]))
    return out


def _numeric_ccd_equal(requested, value):
    """Compare numeric IDs without matching unrelated header values."""
    req = re.fullmatch(r"(?:CCD|S|N)?0*(\d+)", requested)
    val = re.fullmatch(r"(?:CCD|S|N)?0*(\d+)", value)
    return bool(req and val and int(req.group(1)) == int(val.group(1)))


def _ccd_matches(requested, h, metadata_field=""):
    groups = _header_values_by_key(h)
    # If metadata explicitly names CCS, only match the CCS header when it is
    # present anywhere in the file. This prevents CCDNUM/EXTNAME/IMAGEID values
    # from creating several false matches (the previous bug).
    if metadata_field == "ccs":
        if groups["ccs"]:
            return any(v == requested or _numeric_ccd_equal(requested, v) for v in groups["ccs"])
        # Some archive products omit CCS in the extension header. Fall through
        # to explicitly typed CCD identifiers, but never IMAGEID or arbitrary
        # numeric header cards.
    for group in ("ccs", "explicit", "names"):
        values = groups[group]
        if not values:
            continue
        if any(v == requested for v in values):
            return True
        if group in ("ccs", "explicit") and any(_numeric_ccd_equal(requested, v) for v in values):
            return True
        if group == "names" and any(_numeric_ccd_equal(requested, v) for v in values):
            return True
        # An authoritative header group exists but does not match. Do not fall
        # through to a weaker group: that would reintroduce ambiguous matches.
        return False
    return False


def _get_frame_info(path, metadata_row):
    hdus = list_image_hdus(path)
    metadata_field, requested_ccd = _row_ccd(metadata_row)
    if not requested_ccd:
        keys = ", ".join(str(k) for k in (metadata_row or {}).keys())
        raise ValueError(
            f"metadata row has no CCD/CCS identifier for {path.name}; columns=[{keys}]"
        )
    matches = [h for h in hdus if _ccd_matches(requested_ccd, h, metadata_field)]
    if len(matches) != 1:
        available = []
        for h in hdus:
            values = _header_values_by_key(h)
            desc = "/".join(values["ccs"] + values["explicit"] + values["names"])
            available.append(f"HDU{h['index']}:{desc or h['extname']}")
        raise ValueError(
            f"metadata {metadata_field}={requested_ccd!r} maps to {len(matches)} HDUs "
            f"in {path.name}; available=" + "; ".join(available[:30])
        )
    h = matches[0]
    with _fits_open(path, mode="readonly", memmap=True, lazy_load_hdus=False) as hdul:
        header = hdul[h["index"]].header.copy()
    values = _header_values_by_key(h)
    identity = (values["ccs"] or values["explicit"] or values["names"] or [requested_ccd])[0]
    return {
        "path": path, "hdu_index": h["index"], "width": h["width"],
        "height": h["height"], "header": header, "ccd": requested_ccd,
        "ccd_identity": identity, "metadata_ccd_field": metadata_field,
        "extname": h["extname"]
    }

def _inject_psf(image, x, y, amplitude, sigma):
    """Add a subpixel Gaussian point source to a normalized image in place."""
    h, w = image.shape
    radius = max(3, int(math.ceil(4 * sigma)))
    x0, x1 = max(0, int(math.floor(x)) - radius), min(w, int(math.floor(x)) + radius + 1)
    y0, y1 = max(0, int(math.floor(y)) - radius), min(h, int(math.floor(y)) + radius + 1)
    if x0 >= x1 or y0 >= y1:
        return
    yy, xx = np.mgrid[y0:y1, x0:x1]
    image[y0:y1, x0:x1] += amplitude * np.exp(-((xx-x)**2 + (yy-y)**2) / (2*sigma*sigma)).astype(np.float32)


class SyntheticTrackletDataset(Dataset):
    """Real science-image backgrounds with fresh random synthetic moving sources.

    Every __getitem__ selects a new crop, velocity, start position, PSF, and flux.
    The network input is the robustly normalized raw FITS crop (per-frame median/MAD scaling).
    """
    def __init__(self, folders, cfg, training, items=None):
        self.cfg = dict(cfg)
        self.training = bool(training)
        self.size = int(cfg.get("crop_size", IMAGE_SIZE))
        self.min_frames = int(cfg.get("min_frames", 3))
        self.seed = int(cfg.get("seed", 42))
        self.synthetic_per_tracklet = int(cfg.get("synthetic_per_tracklet", 32 if training else 4))
        self.negative_per_tracklet = int(cfg.get("synthetic_negatives_per_tracklet", 32 if training else 4))
        self.items = list(items) if items is not None else []
        if items is None:
            for folder in folders:
                try:
                    files = raw_files(folder)
                    if len(files) < self.min_frames:
                        raise ValueError(f"only {len(files)} FITS files; need at least {self.min_frames}")
                    rows = read_metadata(folder)
                    if not rows:
                        raise ValueError("metadata.csv is missing or empty")
                    # Frame N maps to metadata row N-1. This remains correct if an
                    # individual download is missing, unlike blindly zipping lists.
                    valid_frames, valid_rows = [], []
                    for frame_path in files:
                        match = re.search(r"frame_(\d+)", frame_path.name, flags=re.IGNORECASE)
                        if not match:
                            continue
                        row_index = int(match.group(1)) - 1
                        if row_index < 0 or row_index >= len(rows):
                            print(f"[SKIP UNMAPPED FRAME] {folder.name}/{frame_path.name}: no matching metadata row")
                            continue
                        row = rows[row_index]
                        try:
                            product = primary_product_type(frame_path)
                            if product["prodtype"] != "image":
                                print(f"[SKIP NON-SCIENCE FRAME] {folder.name}/{frame_path.name}: PRODTYPE={product['prodtype'] or '<missing>'}; expected image")
                                continue
                            if product["obstype"] and product["obstype"] != "object":
                                print(f"[SKIP NON-OBJECT FRAME] {folder.name}/{frame_path.name}: OBSTYPE={product['obstype']}")
                                continue
                            valid_frames.append(frame_path)
                            valid_rows.append(row)
                        except Exception as exc:
                            print(f"[SKIP INVALID FITS PRODUCT] {folder.name}/{frame_path.name}: {exc}")
                    if len(valid_frames) < self.min_frames:
                        raise ValueError(f"only {len(valid_frames)} valid science-image frames remain; need at least {self.min_frames} (original files={len(files)})")
                    infos = [_get_frame_info(p, row) for p, row in zip(valid_frames, valid_rows)]
                    common_w = min(q["width"] for q in infos)
                    common_h = min(q["height"] for q in infos)
                    if common_w < self.size or common_h < self.size:
                        raise ValueError(f"common image bounds {common_w}x{common_h} smaller than {self.size}x{self.size} crop")
                    for info in infos:
                        info["wcs"] = _wcs_for_header(info["header"])
                    has_wcs = [q["wcs"] is not None for q in infos]
                    if any(has_wcs) and not all(has_wcs):
                        raise ValueError("only some valid frames have celestial WCS; cannot safely align sequence crops")
                    identities = {str(q["ccd_identity"]).strip().upper() for q in infos}
                    if len(identities) > 1 and not all(has_wcs):
                        raise ValueError("tracklet crosses CCD identities without complete WCS")
                    jdata = read_json(folder)
                    motion, motion_source = _extract_motion(folder, valid_rows, jdata, [q["header"] for q in infos], cfg)
                    self.items.append({
                        "folder": folder, "frames": infos, "rows": valid_rows,
                        "motion": motion, "motion_source": motion_source,
                        "has_wcs": all(has_wcs), "max_x0": common_w-self.size,
                        "max_y0": common_h-self.size,
                    })
                    print(f"[USABLE TRACKLET] {folder.name}: T={len(infos)} science frames; "
                          f"CCDs={','.join(str(q['ccd']) for q in infos)}; "
                          f"HDUs={','.join(str(q['hdu_index']) for q in infos)}")
                except Exception as exc:
                    print(f"[SKIP TRACKLET] {folder.name}: {exc}")
        if not self.items:
            raise RuntimeError(f"No usable tracklets with at least {self.min_frames} valid science frames under the supplied folders.")
        self.samples = []
        self._rebuild_samples()
        print(f"[ORIGINAL-ARCHITECTURE DATASET] usable tracklets={len(self.items)} samples={len(self.samples)}; "
              f"T >= {self.min_frames}; crop={self.size}x{self.size}; velocity range={cfg.get('synthetic_speed_px_frame_range',(4,18))} px/frame")

    def _rebuild_samples(self):
        self.samples = []
        for i in range(len(self.items)):
            self.samples.extend((i, True) for _ in range(self.synthetic_per_tracklet))
            self.samples.extend((i, False) for _ in range(self.negative_per_tracklet))

    def subset(self, item_indices, training):
        """Create a split from already validated items without rescanning FITS files."""
        clone = object.__new__(SyntheticTrackletDataset)
        clone.cfg = dict(self.cfg)
        clone.training = bool(training)
        clone.size = self.size
        clone.min_frames = self.min_frames
        clone.seed = self.seed + (0 if training else 1000003)
        clone.synthetic_per_tracklet = int(self.cfg.get("synthetic_per_tracklet", 32) if training else self.cfg.get("validation_per_tracklet", 4))
        clone.negative_per_tracklet = int(self.cfg.get("synthetic_negatives_per_tracklet", 32) if training else self.cfg.get("validation_negatives_per_tracklet", 4))
        clone.items = [self.items[i] for i in item_indices]
        clone._rebuild_samples()
        return clone

    def __len__(self):
        return len(self.samples)

    def _crop_origins(self, item, rng):
        """Choose a common sky footprint; per-frame pixel origins may differ.

        Actual pixel registration is performed in _read_registered_crops using
        the reference frame's WCS grid, not by assuming that equal crop sizes
        with different origins are aligned.
        """
        size = self.size
        ref = item["frames"][0]
        for _ in range(64):
            rx0 = rng.randint(0, item["max_x0"])
            ry0 = rng.randint(0, item["max_y0"])
            if not item["has_wcs"]:
                # Without WCS, a common pixel origin is only safe when the
                # selected detector/CCD identity is the same in every frame.
                identities = {str(f["ccd_identity"]).strip().upper() for f in item["frames"]}
                if len(identities) != 1:
                    raise RuntimeError("Cannot align frames without WCS across different CCD identities")
                return [(rx0, ry0)] * len(item["frames"])
            cx, cy = rx0 + (size-1)/2, ry0 + (size-1)/2
            try:
                ra0, dec0 = ref["wcs"].pixel_to_world_values(cx, cy)
                origins, valid = [], True
                for frame in item["frames"]:
                    px, py = frame["wcs"].world_to_pixel_values(ra0, dec0)
                    ox = int(round(float(px) - (size-1)/2))
                    oy = int(round(float(py) - (size-1)/2))
                    if ox < 0 or oy < 0 or ox+size > frame["width"] or oy+size > frame["height"]:
                        valid = False
                        break
                    origins.append((ox, oy))
                if valid:
                    return origins
            except Exception:
                continue
        raise RuntimeError(f"No overlapping {size}x{size} WCS crop for {item['folder'].name}")

    def _register_source_crops(self, item, origins, raw):
        """Register already-read source crops onto frame-0's reference grid."""
        frames = item["frames"]
        size = self.size
        if not item["has_wcs"]:
            stacked = np.stack(raw).astype(np.float32)
            if stacked.shape != (len(frames), size, size):
                raise RuntimeError(f"Unaligned crop shape for {item['folder'].name}: {stacked.shape}")
            return stacked, np.ones((size, size), dtype=bool)

        yy, xx = np.mgrid[0:size, 0:size].astype(np.float64)
        ref_x0, ref_y0 = origins[0]
        ref_x = xx + ref_x0
        ref_y = yy + ref_y0
        ra, dec = frames[0]["wcs"].pixel_to_world_values(ref_x, ref_y)
        aligned = [raw[0].astype(np.float32)]
        valid_masks = [np.ones((size, size), dtype=bool)]
        for index in range(1, len(frames)):
            fx, fy = frames[index]["wcs"].world_to_pixel_values(ra, dec)
            local_x = np.asarray(fx, dtype=np.float64) - origins[index][0]
            local_y = np.asarray(fy, dtype=np.float64) - origins[index][1]
            inside = ((local_x >= 0) & (local_x <= size-1) &
                      (local_y >= 0) & (local_y <= size-1) &
                      np.isfinite(local_x) & np.isfinite(local_y))
            if float(inside.mean()) < float(self.cfg.get("min_registered_fraction", 0.90)):
                raise RuntimeError(
                    f"WCS overlap too small for frame {index+1}: {inside.mean():.3%} "
                    f"(minimum={self.cfg.get('min_registered_fraction', 0.90):.1%})"
                )
            mapped = map_coordinates(
                raw[index], [np.nan_to_num(local_y, nan=-1), np.nan_to_num(local_x, nan=-1)],
                order=1, mode="constant", cval=0.0, prefilter=False
            ).astype(np.float32)
            mapped[~inside] = 0.0
            aligned.append(mapped)
            valid_masks.append(inside)
        stacked = np.stack(aligned).astype(np.float32)
        if stacked.shape != (len(frames), size, size):
            raise RuntimeError(f"WCS-registered crop shape mismatch for {item['folder'].name}: {stacked.shape}")
        return stacked, np.logical_and.reduce(valid_masks)

    def _read_registered_crops(self, item, origins, return_source=False):
        """Read source crops and register them on a shared frame-0 WCS grid.

        If return_source=True, also return the native-grid crop arrays so
        synthetic sources can be injected BEFORE the same reprojection applied
        to real stars/background structure.
        """
        raw_source = [read_crop(f["path"], ox, oy, self.size, hdu_index=f["hdu_index"], clip=False)
                      for f, (ox, oy) in zip(item["frames"], origins)]
        stacked, common_valid = self._register_source_crops(item, origins, raw_source)
        if return_source:
            return stacked, common_valid, raw_source
        return stacked, common_valid

    def _map_reference_position_to_native(self, item, origins, t, x, y):
        """Map a shared reference-grid position into frame t's native crop."""
        if not item.get("has_wcs", False) or t == 0:
            return float(x), float(y)
        ref_x = float(origins[0][0]) + float(x)
        ref_y = float(origins[0][1]) + float(y)
        ra, dec = item["frames"][0]["wcs"].pixel_to_world_values(ref_x, ref_y)
        full_x, full_y = item["frames"][t]["wcs"].world_to_pixel_values(ra, dec)
        return float(full_x) - float(origins[t][0]), float(full_y) - float(origins[t][1])

    def _inject_reference_position_into_native(self, item, origins, raw_source, t, x, y, amplitude, sigma):
        """Inject a validated reference-grid position into frame t's native pixels."""
        nx, ny = self._map_reference_position_to_native(item, origins, t, x, y)
        margin = max(5.0, 4.0 * float(sigma))
        if not (np.isfinite(nx) and np.isfinite(ny) and
                margin <= nx < self.size-margin and margin <= ny < self.size-margin):
            # This should be prevented by trajectory pre-validation in __getitem__.
            # Keep the guard to avoid corrupting data if the helper is called elsewhere.
            raise ValueError(f"Mapped injection falls outside native crop at frame {t}: ({nx:.2f},{ny:.2f})")
        _inject_psf(raw_source[t], nx, ny, amplitude, sigma)

    def _sample_velocity(self, item, T, rng):
        lo, hi = map(float, self.cfg.get("synthetic_speed_px_frame_range", (4.0, 18.0)))
        # Ensure a trajectory can fit inside the crop even for long sequences.
        hi = min(hi, (self.size - 40.0) / max(T-1, 1))
        lo = min(lo, hi)
        if hi <= 0:
            raise RuntimeError(f"Sequence length T={T} is too long for {self.size}px crop and configured velocity")
        base_vx, base_vy = item.get("motion", (0.0, 0.0))
        base_speed = math.hypot(float(base_vx), float(base_vy))
        if base_speed > 0 and rng.random() < 0.20:
            speed = min(hi, max(lo, base_speed * math.exp(rng.uniform(math.log(0.7), math.log(1.5)))))
            angle = math.atan2(float(base_vy), float(base_vx)) + rng.uniform(-math.radians(60), math.radians(60))
        else:
            # Match the original generator's faster movers, while sampling a new
            # speed and direction for every injected asteroid rather than fixing
            # one velocity per real tracklet.
            speed = rng.uniform(lo, hi)
            angle = rng.uniform(0.0, 2.0 * math.pi)
        return np.asarray([speed*math.cos(angle), speed*math.sin(angle)], dtype=np.float32)

    def __getitem__(self, index):
        item_idx, positive = self.samples[index]
        item = self.items[item_idx]
        rng = random.Random(self.seed + int(index)) if not self.training else random
        frames = item["frames"]
        T = len(frames)
        if T < self.min_frames:
            raise RuntimeError(f"{item['folder'].name}: expected >= {self.min_frames} frames, got {T}")
        origins = self._crop_origins(item, rng)
        raw, common_valid_mask, raw_source = self._read_registered_crops(item, origins, return_source=True)

        # Variable-cardinality labels: N is sampled from config for positive
        # examples and is zero for negatives. N is not a model output-slot limit.
        count_lo, count_hi = map(int, self.cfg.get("injection_count_range", (1, 4)))
        if count_lo < 1 or count_hi < count_lo:
            raise ValueError(f"Invalid injection_count_range={(count_lo, count_hi)}")
        requested_count = rng.randint(count_lo, count_hi) if positive else 0
        trajectories = []
        velocities_list = []
        amps = []
        sigmas = []
        for slot in range(requested_count):
            placed = False
            for _attempt in range(200):
                vx, vy = self._sample_velocity(item, T, rng)
                offsets_x = np.arange(T, dtype=np.float32) * vx
                offsets_y = np.arange(T, dtype=np.float32) * vy
                x_low, x_high = 16.0 - float(offsets_x.min()), self.size - 17.0 - float(offsets_x.max())
                y_low, y_high = 16.0 - float(offsets_y.min()), self.size - 17.0 - float(offsets_y.max())
                if x_low > x_high or y_low > y_high:
                    continue
                x0, y0 = rng.uniform(x_low, x_high), rng.uniform(y_low, y_high)
                trajectory = np.stack([x0 + offsets_x, y0 + offsets_y], axis=1).astype(np.float32)
                # Reject trajectories that touch WCS-filled borders; require a small
                # PSF margin around every injected centre in every frame.
                ix = np.rint(trajectory[:, 0]).astype(int)
                iy = np.rint(trajectory[:, 1]).astype(int)
                margin = 5
                if (np.any(ix < margin) or np.any(iy < margin) or
                    np.any(ix >= self.size-margin) or np.any(iy >= self.size-margin)):
                    continue
                if not all(common_valid_mask[max(0, yy-margin):min(self.size, yy+margin+1),
                                             max(0, xx-margin):min(self.size, xx+margin+1)].all()
                           for xx, yy in zip(ix, iy)):
                    continue
                if any(np.min(np.linalg.norm(trajectory - other, axis=1)) < 32.0 for other in trajectories):
                    continue
                amp = _sample_brightness(self.cfg, rng)
                sigma = rng.uniform(*self.cfg.get("injection_sigma_range", (1.2, 2.2)))
                # Validate the entire trajectory in every native crop BEFORE
                # modifying any frame. Previously, an out-of-crop mapping late
                # in the sequence could raise from a DataLoader worker after
                # earlier frames had already been injected, aborting the epoch
                # and contaminating retries with partial injections.
                jitter_std = float(self.cfg.get("motion_jitter_std", 0.15))
                jittered_positions = []
                native_positions = []
                margin = max(5.0, 4.0 * float(sigma))
                valid_native_trajectory = True
                for t, (x, y) in enumerate(trajectory):
                    jx = rng.gauss(0.0, jitter_std)
                    jy = rng.gauss(0.0, jitter_std)
                    jpos = (float(x+jx), float(y+jy))
                    nx, ny = self._map_reference_position_to_native(item, origins, t, *jpos)
                    if not (np.isfinite(nx) and np.isfinite(ny) and
                            margin <= nx < self.size-margin and margin <= ny < self.size-margin):
                        valid_native_trajectory = False
                        break
                    jittered_positions.append(jpos)
                    native_positions.append((nx, ny))
                if not valid_native_trajectory:
                    # Reject this candidate and sample another; do not raise and
                    # do not inject anything into raw_source for this attempt.
                    continue
                for t, ((x, y), (nx, ny)) in enumerate(zip(jittered_positions, native_positions)):
                    _inject_psf(raw_source[t], nx, ny,
                                amp*rng.uniform(0.92, 1.08), sigma)
                velocities_list.append((float(vx), float(vy)))
                trajectories.append(trajectory)
                amps.append(amp)
                sigmas.append(sigma)
                placed = True
                break
            if not placed:
                print(f"[WARN] Could not place target slot {slot+1} in {item['folder'].name}; slot marked absent")

        # Reproject AFTER native-grid injection. This ensures synthetic PSFs
        # undergo the same interpolation/resampling as real sources in frames
        # 1..T-1. Labels remain on the frame-0 reference pixel grid.
        if trajectories:
            raw, common_valid_mask = self._register_source_crops(item, origins, raw_source)
        # Clip only after synthetic sources have been added and all frames have
        # undergone the same WCS reprojection. This avoids injected pixels exceeding
        # a clipping ceiling that was already applied to the real background.
        raw = np.clip(raw, -8.0, 8.0).astype(np.float32)
        coords = np.stack(trajectories).astype(np.float32) if trajectories else np.empty((0, T, 2), dtype=np.float32)
        velocities = np.asarray(velocities_list, dtype=np.float32).reshape(-1, 2)
        target_mask = np.ones((len(trajectories),), dtype=np.float32)
        # Native crops are robust-normalized before reprojection; final clipping follows injection.
        return {
            "frames": torch.from_numpy(raw[:, None]).float(),
            "raw_frames": torch.from_numpy(raw).float(),
            "coords": torch.from_numpy(coords),
            "velocities": torch.from_numpy(velocities),
            "target_mask": torch.from_numpy(target_mask),
            "folder": str(item["folder"]),
            "positive_sample": bool(target_mask.sum() > 0),
            # Every returned tensor frame shares frame-0's reference pixel grid.
            # Keep the common crop origin distinct from source cutout origins,
            # which naturally differ between exposures before WCS reprojection.
            "crop_origins": torch.tensor([origins[0]] * len(origins), dtype=torch.float32),
            "source_crop_origins": torch.tensor(origins, dtype=torch.float32),
            "valid_mask_fraction": float(common_valid_mask.mean()),
            "wcs_registered": bool(item.get("has_wcs", False)),
            "injection_amplitudes": torch.tensor(amps, dtype=torch.float32),
            "injection_sigmas": torch.tensor(sigmas, dtype=torch.float32),
            "motion_source": item.get("motion_source", "synthetic_random"),
        }

def group_key(folder):
    """Leakage grouping without trusting the broken asteroid sky coordinates."""
    rows = read_metadata(folder)
    if not rows:
        return None
    row = {_norm_key(k): v for k, v in rows[0].items()}
    date = str(row.get("dateobs") or row.get("date_obs") or "").strip()
    if not date:
        raise RuntimeError(f"Missing DATE-OBS for leakage-safe split: {folder}")
    night = date[:10]
    # Prefer stable survey/field identifiers; never use asteroid RA/Dec here.
    field = ""
    for key in ("field", "fieldid", "field_id", "dataset", "dataset_id", "pointing_id"):
        value = str(row.get(key) or "").strip()
        if value:
            field = f"{key}={value}"
            break
    return f"{night}|{field}" if field else night


