"""Hybrid index-fingertip / palm-fallback movement analysis for a 57 x 57 cm board.

The index fingertip is the preferred measurement point when the index finger is
extended. When the child makes a fist or the index finger is not reliably extended,
the measured point falls back to the palm centre. Source changes are recorded and
kinematic calculations are split at those changes to avoid artificial jump artefacts.

The expected recording format is portrait 1080 x 1920 at 60 fps.
The activity zone is expanded by 50 image pixels beyond the marker-centre polygon.
The homography remains tied to the unexpanded physical-board calibration polygon.

Signal-quality design (v2)
--------------------------
* Positions are smoothed offline with a zero-lag local-polynomial (Savitzky-Golay type)
  smoother. The earlier causal One-Euro filter was removed: every restart of a tracking
  run (e.g. after a fingertip -> palm switch) produced a start-up transient that was
  read as a jerk spike.
* Jerk is estimated by a time-based local polynomial fit; the half-window at both ends of
  each continuous run is discarded (no one-sided, noise-dominated estimates).
* The fingertip/palm source is debounced (SOURCE_SWITCH_MIN_SECONDS) so that short
  landmark dropouts do not cut the trajectory into many short runs.
* Impossible position jumps are rejected twice: online (palm centre is checked even
  when the measured source changes) and offline (excursion removal over the whole
  segment, see _find_jump_outliers).
* Normalised jerk is evaluated per movement stroke, because the dimensionless jerk of
  a whole multi-stroke run grows with the number of strokes.
"""
from __future__ import annotations

import csv
import json
import os
import traceback
from datetime import datetime
from math import gcd
from typing import Callable, Optional

import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
from shapely.geometry import Point, Polygon


VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".wmv", ".webm", ".m4v"}
MARKER_IDS = {0, 1, 2, 3}
BOARD_SIDE_CM = 57.0
EXPECTED_VIDEO_WIDTH = 1080
EXPECTED_VIDEO_HEIGHT = 1920
ALLOWED_VIDEO_SIZES = {(1080, 1920)}  # fixed portrait input (9:16)
EXPECTED_VIDEO_FPS = 60.0
VIDEO_FPS_TOLERANCE = 0.5
POSE_VISIBILITY_THRESHOLD = 0.45

# Set this to the dictionary used to print the physical markers. Examples include
# "DICT_4X4_50" or "DICT_APRILTAG_36h11" (if available in your OpenCV build).
MARKER_DICTIONARY_NAME = "DICT_4X4_50"

# Physical distance from each board corner to its corresponding marker centre.
# Leave at 0.0 only when marker centres represent the 57 x 57 cm board corners.
# Example: if marker centres are 2 cm inward from each edge, set this to 2.0.
MARKER_CENTER_INSET_CM = 0.0

# The activity-detection polygon is buffered outward from the marker-centre
# quadrilateral by this many ORIGINAL-IMAGE pixels. This is deliberately separate
# from the 57 cm homography, which must not be expanded.
ACTIVITY_ZONE_MARGIN_PX = 50.0
EXIT_CONFIRM_SECONDS = 0.25
MARKER_CALIBRATION_SAMPLES = 3
MARKER_SAMPLE_WINDOW = 15

# The old MediaPipe Hands API assumes mirrored input for handedness labels.
# These raw recordings are assumed to be unmirrored. Set True only if videos are
# actually horizontally mirrored before they are passed to this pipeline.
INPUT_VIDEO_IS_MIRRORED = False

# Palm-centre estimate: mean of wrist and four MCP landmarks (0, 5, 9, 13, 17).
PALM_LANDMARK_INDICES = (0, 5, 9, 13, 17)
# Index finger is selected when both PIP and DIP joints are sufficiently extended.
# Hysteresis avoids frame-to-frame switching when the angles hover near a threshold.
INDEX_EXTENSION_ENTER_PIP_DEG = 145.0
INDEX_EXTENSION_ENTER_DIP_DEG = 140.0
INDEX_EXTENSION_EXIT_PIP_DEG = 130.0
INDEX_EXTENSION_EXIT_DIP_DEG = 125.0
INTERPOLATE_GAP_LIMIT_SECONDS = 0.20

# Tracking safeguards. The identity matcher uses the palm centre (a stable point
# shared by both the fingertip and fist modes) rather than the selected fingertip.
# A deliberately generous speed limit is used only to reject gross jumps, not to
# describe normal movement. Validate it against representative recordings.
MAX_IDENTITY_SPEED_CM_S = 350.0
# Physical speed limit of the measured point between two accepted samples. It is used
# online (live rejection) and offline (_find_jump_outliers). At 60 fps 350 cm/s is
# 5.8 cm per frame; a real child's hand practically never moves that fast, while a
# false detection on the other side of the 57 cm board is 20-60 cm per frame.
MAX_MEASURED_POINT_SPEED_CM_S = 350.0
# A run of samples separated from both neighbours by impossible jumps is treated as a
# false-detection excursion when it is shorter than this and the neighbours join up.
JUMP_MAX_EXCURSION_SECONDS = 0.50

# Fingertip <-> palm source debounce: the raw decision must disagree with the active
# source continuously for this long before the measured point is switched.
SOURCE_SWITCH_MIN_SECONDS = 0.10

# Offline smoothing / differentiation (all time based, therefore fps independent).
SMOOTHING_WINDOW_SECONDS = 0.30   # local polynomial smoother, ~4-5 Hz cut-off
SMOOTHING_POLY_DEGREE = 3
JERK_WINDOW_SECONDS = 0.30        # local polynomial window for the 3rd derivative
JERK_POLY_DEGREE = 4

# Normalised (dimensionless) jerk: "per_stroke" (recommended, comparable between
# recordings of different length) or "per_run" (legacy whole-run value).
NORM_JERK_MODE = "per_stroke"
STROKE_REST_FRACTION = 0.05       # speed below 5 % of the run's p95 speed = rest
STROKE_MIN_REST_SPEED_CM_S = 2.0
STROKE_MIN_SECONDS = 0.25
STROKE_MAX_SECONDS = 2.0          # longer continuous motion is cut into equal parts
STROKE_MIN_PATH_CM = 2.0

# Elbow angle: "world_3d" uses MediaPipe pose_world_landmarks (metric 3-D, removes the
# foreshortening of the oblique camera); "image_2d" is the legacy image-plane angle.
ELBOW_ANGLE_MODE = "world_3d"
ELBOW_MEDIAN_WINDOW_SECONDS = 0.12

# A hand must be at least this far (cm) on both sides of the board centre line to count
# as "crossed"; avoids reporting landmark jitter on the line as a crossing.
CENTER_LINE_MARGIN_CM = 1.0

PIPELINE_VERSION = "2.1.0"

# Decimal separator of the CSV. "." is unambiguous and machine readable (pandas/R).
# Do NOT rely on "," : a spreadsheet with a different locale reads "10,217" as the
# thousands-grouped number 10217 and leaves most other values as text. For people,
# use the .xlsx report written next to the CSV: its cells are real numbers and
# therefore independent of the locale of the computer that opens it.
CSV_DECIMAL_SEPARATOR = "."

# --- data-quality rules used for the "kvalita dat" report section -----------------
QA_MIN_VALID_FRAME_PERCENT = 70.0      # hand point valid in less of the activity -> review
QA_MAX_REJECTED_FRAME_PERCENT = 3.0    # more frames rejected as false jumps -> review
QA_SPEED_NEAR_LIMIT_FRACTION = 0.90    # max speed this close to the jump limit -> review
QA_MIN_INTERVALS_FOR_RELIABLE_CV = 4   # tempo CV from fewer intervals is only indicative
QA_MAX_SOURCE_SWITCHES_PER_SECOND = 1.0
# A stroke needs this share of valid jerk samples (the half-window at run ends is lost)
# to be evaluated; shorter coverage would truncate T and bias the dimensionless jerk.
STROKE_MIN_JERK_COVERAGE = 0.95

REPORT_CATEGORIES = [
    "R ruka",
    "L ruka",
    "společné metriky",
    "kvalita dat",
    "obecné parametry videa",
]

REPORT_MAX_SEGMENTS: Optional[int] = None

SEGMENT_REPORT_SCHEMA = [
    # Right hand: 12 rows
    ("R ruka", "Trajektorie", "Délka trajektorie", "coord", "kinematika.delka_drahy_{coord_unit}.R"),
    ("R ruka", "Rychlost", "Maximální rychlost", "speed", "kinematika.rychlost.max_R"),
    ("R ruka", "Rychlost", "Průměrná rychlost", "speed", "kinematika.rychlost.mean_R"),
    ("R ruka", "Rychlost", "95. percentil rychlosti", "speed", "kinematika.rychlost.p95_R"),
    ("R ruka", "Loket", "Maximální extenze", "degree", "uhly_a_rozsah.max_extenze_deg.R"),
    ("R ruka", "Loket", "Rozsah pohybu (ROM)", "degree", "uhly_a_rozsah.rom_deg.R"),
    ("R ruka", "Plynulost", "Normovaný jerk", "none", "kinematika.plynulost_norm_jerk.R"),
    ("R ruka", "Plynulost", "Maximální jerk", "jerk", "kinematika.max_jerk_{jerk_unit}.R"),
    ("R ruka", "Plynulost", "95. percentil jerku", "jerk", "kinematika.p95_jerk_{jerk_unit}.R"),
    ("R ruka", "Rytmus", "Variabilita tempa (CV)", "percent", "rytmicita_a_cykly.stabila_tempa_cv_percent_R"),
    ("R ruka", "Rytmus", "Počet pohybových period", "count", "rytmicita_a_cykly.pocet_detectovanych_period_R"),
    ("R ruka", "Prostorová přesnost", "Překročení středové linie", "bool", "prostorova_presnost.prekroceni_stredove_linie.R"),

    # Left hand: 12 rows
    ("L ruka", "Trajektorie", "Délka trajektorie", "coord", "kinematika.delka_drahy_{coord_unit}.L"),
    ("L ruka", "Rychlost", "Maximální rychlost", "speed", "kinematika.rychlost.max_L"),
    ("L ruka", "Rychlost", "Průměrná rychlost", "speed", "kinematika.rychlost.mean_L"),
    ("L ruka", "Rychlost", "95. percentil rychlosti", "speed", "kinematika.rychlost.p95_L"),
    ("L ruka", "Loket", "Maximální extenze", "degree", "uhly_a_rozsah.max_extenze_deg.L"),
    ("L ruka", "Loket", "Rozsah pohybu (ROM)", "degree", "uhly_a_rozsah.rom_deg.L"),
    ("L ruka", "Plynulost", "Normovaný jerk", "none", "kinematika.plynulost_norm_jerk.L"),
    ("L ruka", "Plynulost", "Maximální jerk", "jerk", "kinematika.max_jerk_{jerk_unit}.L"),
    ("L ruka", "Plynulost", "95. percentil jerku", "jerk", "kinematika.p95_jerk_{jerk_unit}.L"),
    ("L ruka", "Rytmus", "Variabilita tempa (CV)", "percent", "rytmicita_a_cykly.stabila_tempa_cv_percent_L"),
    ("L ruka", "Rytmus", "Počet pohybových period", "count", "rytmicita_a_cykly.pocet_detectovanych_period_L"),
    ("L ruka", "Prostorová přesnost", "Překročení středové linie", "bool", "prostorova_presnost.prekroceni_stredove_linie.L"),

    # Bilateral metrics: 6 rows
    ("společné metriky", "Symetrie", "Rozdíl délek trajektorií", "coord", "kinematika.delka_drahy_{coord_unit}.rozdil"),
    ("společné metriky", "Symetrie", "Asymetrie délek trajektorií", "percent", "kinematika.delka_drahy_{coord_unit}.asymetrie_percent"),
    ("společné metriky", "Synchronizace", "Korelace rychlostí", "correlation", "synchronizace_a_symetrie.korelace_rychlosti_r_LR"),
    ("společné metriky", "Synchronizace", "Korelace pohybu v ose X", "correlation", "synchronizace_a_symetrie.korelace_ve_smeru_x_r_vx"),
    ("společné metriky", "Synchronizace", "Korelace pohybu v ose Y", "correlation", "synchronizace_a_symetrie.korelace_ve_smeru_y_r_vy"),
    ("společné metriky", "Rytmus", "Variabilita tempa obou rukou (CV)", "percent", "rytmicita_a_cykly.stabila_tempa_cv_percent"),


    # Data-quality rows: tell the reader which results can be trusted.
    ("kvalita dat", "R ruka – sledování", "Podíl snímků s platným bodem", "percent", "metadata.kvalita_trackingu.R.podil_platnych_snimku_percent"),
    ("kvalita dat", "R ruka – sledování", "Zamítnuté snímky (falešné skoky)", "percent", "metadata.kvalita_trackingu.R.zamitnuto_celkem_percent"),
    ("kvalita dat", "R ruka – sledování", "Přepnutí prst/dlaň", "count", "metadata.kvalita_trackingu.R.prepnuti_zdroje_prst_dlan"),
    ("kvalita dat", "R ruka – sledování", "Zdroj normovaného jerku", "text", "kinematika.plynulost_norm_jerk_zdroj.R"),
    ("kvalita dat", "L ruka – sledování", "Podíl snímků s platným bodem", "percent", "metadata.kvalita_trackingu.L.podil_platnych_snimku_percent"),
    ("kvalita dat", "L ruka – sledování", "Zamítnuté snímky (falešné skoky)", "percent", "metadata.kvalita_trackingu.L.zamitnuto_celkem_percent"),
    ("kvalita dat", "L ruka – sledování", "Přepnutí prst/dlaň", "count", "metadata.kvalita_trackingu.L.prepnuti_zdroje_prst_dlan"),
    ("kvalita dat", "L ruka – sledování", "Zdroj normovaného jerku", "text", "kinematika.plynulost_norm_jerk_zdroj.L"),
    ("kvalita dat", "Celkově", "Spolehlivost výsledků", "text", "metadata.spolehlivost"),
    ("kvalita dat", "Celkově", "Upozornění ke kvalitě dat", "text", "metadata.upozorneni_kvality"),

    # Segment metadata: 2 rows
    ("obecné parametry videa", "Aktivita", "Typ aktivity", "text", "metadata.typ_aktivity"),
    ("obecné parametry videa", "Aktivita", "Doba aktivity", "seconds", "metadata.doba_pohybu_s"),
]

VIDEO_REPORT_SCHEMA = [
    ("Formát videa", "Rozlišení", "resolution", "video.resolution"),
    ("Formát videa", "Orientace", "text", "video.video_orientation"),
    ("Formát videa", "Poměr stran", "ratio", "video.video_display_aspect_ratio"),
    ("Formát videa", "Snímková frekvence", "fps", "video.video_fps"),
    ("Kalibrace", "Jednotky souřadnic", "text", "video.jednotky_souradnic"),
    ("Kalibrace", "Jednotky rychlosti", "text", "video.jednotky_rychlosti"),
    ("Kalibrace", "Jednotky jerku", "text", "video.jednotky_jerku"),
    ("Kalibrace", "Metoda kalibrace", "text", "video.kalibrace_desky"),
    ("Formát videa", "Upozornění formátu", "text", "video.warnings"),
    ("Zpracování", "Stav zpracování", "text", "video.status"),
    ("Zpracování", "Verze pipeline", "text", "video.pipeline_version"),
]
IDENTITY_LABEL_PENALTY_BOARD_DIAGONALS = 0.025
IDENTITY_STALE_AFTER_SECONDS = 0.60


# =========================================================================
# 1. FILTERING & UTILITIES
# =========================================================================
def get_video_format(width: int, height: int) -> dict:
    """Return the true orientation and the reduced aspect ratio of a video."""
    width, height = int(width), int(height)
    if width <= 0 or height <= 0:
        return {"video_orientation": "neznámá", "video_display_aspect_ratio": "neznámý"}
    divisor = gcd(width, height)
    if width == height:
        orientation = "square"
    elif height > width:
        orientation = "portrait"
    else:
        orientation = "landscape"
    return {
        "video_orientation": orientation,
        "video_display_aspect_ratio": f"{width // divisor}:{height // divisor}",
    }


def _build_video_metadata(
    width: int,
    height: int,
    fps: float,
    homography: Optional[np.ndarray],
    warnings=None,
) -> dict:
    """Video format, calibration state and units shared by the JSON and CSV reports."""
    video_format = get_video_format(width, height)
    calibrated = homography is not None
    return {
        "video_width_px": int(width),
        "video_height_px": int(height),
        "video_fps": round(float(fps), 3),
        **video_format,
        "jednotky_souradnic": "cm" if calibrated else "px",
        "jednotky_rychlosti": "cm/s" if calibrated else "px/s",
        "jednotky_jerku": "cm/s³" if calibrated else "px/s³",
        "kalibrace_desky": "homografie_57cm" if calibrated else "nekalibrovano_px",
        "upozorneni_formatu_videa": list(warnings or []),
        "pipeline_version": PIPELINE_VERSION,
    }


def _local_poly_fit(
    timestamps: np.ndarray,
    values: np.ndarray,
    derivative_order: int,
    degree: int,
    window_seconds: float,
    allow_one_sided: bool,
) -> np.ndarray:
    """Local polynomial regression on a time base (Savitzky-Golay for irregular sampling).

    A polynomial of ``degree`` is fitted to all samples within +-window/2 of each
    sample and its ``derivative_order``-th derivative is evaluated at the sample time.
    The estimate is zero-lag (symmetric window). Where the full symmetric window does
    not fit (the two ends of a run) the result is NaN unless ``allow_one_sided`` is
    True: then the window keeps its full length but is shifted inwards and the fit is
    evaluated at the edge sample (the "interp" edge mode of Savitzky-Golay filters),
    which is much less noisy than a truncated window.
    """
    t = np.asarray(timestamps, dtype=float)
    v = np.asarray(values, dtype=float)
    n = len(v)
    result = np.full(n, np.nan, dtype=float)
    min_points = degree + 2
    if n < min_points or window_seconds <= 0 or derivative_order > degree:
        return result
    half = float(window_seconds) / 2.0
    eps = 1e-9
    span = float(t[-1] - t[0])
    if span < 2.0 * half:
        # Run shorter than one window: only the all-samples fit is available.
        if not allow_one_sided:
            return result
        half = max(span / 2.0, eps)
    for i in range(n):
        centre = t[i]
        if centre - half < t[0] - eps:
            if not allow_one_sided:
                continue
            centre = t[0] + half
        elif centre + half > t[-1] + eps:
            if not allow_one_sided:
                continue
            centre = t[-1] - half
        lo = int(np.searchsorted(t, centre - half - eps, side="left"))
        hi = int(np.searchsorted(t, centre + half + eps, side="right"))
        if hi - lo < min_points:
            continue
        u = (t[lo:hi] - centre) / half
        design = np.vander(u, degree + 1, increasing=True)
        try:
            coefficients = np.linalg.lstsq(design, v[lo:hi], rcond=None)[0]
        except np.linalg.LinAlgError:
            continue
        poly = np.polynomial.polynomial.polyder(coefficients, derivative_order) if derivative_order else coefficients
        estimate = float(np.polynomial.polynomial.polyval((t[i] - centre) / half, poly)) / (half ** derivative_order)
        if np.isfinite(estimate):
            result[i] = estimate
    return result


def _local_poly_smooth(
    timestamps: np.ndarray,
    values: np.ndarray,
    window_seconds: float = SMOOTHING_WINDOW_SECONDS,
    degree: int = SMOOTHING_POLY_DEGREE,
) -> np.ndarray:
    """Zero-lag smoothing of one continuous run (falls back to raw values if too short)."""
    values = np.asarray(values, dtype=float)
    smoothed = _local_poly_fit(
        timestamps, values, 0, degree, window_seconds, allow_one_sided=True
    )
    return np.where(np.isfinite(smoothed), smoothed, values)


def _median_filter_series(timestamps: np.ndarray, values: np.ndarray, window_seconds: float) -> np.ndarray:
    """Time-based running median that ignores NaN and keeps NaN gaps as gaps."""
    t = np.asarray(timestamps, dtype=float)
    v = np.asarray(values, dtype=float)
    result = np.full(v.shape, np.nan, dtype=float)
    half = float(window_seconds) / 2.0
    for i in range(len(v)):
        if not np.isfinite(v[i]) or not np.isfinite(t[i]):
            continue
        lo = int(np.searchsorted(t, t[i] - half, side="left"))
        hi = int(np.searchsorted(t, t[i] + half, side="right"))
        window = v[lo:hi]
        window = window[np.isfinite(window)]
        result[i] = float(np.median(window)) if window.size else v[i]
    return result


def calculate_angle(a, b, c):
    a, b, c = np.asarray(a, dtype=float), np.asarray(b, dtype=float), np.asarray(c, dtype=float)
    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = abs(radians * 180.0 / np.pi)
    if angle > 180.0:
        angle = 360.0 - angle
    return float(angle)


def get_activity_config(activity_id: int):
    config = {"track_left_hand": True, "track_right_hand": True, "track_elbows": True}
    if activity_id == 8:
        config["track_left_hand"] = False
        config["track_right_hand"] = True
    elif activity_id == 9:
        config["track_left_hand"] = True
        config["track_right_hand"] = False
    return config


def _contiguous_true_runs(mask: np.ndarray):
    """Yield slices for contiguous True runs in a boolean array."""
    mask = np.asarray(mask, dtype=bool)
    if mask.size == 0:
        return
    edges = np.diff(np.r_[False, mask, False].astype(np.int8))
    starts = np.flatnonzero(edges == 1)
    ends = np.flatnonzero(edges == -1)
    for start, end in zip(starts, ends):
        yield slice(int(start), int(end))


def _contiguous_tracking_runs(valid: np.ndarray, sources: np.ndarray):
    """Yield finite runs, additionally splitting whenever reference point changes.

    A switch from the index fingertip to the palm centre changes the anatomical
    reference location. Never differentiate or sum a path across that switch.
    """
    valid = np.asarray(valid, dtype=bool)
    sources = np.asarray(sources, dtype=object)
    for base_run in _contiguous_true_runs(valid):
        start = base_run.start
        for i in range(start + 1, base_run.stop):
            if sources[i] != sources[i - 1]:
                yield slice(start, i)
                start = i
        if start < base_run.stop:
            yield slice(start, base_run.stop)


def _filter_missing_series(
    timestamps: np.ndarray,
    values: np.ndarray,
    tracking_sources: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Smooth finite runs separately (zero-lag); never smooth across reference-point switches."""
    timestamps = np.asarray(timestamps, dtype=float)
    values = np.asarray(values, dtype=float)
    result = np.full(values.shape, np.nan, dtype=float)
    valid = np.isfinite(values) & np.isfinite(timestamps)
    if tracking_sources is None:
        runs = _contiguous_true_runs(valid)
    else:
        runs = _contiguous_tracking_runs(valid, np.asarray(tracking_sources, dtype=object))
    for run in runs:
        start, end = run.start, run.stop
        if start >= end:
            continue
        result[start:end] = _local_poly_smooth(timestamps[start:end], values[start:end])
    return result


def _safe_corr(a: np.ndarray, b: np.ndarray) -> Optional[float]:
    mask = np.isfinite(a) & np.isfinite(b)
    if int(mask.sum()) < 3:
        return None
    a_valid, b_valid = a[mask], b[mask]
    if np.std(a_valid) <= 1e-12 or np.std(b_valid) <= 1e-12:
        return None
    value = float(np.corrcoef(a_valid, b_valid)[0, 1])
    return value if np.isfinite(value) else None


def _local_polynomial_derivative(
    values: np.ndarray,
    timestamps: np.ndarray,
    derivative_order: int = 3,
    window_seconds: float = JERK_WINDOW_SECONDS,
    degree: int = JERK_POLY_DEGREE,
) -> np.ndarray:
    """Estimate a derivative of a continuous run by local polynomial regression.

    Repeated finite differences amplify landmark noise dramatically at the third
    derivative. A centred, time-based local fit is far more robust and independent of
    the frame rate. The estimate is NaN within window/2 of both run ends, where only a
    one-sided (and therefore noise- and transient-dominated) fit would be possible.
    """
    return _local_poly_fit(
        timestamps, values, derivative_order, degree, window_seconds, allow_one_sided=False
    )


def _segment_strokes(timestamps: np.ndarray, speed: np.ndarray) -> list:
    """Split one continuous run into movement strokes; returns [(start, stop), ...].

    A stroke is a contiguous stretch with speed above a rest threshold (5 % of the run's
    95th speed percentile, at least 2 cm/s). Strokes shorter than STROKE_MIN_SECONDS are
    ignored and continuous motion longer than STROKE_MAX_SECONDS (for example circling
    without a pause) is cut into equal parts, so the result never depends on how many
    repetitions a recording contains.
    """
    t = np.asarray(timestamps, dtype=float)
    v = np.asarray(speed, dtype=float)
    finite = np.isfinite(v)
    if t.size < 5 or not finite.any():
        return []
    threshold = max(
        STROKE_MIN_REST_SPEED_CM_S, STROKE_REST_FRACTION * float(np.percentile(v[finite], 95))
    )
    moving = finite & (v >= threshold)
    strokes = []
    for run in _contiguous_true_runs(moving):
        start, stop = run.start, run.stop
        if stop - start < 3:
            continue
        duration = float(t[stop - 1] - t[start])
        if duration < STROKE_MIN_SECONDS:
            continue
        parts = max(1, int(np.ceil(duration / STROKE_MAX_SECONDS)))
        edges = np.rint(np.linspace(start, stop, parts + 1)).astype(int)
        for a, b in zip(edges[:-1], edges[1:]):
            if b - a >= 3:
                strokes.append((int(a), int(b)))
    return strokes


def _nested_value(data: dict, dotted_path: str):
    value = data
    for part in dotted_path.split("."):
        if not isinstance(value, dict) or part not in value:
            return None
        value = value[part]
    return value


def _estimate_periodic_tempo_cv(
    timestamps: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    tracking_sources: Optional[np.ndarray] = None,
) -> tuple[Optional[float], int]:
    """Estimate variability of repeated motion periods within one activity segment.

    The 2-D trajectory is projected onto its dominant principal direction. The
    detrended, lightly smoothed projection is used to find same-direction peaks;
    the time gaps between peaks approximate repeated movement periods. A CV is
    returned only when at least two full period intervals can be measured.
    This is a geometry-agnostic heuristic, not a task-specific clinical measure.
    """
    timestamps = np.asarray(timestamps, dtype=float)
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    valid = np.isfinite(timestamps) & np.isfinite(x) & np.isfinite(y)
    if tracking_sources is None:
        runs = list(_contiguous_true_runs(valid))
    else:
        runs = list(_contiguous_tracking_runs(valid, np.asarray(tracking_sources, dtype=object)))
    if not runs:
        return None, 0
    run = max(runs, key=lambda part: part.stop - part.start)
    t = timestamps[run]
    points = np.column_stack((x[run], y[run]))
    if len(t) < 12 or float(t[-1] - t[0]) < 0.8:
        return None, 0
    positive_dt = np.diff(t)
    positive_dt = positive_dt[positive_dt > 0]
    if not positive_dt.size:
        return None, 0
    dt = float(np.median(positive_dt))
    if not np.isfinite(dt) or dt <= 0:
        return None, 0

    # Project to the direction with greatest positional variation.
    centered = points - np.mean(points, axis=0)
    try:
        eigenvalues, eigenvectors = np.linalg.eigh(np.cov(centered.T))
        direction = eigenvectors[:, int(np.argmax(eigenvalues))]
        signal = centered @ direction
    except (np.linalg.LinAlgError, ValueError):
        return None, 0
    if not np.all(np.isfinite(signal)) or float(np.ptp(signal)) <= 1e-8:
        return None, 0

    # Remove slow drift (for example, a gradual shift across the board).
    time_from_start = t - t[0]
    try:
        trend = np.polyval(np.polyfit(time_from_start, signal, 1), time_from_start)
        signal = signal - trend
    except (np.linalg.LinAlgError, ValueError):
        pass

    # Light moving-average smoothing to reduce frame-to-frame landmark jitter.
    smooth_window = max(3, int(round(0.08 / dt)))
    if smooth_window % 2 == 0:
        smooth_window += 1
    smooth_window = min(smooth_window, max(3, (len(signal) // 2) * 2 - 1))
    if smooth_window >= 3 and len(signal) >= smooth_window:
        kernel = np.ones(smooth_window, dtype=float) / smooth_window
        signal = np.convolve(signal, kernel, mode="same")

    if float(np.ptp(signal)) <= 1e-8:
        return None, 0

    # Find local maxima. Same-polarity maxima are one approximate period apart.
    candidates = [
        i for i in range(1, len(signal) - 1)
        if signal[i] >= signal[i - 1] and signal[i] > signal[i + 1]
    ]
    min_distance = max(2, int(round(0.25 / dt)))
    # Non-maximum suppression prevents closely spaced jitter peaks from counting
    # as separate movement cycles; keep the higher peak in each short interval.
    selected = []
    for idx in sorted(candidates, key=lambda i: signal[i], reverse=True):
        if all(abs(idx - existing) >= min_distance for existing in selected):
            selected.append(idx)
    peaks = sorted(selected)
    if len(peaks) < 3:
        return None, max(0, len(peaks) - 1)

    amplitude = float(np.ptp(signal))
    prominence_threshold = max(amplitude * 0.07, float(np.std(signal)) * 0.12)
    prominence_window = max(min_distance, int(round(1.5 / dt)))
    prominent = []
    for i, peak in enumerate(peaks):
        left_bound = max(0, peak - prominence_window)
        right_bound = min(len(signal), peak + prominence_window + 1)
        left_min = float(np.min(signal[left_bound:peak + 1]))
        right_min = float(np.min(signal[peak:right_bound]))
        prominence = min(float(signal[peak]) - left_min, float(signal[peak]) - right_min)
        # At a segment edge, use the available side instead of rejecting a peak
        # only because the full prominence window is not present.
        if peak - left_bound < min_distance:
            prominence = float(signal[peak]) - right_min
        if right_bound - peak - 1 < min_distance:
            prominence = float(signal[peak]) - left_min
        if prominence >= prominence_threshold:
            prominent.append(peak)

    if len(prominent) < 3:
        return None, max(0, len(prominent) - 1)
    intervals = np.diff(t[np.asarray(prominent, dtype=int)])
    intervals = intervals[np.isfinite(intervals) & (intervals > 0)]
    detected_period_count = int(len(intervals))
    if len(intervals) < 2:
        return None, detected_period_count
    # The zero-lag smoother has no start-up transient, so every interval is used.
    cv_intervals = intervals
    mean_interval = float(np.mean(cv_intervals))
    if mean_interval <= 0:
        return None, detected_period_count
    cv_percent = float(np.std(cv_intervals) / mean_interval * 100.0)
    return (cv_percent if np.isfinite(cv_percent) else None), detected_period_count


def _assess_data_quality(metrics: dict) -> list:
    """Return human-readable warnings for results that should be reviewed before use."""
    notes = []
    quality = _nested_value(metrics, "metadata.kvalita_trackingu") or {}
    calibrated = str(_nested_value(metrics, "metadata.jednotky_souradnic") or "cm") == "cm"
    duration = _nested_value(metrics, "metadata.doba_pohybu_s")
    for side, label in (("L", "levá ruka"), ("R", "pravá ruka")):
        side_quality = quality.get(side)
        if not side_quality:
            continue  # hand not part of this activity
        valid_pct = side_quality.get("podil_platnych_snimku_percent")
        if valid_pct is not None and valid_pct < QA_MIN_VALID_FRAME_PERCENT:
            notes.append(f"{label}: platný bod jen v {valid_pct:.0f} % snímků")
        rejected_pct = side_quality.get("zamitnuto_celkem_percent")
        if rejected_pct is not None and rejected_pct > QA_MAX_REJECTED_FRAME_PERCENT:
            notes.append(f"{label}: {rejected_pct:.1f} % snímků zamítnuto jako falešné skoky")
        switches = side_quality.get("prepnuti_zdroje_prst_dlan") or 0
        if duration and duration > 0 and switches / duration > QA_MAX_SOURCE_SWITCHES_PER_SECOND:
            notes.append(f"{label}: časté přepínání prst/dlaň ({switches}×)")
        max_speed = _nested_value(metrics, f"kinematika.rychlost.max_{side}")
        if calibrated and max_speed is not None and max_speed > QA_SPEED_NEAR_LIMIT_FRACTION * MAX_MEASURED_POINT_SPEED_CM_S:
            notes.append(f"{label}: maximální rychlost {max_speed:.0f} cm/s je u fyzikálního limitu – zkontrolovat sledování")
        intervals = _nested_value(metrics, f"rytmicita_a_cykly.pocet_detectovanych_period_{side}")
        cv_value = _nested_value(metrics, f"rytmicita_a_cykly.stabila_tempa_cv_percent_{side}")
        if cv_value is not None and intervals is not None and intervals < QA_MIN_INTERVALS_FOR_RELIABLE_CV:
            notes.append(f"{label}: CV tempa z {intervals} intervalů je jen orientační")
        if _nested_value(metrics, f"kinematika.plynulost_norm_jerk_zdroj.{side}") == "cely_usek":
            notes.append(f"{label}: normovaný jerk z celého úseku (žádný úplný tah) – nelze srovnávat")
    return notes


def calculate_all_metrics(
    df: pd.DataFrame,
    activity_id: Optional[int],
    center_x: float = None,
    num_cycles: int = 3,
    compare_hands: bool = True,
) -> dict:
    """Calculate all kinematic metrics and attach a data-quality assessment.

    "spolehlivost" is "Ke kontrole" when tracking coverage, rejected false jumps, source
    flicker or a speed at the physical limit make the numbers doubtful; purely
    informational remarks (few tempo intervals, whole-run jerk) are listed in
    "upozorneni_kvality" without changing that verdict.
    """
    metrics = _calculate_all_metrics_core(
        df, activity_id, center_x=center_x, num_cycles=num_cycles, compare_hands=compare_hands
    )
    notes = _assess_data_quality(metrics)
    informational = ("jen orientační", "nelze srovnávat")
    serious = [note for note in notes if not any(token in note for token in informational)]
    metrics["metadata"]["upozorneni_kvality"] = notes
    metrics["metadata"]["spolehlivost"] = "Ke kontrole" if serious else "Dobrá"
    return metrics


def _order_quad_points(points: np.ndarray) -> np.ndarray:
    """Return four image points in a consistent clockwise order, starting near top-left."""
    pts = np.asarray(points, dtype=np.float32).reshape(4, 2)
    center = np.mean(pts, axis=0)
    angles = np.arctan2(pts[:, 1] - center[1], pts[:, 0] - center[0])
    ordered = pts[np.argsort(angles)]  # clockwise in image coordinates (y increases downwards)
    start = int(np.argmin(ordered[:, 0] + ordered[:, 1]))
    ordered = np.roll(ordered, -start, axis=0)
    return ordered.astype(np.float32)



# =========================================================================
# 2. METRICS CALCULATOR
# =========================================================================
def _calculate_all_metrics_core(
    df: pd.DataFrame,
    activity_id: Optional[int],
    center_x: float = None,
    num_cycles: int = 3,
    compare_hands: bool = True,
) -> dict:
    """Calculate metrics in calibrated board coordinates when homography is available."""
    epsilon = 1e-9
    coordinate_units = str(df.attrs.get("coordinate_units", "px"))
    metric_x_col = lambda side: f"{side}_x_metric_filtered" if f"{side}_x_metric_filtered" in df else f"{side}_x_filtered"
    metric_y_col = lambda side: f"{side}_y_metric_filtered" if f"{side}_y_metric_filtered" in df else f"{side}_y_filtered"
    if df.empty:
        raise ValueError("Nelze vypočítat metriky z prázdného úseku.")

    timestamps = pd.to_numeric(df["timestamp"], errors="coerce").to_numpy(dtype=float)
    valid_ts = timestamps[np.isfinite(timestamps)]
    if len(valid_ts) >= 2:
        T = max(0.0, float(valid_ts[-1] - valid_ts[0]))
        positive_diffs = np.diff(valid_ts)
        positive_diffs = positive_diffs[positive_diffs > 0]
        fallback_dt = float(np.median(positive_diffs)) if positive_diffs.size else 1.0 / 30.0
    else:
        T = 0.0
        fallback_dt = 1.0 / 30.0

    def hand_kinematics(prefix):
        x = pd.to_numeric(df.get(metric_x_col(prefix), pd.Series(np.nan, index=df.index)), errors="coerce").to_numpy(dtype=float)
        y = pd.to_numeric(df.get(metric_y_col(prefix), pd.Series(np.nan, index=df.index)), errors="coerce").to_numpy(dtype=float)
        valid = np.isfinite(x) & np.isfinite(y) & np.isfinite(timestamps)
        source_col = f"{prefix}_tracking_source_for_metrics"
        if source_col not in df:
            source_col = f"{prefix}_tracking_source"
        if source_col in df:
            sources = df[source_col].fillna("unknown").astype(str).to_numpy(dtype=object)
        else:
            # Compatibility with older records containing only one point definition.
            sources = np.full(len(df), "legacy_single_point", dtype=object)
        vx = np.full(len(df), np.nan, dtype=float)
        vy = np.full(len(df), np.nan, dtype=float)
        speed = np.full(len(df), np.nan, dtype=float)
        path_length = 0.0
        has_path_segment = False
        normalized_jerk_runs = []
        run_level_jerk = []
        jerk_magnitudes = []
        has_valid = bool(valid.any())

        use_strokes = NORM_JERK_MODE == "per_stroke" and coordinate_units == "cm"
        for run in _contiguous_tracking_runs(valid, sources):
            n_run = run.stop - run.start
            if n_run < 2:
                continue
            t_run = timestamps[run]
            x_run, y_run = x[run], y[run]
            # Guard against invalid/repeated timestamps in externally supplied data.
            if np.any(np.diff(t_run) <= 0):
                t_run = np.arange(len(t_run), dtype=float) * fallback_dt
            run_vx = np.gradient(x_run, t_run)
            run_vy = np.gradient(y_run, t_run)
            run_speed = np.hypot(run_vx, run_vy)
            vx[run], vy[run], speed[run] = run_vx, run_vy, run_speed
            run_path_length = float(
                np.sum(np.hypot(np.diff(x_run), np.diff(y_run)))
            )
            path_length += run_path_length
            has_path_segment = True

            run_duration = float(t_run[-1] - t_run[0])
            if run_duration <= 0 or run_path_length <= epsilon:
                continue
            # Third derivative from a centred local polynomial fit. It is NaN within
            # half a window of both run ends (including every fingertip <-> palm
            # switch, which always starts a new run), so no switch can create a spike.
            jx = _local_polynomial_derivative(x_run, t_run, derivative_order=3)
            jy = _local_polynomial_derivative(y_run, t_run, derivative_order=3)
            jerk_mag = np.hypot(jx, jy)
            valid_jerk = np.isfinite(jerk_mag)
            if not np.any(valid_jerk):
                continue
            jerk_magnitudes.append(jerk_mag[valid_jerk])

            def integrate(values_, times_):
                if hasattr(np, "trapezoid"):
                    return float(np.trapezoid(values_, times_))
                return float(np.trapz(values_, times_))  # NumPy < 2.0

            if use_strokes:
                # Dimensionless jerk is defined for ONE movement. Evaluating it over a
                # whole multi-stroke run makes the value grow with the stroke count,
                # so it is evaluated per stroke and the median is reported.
                for a, b in _segment_strokes(t_run, run_speed):
                    covered = valid_jerk[a:b]
                    if covered.size == 0 or float(covered.mean()) < STROKE_MIN_JERK_COVERAGE:
                        continue  # stroke touches a run end: truncating it would bias T
                    valid_idx = np.flatnonzero(covered)
                    sub_a, sub_b = a + int(valid_idx[0]), a + int(valid_idx[-1]) + 1
                    if sub_b - sub_a < 3:
                        continue
                    t_s = t_run[sub_a:sub_b]
                    stroke_duration = float(t_s[-1] - t_s[0])
                    stroke_length = float(np.sum(np.hypot(
                        np.diff(x_run[sub_a:sub_b]), np.diff(y_run[sub_a:sub_b])
                    )))
                    if stroke_duration < STROKE_MIN_SECONDS or stroke_length < STROKE_MIN_PATH_CM:
                        continue
                    energy = integrate(jerk_mag[sub_a:sub_b] ** 2, t_s)
                    if np.isfinite(energy) and energy >= 0:
                        value = float(np.sqrt(0.5 * (stroke_duration ** 5) * energy) / (stroke_length + epsilon))
                        if np.isfinite(value):
                            normalized_jerk_runs.append((value, stroke_length))

            # Whole-run value over the span with valid jerk. In stroke mode it is only a
            # fallback for recordings without a single complete stroke and is reported
            # as such, because it is not comparable with per-stroke values.
            jerk_idx = np.flatnonzero(valid_jerk)
            if jerk_idx.size >= 4:
                v_a, v_b = int(jerk_idx[0]), int(jerk_idx[-1]) + 1
                t_v = t_run[v_a:v_b]
                v_duration = float(t_v[-1] - t_v[0])
                v_length = float(np.sum(np.hypot(np.diff(x_run[v_a:v_b]), np.diff(y_run[v_a:v_b]))))
                if v_duration > 0 and v_length > epsilon:
                    energy = integrate(jerk_mag[v_a:v_b] ** 2, t_v)
                    if np.isfinite(energy) and energy >= 0:
                        value = float(np.sqrt(0.5 * (v_duration ** 5) * energy) / (v_length + epsilon))
                        if np.isfinite(value):
                            (run_level_jerk if use_strokes else normalized_jerk_runs).append((value, v_length))

        finite_speed = speed[np.isfinite(speed)]
        finite_jerk = np.concatenate(jerk_magnitudes) if jerk_magnitudes else np.asarray([], dtype=float)
        max_jerk = float(np.max(finite_jerk)) if finite_jerk.size else None
        p95_jerk = float(np.percentile(finite_jerk, 95)) if finite_jerk.size else None
        max_speed = float(np.max(finite_speed)) if finite_speed.size else None
        mean_speed = float(np.mean(finite_speed)) if finite_speed.size else None
        p95_speed = float(np.percentile(finite_speed, 95)) if finite_speed.size else None
        norm_jerk_source = None
        if use_strokes and normalized_jerk_runs:
            # Median over strokes: robust against a single noisy stroke.
            norm_jerk = float(np.median([value for value, _ in normalized_jerk_runs]))
            norm_jerk_source = "tahy"
        else:
            fallback = run_level_jerk if use_strokes else normalized_jerk_runs
            total_weight = sum(weight for _, weight in fallback)
            if fallback and total_weight > epsilon:
                # Weighted by the distance of each tracked run.
                norm_jerk = float(sum(value * weight for value, weight in fallback) / total_weight)
                norm_jerk_source = "cely_usek"
            else:
                norm_jerk = None
        return {
            "vx": vx,
            "vy": vy,
            "speed": speed,
            "path_length": path_length if has_path_segment else None,
            "norm_jerk": norm_jerk,
            "norm_jerk_source": norm_jerk_source,
            "max_jerk": max_jerk,
            "p95_jerk": p95_jerk,
            "max_speed": max_speed,
            "mean_speed": mean_speed,
            "p95_speed": p95_speed,
            "has_data": has_valid,
        }

    kin_L = hand_kinematics("L")
    kin_R = hand_kinematics("R")
    has_L, has_R = kin_L["has_data"], kin_R["has_data"]

    path_diff = None
    path_asymmetry_percent = None
    r_LR = r_vx = r_vy = None
    if (
        has_L and has_R and compare_hands
        and kin_L["path_length"] is not None and kin_R["path_length"] is not None
    ):
        path_diff = abs(kin_L["path_length"] - kin_R["path_length"])
        path_mean = (kin_L["path_length"] + kin_R["path_length"]) / 2.0
        if path_mean > epsilon:
            path_asymmetry_percent = path_diff / path_mean * 100.0
    if has_L and has_R and compare_hands:
        r_LR = _safe_corr(kin_L["speed"], kin_R["speed"])
        r_vx = _safe_corr(kin_L["vx"], kin_R["vx"])
        r_vy = _safe_corr(kin_L["vy"], kin_R["vy"])

    def process_angles(prefix):
        col_name = f"{prefix}_elbow_angle"
        if col_name not in df:
            return {"max_deg": None, "rom_deg": None}
        values = pd.to_numeric(df[col_name], errors="coerce").to_numpy(dtype=float)
        # A running median removes single-frame pose glitches that would otherwise
        # dominate max and range-of-motion (both are extreme-value statistics).
        values = _median_filter_series(timestamps, values, ELBOW_MEDIAN_WINDOW_SECONDS)
        values = values[np.isfinite(values)]
        if values.size == 0:
            return {"max_deg": None, "rom_deg": None}
        return {"max_deg": float(np.max(values)), "rom_deg": float(np.max(values) - np.min(values))}

    ang_L = process_angles("L")
    ang_R = process_angles("R")

    crossed_center_L = crossed_center_R = None
    if center_x is not None:
        for side in ("L", "R"):
            x_values = pd.to_numeric(
                df.get(metric_x_col(side), pd.Series(np.nan, index=df.index)), errors="coerce"
            ).to_numpy(dtype=float)
            x_values = x_values[np.isfinite(x_values)]
            if x_values.size:
                margin = CENTER_LINE_MARGIN_CM if coordinate_units == "cm" else 0.0
                crossed = bool(
                    np.min(x_values) < center_x - margin and np.max(x_values) > center_x + margin
                )
                if side == "L":
                    crossed_center_L = crossed
                else:
                    crossed_center_R = crossed

    tempo_cv = {}
    tempo_period_counts = {}
    for side, kin in (("L", kin_L), ("R", kin_R)):
        cv_value, period_count = _estimate_periodic_tempo_cv(
            timestamps,
            pd.to_numeric(df.get(metric_x_col(side), pd.Series(np.nan, index=df.index)), errors="coerce").to_numpy(dtype=float),
            pd.to_numeric(df.get(metric_y_col(side), pd.Series(np.nan, index=df.index)), errors="coerce").to_numpy(dtype=float),
            df[f"{side}_tracking_source_for_metrics"].fillna("unknown").astype(str).to_numpy(dtype=object)
            if f"{side}_tracking_source_for_metrics" in df else (
                df[f"{side}_tracking_source"].fillna("unknown").astype(str).to_numpy(dtype=object)
                if f"{side}_tracking_source" in df else None
            ),
        )
        if not kin["has_data"]:
            cv_value, period_count = None, 0
        tempo_cv[side] = cv_value
        tempo_period_counts[side] = period_count
    available_tempo_cv = [v for v in tempo_cv.values() if v is not None and np.isfinite(v)]
    # The combined figure is the mean of the two hand-specific CVs and is reported only
    # when BOTH hands have one; with a single hand it would just repeat that hand's value
    # under a "both hands" label. Raw cycle durations of the two hands are never mixed,
    # because their mean tempos may differ.
    cv_tempo = float(np.mean(available_tempo_cv)) if len(available_tempo_cv) == 2 else None

    def round_or_none(value, digits=2):
        if value is None or not np.isfinite(value):
            return None
        return round(float(value), digits)

    return {
        "metadata": {
            "activity_id": int(activity_id) if activity_id is not None else None,
            "verze_pipeline": PIPELINE_VERSION,
            "sledovany_bod": "index_fingertip_preferred_palm_center_fallback",
            "prah_extenze_indexu_deg": {
                "pip_enter": INDEX_EXTENSION_ENTER_PIP_DEG,
                "dip_enter": INDEX_EXTENSION_ENTER_DIP_DEG,
                "pip_exit": INDEX_EXTENSION_EXIT_PIP_DEG,
                "dip_exit": INDEX_EXTENSION_EXIT_DIP_DEG,
            },
            "aktivni_zona_rozsireni_px": float(ACTIVITY_ZONE_MARGIN_PX),
            "doba_pohybu_s": round(float(T), 3),
            "jednotky_souradnic": "cm" if coordinate_units == "cm" else "px",
            "jednotky_rychlosti": "cm/s" if coordinate_units == "cm" else "px/s",
            "kalibrace_desky": "homografie_57cm" if coordinate_units == "cm" else "nekalibrovano_px",
            "metoda_odhadu_jerk": (
                f"lokalni_polynom_stupne_{JERK_POLY_DEGREE}_okno_{JERK_WINDOW_SECONDS:g}s"
                "_okraje_useku_vyrazeny"
            ),
            "vyhlazeni_polohy": (
                f"lokalni_polynom_stupne_{SMOOTHING_POLY_DEGREE}_okno_{SMOOTHING_WINDOW_SECONDS:g}s_bez_zpozdeni"
            ),
            "metoda_norm_jerk": (
                "median_pres_tahy" if (NORM_JERK_MODE == "per_stroke" and coordinate_units == "cm")
                else "vazeny_prumer_pres_useky"
            ),
            "metoda_uhlu_lokte": ELBOW_ANGLE_MODE,
            "limit_prijate_rychlosti_bodu_cm_s": MAX_MEASURED_POINT_SPEED_CM_S,
            "limit_identity_tracking_cm_s": MAX_IDENTITY_SPEED_CM_S,
            "debounce_prepnuti_zdroje_s": SOURCE_SWITCH_MIN_SECONDS,
            "kvalita_trackingu": df.attrs.get("tracking_quality"),
        },
        "kinematika": {
            ("delka_drahy_cm" if coordinate_units == "cm" else "delka_drahy_px"): {
                "L": round_or_none(kin_L["path_length"], 2),
                "R": round_or_none(kin_R["path_length"], 2),
                "rozdil": round_or_none(path_diff, 2),
                "asymetrie_percent": round_or_none(path_asymmetry_percent, 2),
            },
            "plynulost_norm_jerk": {
                "L": round_or_none(kin_L["norm_jerk"], 4),
                "R": round_or_none(kin_R["norm_jerk"], 4),
            },
            "plynulost_norm_jerk_zdroj": {
                "L": kin_L["norm_jerk_source"],
                "R": kin_R["norm_jerk_source"],
            },
            ("max_jerk_cm_s3" if coordinate_units == "cm" else "max_jerk_px_s3"): {
                "L": round_or_none(kin_L["max_jerk"], 2),
                "R": round_or_none(kin_R["max_jerk"], 2),
            },
            ("p95_jerk_cm_s3" if coordinate_units == "cm" else "p95_jerk_px_s3"): {
                "L": round_or_none(kin_L["p95_jerk"], 2),
                "R": round_or_none(kin_R["p95_jerk"], 2),
            },
            "rychlost": {
                "max_L": round_or_none(kin_L["max_speed"], 2),
                "max_R": round_or_none(kin_R["max_speed"], 2),
                "mean_L": round_or_none(kin_L["mean_speed"], 2),
                "mean_R": round_or_none(kin_R["mean_speed"], 2),
                "p95_L": round_or_none(kin_L["p95_speed"], 2),
                "p95_R": round_or_none(kin_R["p95_speed"], 2),
            },
        },
        "synchronizace_a_symetrie": {
            "korelace_rychlosti_r_LR": round_or_none(r_LR, 4),
            "korelace_ve_smeru_x_r_vx": round_or_none(r_vx, 4),
            "korelace_ve_smeru_y_r_vy": round_or_none(r_vy, 4),
        },
        "uhly_a_rozsah": {
            "max_extenze_deg": {
                "L": round_or_none(ang_L["max_deg"], 2),
                "R": round_or_none(ang_R["max_deg"], 2),
            },
            "rom_deg": {
                "L": round_or_none(ang_L["rom_deg"], 2),
                "R": round_or_none(ang_R["rom_deg"], 2),
            },
        },
        "rytmicita_a_cykly": {
            "stabila_tempa_cv_percent": round_or_none(cv_tempo, 2),
            "stabila_tempa_cv_percent_L": round_or_none(tempo_cv["L"], 2),
            "stabila_tempa_cv_percent_R": round_or_none(tempo_cv["R"], 2),
            "pocet_detectovanych_period_L": int(tempo_period_counts["L"]),
            "pocet_detectovanych_period_R": int(tempo_period_counts["R"]),
        },
        "prostorova_presnost": {
            "prekroceni_stredove_linie": {
                "L": crossed_center_L,
                "R": crossed_center_R,
            }
        },

    }


# =========================================================================
# 3. VIDEO DETECTION / SEGMENTATION
# =========================================================================
def _create_marker_detector():
    """Create an ArUco/AprilTag detector using the configured OpenCV dictionary."""
    if not hasattr(cv2, "aruco"):
        raise RuntimeError(
            "Není dostupný modul cv2.aruco. Nainstalujte opencv-contrib-python."
        )
    aruco = cv2.aruco
    dictionary_id = getattr(aruco, MARKER_DICTIONARY_NAME, None)
    if dictionary_id is None:
        raise RuntimeError(
            f"OpenCV nezná slovník {MARKER_DICTIONARY_NAME!r}. "
            "Zkontrolujte MARKER_DICTIONARY_NAME a verzi opencv-contrib-python."
        )
    dictionary = aruco.getPredefinedDictionary(dictionary_id)
    if hasattr(aruco, "ArucoDetector"):
        parameters = aruco.DetectorParameters()
        return (aruco.ArucoDetector(dictionary, parameters), dictionary, parameters)
    parameters = aruco.DetectorParameters_create()
    return (None, dictionary, parameters)


def _detect_markers(gray, detector, dictionary, parameters):
    if detector is not None:
        return detector.detectMarkers(gray)
    return cv2.aruco.detectMarkers(gray, dictionary, parameters=parameters)


def _empty_record(frame_index, timestamp, width, height):
    return {
        "frame_id": int(frame_index),
        "timestamp": float(timestamp),
        # Legacy hand_center fields now hold the selected hybrid tracking point.
        "L_hand_center": None,
        "R_hand_center": None,
        "L_hand_center_raw": None,
        "R_hand_center_raw": None,
        "L_tracking_point": None,
        "R_tracking_point": None,
        "L_tracking_point_raw": None,
        "R_tracking_point_raw": None,
        "L_tracking_source": None,
        "R_tracking_source": None,
        "L_raw_tracking_source": None,
        "R_raw_tracking_source": None,
        "L_index_tip_raw": None,
        "R_index_tip_raw": None,
        "L_palm_center_raw": None,
        "R_palm_center_raw": None,
        "L_index_extended": None,
        "R_index_extended": None,
        "L_index_pip_angle_deg": None,
        "R_index_pip_angle_deg": None,
        "L_index_dip_angle_deg": None,
        "R_index_dip_angle_deg": None,
        "L_detected": False,
        "R_detected": False,
        "L_raw_detection": False,
        "R_raw_detection": False,
        "L_tracking_point_rejected": False,
        "R_tracking_point_rejected": False,
        "L_rejection_reason": None,
        "R_rejection_reason": None,
        "L_identity_corrected": False,
        "R_identity_corrected": False,
        "L_identity_ambiguous": False,
        "R_identity_ambiguous": False,
        "L_handedness_label_raw": None,
        "R_handedness_label_raw": None,
        "L_association_distance_px": None,
        "R_association_distance_px": None,
        "L_observed_speed_cm_s": None,
        "R_observed_speed_cm_s": None,
        "L_detection_confidence": None,
        "R_detection_confidence": None,
        "L_in_zone": False,
        "R_in_zone": False,
        "L_elbow_angle": None,
        "R_elbow_angle": None,
    }


def _landmark_xyz(hand_landmarks, landmark_index, roi_width, roi_height):
    """Convert a normalized hand landmark to an approximately isotropic 3-D vector."""
    lm = hand_landmarks.landmark[landmark_index]
    point = np.asarray(
        [float(lm.x) * roi_width, float(lm.y) * roi_height, float(lm.z) * roi_width],
        dtype=float,
    )
    return point if np.all(np.isfinite(point)) else None


def _joint_angle_degrees(a, b, c):
    """Return the 3-D angle ABC in degrees, or None for degenerate geometry."""
    if a is None or b is None or c is None:
        return None
    ba = np.asarray(a, dtype=float) - np.asarray(b, dtype=float)
    bc = np.asarray(c, dtype=float) - np.asarray(b, dtype=float)
    denom = float(np.linalg.norm(ba) * np.linalg.norm(bc))
    if not np.isfinite(denom) or denom <= 1e-9:
        return None
    cosine = float(np.clip(np.dot(ba, bc) / denom, -1.0, 1.0))
    return float(np.degrees(np.arccos(cosine)))


def _palm_center_from_landmarks(hand_landmarks, roi_x, roi_y, roi_width, roi_height):
    """Return the palm-centre estimate in full-frame pixel coordinates.

    MediaPipe landmark 0 is the wrist; 5, 9, 13 and 17 are the four MCP knuckles.
    """
    landmarks = hand_landmarks.landmark
    points = np.asarray(
        [
            [roi_x + float(landmarks[i].x) * roi_width,
             roi_y + float(landmarks[i].y) * roi_height]
            for i in PALM_LANDMARK_INDICES
        ],
        dtype=float,
    )
    if not np.all(np.isfinite(points)):
        return None
    center = np.mean(points, axis=0)
    return float(center[0]), float(center[1])


def _index_tip_and_extension(hand_landmarks, roi_x, roi_y, roi_width, roi_height):
    """Return index fingertip, PIP/DIP extension angles and geometric validity.

    Hand landmark 8 is the index fingertip. The index finger is treated as usable
    when both the PIP (5-6-7) and DIP (6-7-8) joints are extended. These angles are
    estimated in MediaPipe's relative 3-D landmark space, not anatomical-grade 3-D.
    """
    landmarks = hand_landmarks.landmark
    tip_lm = landmarks[8]
    # Normalized landmarks slightly outside [0, 1] can occur at crop edges, but
    # large excursions indicate that the fingertip is not reliably localized.
    if not (
        np.isfinite(float(tip_lm.x)) and np.isfinite(float(tip_lm.y))
        and -0.05 <= float(tip_lm.x) <= 1.05
        and -0.05 <= float(tip_lm.y) <= 1.05
    ):
        return None, None, None
    tip_xy = np.asarray(
        [roi_x + float(tip_lm.x) * roi_width,
         roi_y + float(tip_lm.y) * roi_height],
        dtype=float,
    )
    if not np.all(np.isfinite(tip_xy)):
        return None, None, None

    p5 = _landmark_xyz(hand_landmarks, 5, roi_width, roi_height)
    p6 = _landmark_xyz(hand_landmarks, 6, roi_width, roi_height)
    p7 = _landmark_xyz(hand_landmarks, 7, roi_width, roi_height)
    p8 = _landmark_xyz(hand_landmarks, 8, roi_width, roi_height)
    pip_angle = _joint_angle_degrees(p5, p6, p7)
    dip_angle = _joint_angle_degrees(p6, p7, p8)
    return (float(tip_xy[0]), float(tip_xy[1])), pip_angle, dip_angle


def _select_tracking_reference(
    hand_landmarks, roi_x, roi_y, roi_width, roi_height, previous_source="palm_center"
):
    """Select index fingertip when extended; otherwise use the palm centre.

    Returns selected_point, palm_center, index_tip, source, index_extended,
    pip_angle, dip_angle. Threshold hysteresis makes source changes less jittery.
    """
    palm_center = _palm_center_from_landmarks(
        hand_landmarks, roi_x, roi_y, roi_width, roi_height
    )
    if palm_center is None:
        return None, None, None, None, False, None, None

    index_tip, pip_angle, dip_angle = _index_tip_and_extension(
        hand_landmarks, roi_x, roi_y, roi_width, roi_height
    )
    was_index = previous_source == "index_tip"
    pip_threshold = (
        INDEX_EXTENSION_EXIT_PIP_DEG if was_index
        else INDEX_EXTENSION_ENTER_PIP_DEG
    )
    dip_threshold = (
        INDEX_EXTENSION_EXIT_DIP_DEG if was_index
        else INDEX_EXTENSION_ENTER_DIP_DEG
    )
    index_extended = bool(
        index_tip is not None
        and pip_angle is not None
        and dip_angle is not None
        and pip_angle >= pip_threshold
        and dip_angle >= dip_threshold
    )
    if index_extended:
        return index_tip, palm_center, index_tip, "index_tip", True, pip_angle, dip_angle
    return palm_center, palm_center, index_tip, "palm_center", False, pip_angle, dip_angle


class _SourceDebouncer:
    """Debounce the fingertip <-> palm decision of one hand.

    The raw per-frame decision (index extended or not) may flicker for a few frames
    whenever MediaPipe's finger joints are uncertain. Every switch of the measured point
    ends a continuous kinematic run, so flicker would cut the trajectory into many short
    runs. The active source therefore changes only after the raw decision has disagreed
    with it continuously for ``min_seconds``. A long detection gap resets the state.
    """

    def __init__(self, min_seconds: float = SOURCE_SWITCH_MIN_SECONDS,
                 stale_seconds: float = IDENTITY_STALE_AFTER_SECONDS):
        self.min_seconds = float(min_seconds)
        self.stale_seconds = float(stale_seconds)
        self.stable = None
        self.pending_since = None
        self.last_time = None

    def update(self, raw_source: str, timestamp: float) -> str:
        if (
            self.stable is None or self.last_time is None
            or timestamp - self.last_time > self.stale_seconds
        ):
            self.stable, self.pending_since = raw_source, None
        elif raw_source == self.stable:
            self.pending_since = None
        else:
            if self.pending_since is None:
                self.pending_since = timestamp
            if timestamp - self.pending_since >= self.min_seconds:
                self.stable, self.pending_since = raw_source, None
        self.last_time = timestamp
        return self.stable

    def force(self, source: str) -> None:
        self.stable, self.pending_since = source, None


class TemporalHandIdentityTracker:
    """Associate frame detections with persistent anatomical L/R tracks.

    MediaPipe handedness is treated as a prior, not as an unquestionable identity.
    The palm centre is used for association because the index fingertip can move
    relative to the palm and can switch to a palm fallback when the finger folds.
    """

    def __init__(self, board_diagonal_px: float, fps: float):
        self.board_diagonal_px = max(1.0, float(board_diagonal_px))
        self.fps = max(1.0, float(fps))
        self.history = {"L": [], "R": []}
        self.label_corrections = {"L": 0, "R": 0}
        self.rejected_jumps = {"L": 0, "R": 0}

    @property
    def px_per_cm(self) -> float:
        # Board diagonal is 57*sqrt(2) cm.
        return self.board_diagonal_px / (BOARD_SIDE_CM * np.sqrt(2.0))

    def _recent(self, side: str, timestamp: float):
        history = self.history[side]
        if not history:
            return None
        last = history[-1]
        if timestamp - last[0] > IDENTITY_STALE_AFTER_SECONDS:
            return None
        return history

    def _predict(self, side: str, timestamp: float):
        history = self._recent(side, timestamp)
        if not history:
            return None, None
        last_t, last_xy = history[-1]
        predicted = np.asarray(last_xy, dtype=float)
        if len(history) >= 2:
            prev_t, prev_xy = history[-2]
            dt_hist = last_t - prev_t
            if dt_hist > 1e-6:
                velocity = (np.asarray(last_xy) - np.asarray(prev_xy)) / dt_hist
                speed = float(np.linalg.norm(velocity))
                max_speed_px_s = MAX_IDENTITY_SPEED_CM_S * self.px_per_cm
                if speed > max_speed_px_s:
                    velocity *= max_speed_px_s / speed
                predicted = predicted + velocity * max(0.0, timestamp - last_t)
        return predicted, last_t

    def _gate_px(self, dt: float) -> float:
        dt = max(float(dt), 1.0 / self.fps)
        expected = MAX_IDENTITY_SPEED_CM_S * self.px_per_cm * dt
        return max(35.0, expected * 1.6 + 18.0)

    def _candidate_cost(self, candidate: dict, side: str, timestamp: float):
        predicted, last_t = self._predict(side, timestamp)
        label_side = candidate["label_side"]
        if predicted is None:
            # A first observation is assigned primarily by handedness. Penalise
            # an opposite assignment so two newly appearing hands retain labels.
            mismatch = side != label_side
            base = self.board_diagonal_px * (0.35 if mismatch else 0.0)
            return base, None, None

        dt = max(0.0, timestamp - last_t)
        gate = self._gate_px(dt)
        distance = float(np.linalg.norm(
            np.asarray(candidate["palm_center"], dtype=float) - predicted
        ))
        # Continuity dominates handedness when the label would require an
        # impossible jump. Handedness only breaks close/ambiguous matches.
        label_penalty = (
            self.board_diagonal_px * IDENTITY_LABEL_PENALTY_BOARD_DIAGONALS
            if side != label_side else 0.0
        )
        excess_penalty = max(0.0, distance - gate) * 5.0
        return distance + label_penalty + excess_penalty, distance, gate

    def assign(self, candidates: list[dict], timestamp: float):
        """Return candidate dictionaries annotated with side and association status."""
        if not candidates:
            return []

        ambiguous_assignment = False
        if len(candidates) == 1:
            candidate = candidates[0]
            label_side = candidate["label_side"]
            other_side = "R" if label_side == "L" else "L"
            label_cost, label_distance, label_gate = self._candidate_cost(
                candidate, label_side, timestamp
            )
            other_cost, other_distance, other_gate = self._candidate_cost(
                candidate, other_side, timestamp
            )

            side = label_side
            # Override a flipped MediaPipe label only when the labelled track
            # would jump outside its motion gate and the opposite track is a
            # plausible continuation.
            if (
                label_distance is not None
                and label_gate is not None
                and label_distance > label_gate
                and other_distance is not None
                and other_gate is not None
                and other_distance <= other_gate
            ):
                side = other_side
            elif (
                label_distance is None
                and other_distance is not None
                and other_gate is not None
                and other_distance <= min(other_gate * 0.35, self.board_diagonal_px * 0.12)
            ):
                # This handles a one-frame handedness flip when the other
                # anatomical hand is already tracked and very close to prediction.
                side = other_side

            if (
                label_distance is not None
                and other_distance is not None
                and self._recent("L", timestamp) is not None
                and self._recent("R", timestamp) is not None
                and abs(label_distance - other_distance) < self.board_diagonal_px * 0.06
            ):
                # At a near-overlap, do not guess which hand generated the only
                # remaining detection. The gap is safer than a false identity swap.
                ambiguous_assignment = True

            chosen_cost, distance, gate = self._candidate_cost(candidate, side, timestamp)
            assigned = [(candidate, side, distance, gate)]
        else:
            # MediaPipe returns at most two hands. Evaluate both possible
            # one-to-one assignments and choose the temporally coherent pairing.
            c0, c1 = candidates[0], candidates[1]
            alternatives = []
            for mapping in (("L", "R"), ("R", "L")):
                cost0, dist0, gate0 = self._candidate_cost(c0, mapping[0], timestamp)
                cost1, dist1, gate1 = self._candidate_cost(c1, mapping[1], timestamp)
                alternatives.append((
                    cost0 + cost1,
                    [(c0, mapping[0], dist0, gate0), (c1, mapping[1], dist1, gate1)],
                ))
            alternatives.sort(key=lambda item: item[0])
            _, assigned = alternatives[0]
            if (
                self._recent("L", timestamp) is not None
                and self._recent("R", timestamp) is not None
                and len(alternatives) > 1
                and abs(alternatives[1][0] - alternatives[0][0])
                < self.board_diagonal_px * 0.06
            ):
                ambiguous_assignment = True

        output = []
        for candidate, side, distance, gate in assigned:
            accepted = distance is None or gate is None or distance <= gate
            corrected = side != candidate["label_side"]
            candidate["assigned_side"] = side
            candidate["identity_corrected"] = bool(corrected)
            candidate["association_distance_px"] = distance
            candidate["identity_ambiguous"] = bool(ambiguous_assignment)
            candidate["association_accepted"] = bool(accepted and not ambiguous_assignment)
            if corrected:
                self.label_corrections[side] += 1
            if ambiguous_assignment:
                self.rejected_jumps[side] += 1
            elif not accepted:
                self.rejected_jumps[side] += 1
            else:
                history = self.history[side]
                history.append((float(timestamp), tuple(map(float, candidate["palm_center"]))))
                self.history[side] = history[-2:]
            output.append(candidate)
        return output


def _make_activity_polygon(board_polygon: Polygon) -> Polygon:
    """Expand the activity zone by a fixed image-pixel margin."""
    expanded = board_polygon.buffer(float(ACTIVITY_ZONE_MARGIN_PX), join_style=2)
    if expanded.is_empty or not expanded.is_valid or expanded.area <= board_polygon.area:
        raise ValueError("Nelze vytvořit rozšířenou herní zónu.")
    return expanded


def _polygon_outline_np(polygon: Polygon) -> np.ndarray:
    """Convert a Shapely polygon exterior to an OpenCV contour."""
    return np.round(np.asarray(polygon.exterior.coords, dtype=float)).astype(np.int32).reshape(-1, 1, 2)


def _crop_to_polygon_bounds(frame, polygon: Polygon):
    """Crop the bounding rectangle of a polygon, clipped to the source image."""
    height, width = frame.shape[:2]
    min_x, min_y, max_x, max_y = polygon.bounds
    x0 = max(0, int(np.floor(min_x)))
    y0 = max(0, int(np.floor(min_y)))
    x1 = min(width, int(np.ceil(max_x)) + 1)
    y1 = min(height, int(np.ceil(max_y)) + 1)
    if x1 <= x0 or y1 <= y0:
        raise RuntimeError("Rozšířená oblast hry leží mimo obraz videa.")
    return frame[y0:y1, x0:x1], x0, y0, x1 - x0, y1 - y0


def _record_has_relevant_hand_in_zone(record, cfg):
    return bool(
        (cfg["track_left_hand"] and record.get("L_in_zone", False))
        or (cfg["track_right_hand"] and record.get("R_in_zone", False))
    )


def _trim_trailing_outside_records(records, cfg):
    while records and not _record_has_relevant_hand_in_zone(records[-1], cfg):
        records.pop()
    return records


def _collect_segments(
    video_path: str,
    mode: str,
    activity_id: Optional[int] = None,
    frame_callback: Optional[Callable] = None,
    cancel_event=None,
    video_number: int = 1,
    video_count: int = 1,
):
    """Detect one activity; frame_callback receives native decoded BGR frames (H, W, 3).

    For the fixed camera, frames are portrait 1920 x 1080 (height x width) and
    are never resized or stretched by this pipeline. The UI must preserve 9:16.

    Activity presence is true when either the selected reference point or palm
    centre is in the buffered polygon. The tracked coordinate is the index fingertip
    when extended, otherwise the palm-centre fallback. The activity ends after all
    relevant hands remain outside the buffered polygon for EXIT_CONFIRM_SECONDS.
    """
    if mode not in ("single", "batch"):
        raise ValueError("mode musí být 'single' nebo 'batch'.")
    if not os.path.isfile(video_path):
        raise FileNotFoundError(f"Video neexistuje: {video_path}")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        cap.release()
        raise RuntimeError(f"Video se nepodařilo otevřít: {video_path}")

    total_frames = max(1, int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 1))
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    if not np.isfinite(fps) or fps <= 0:
        fps = EXPECTED_VIDEO_FPS
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    if width <= 0 or height <= 0:
        cap.release()
        raise RuntimeError(f"Video má neplatné rozměry: {video_path}")
    # CAP_PROP dimensions can describe the encoded stream before rotation metadata
    # is applied. The decoded frame shape is checked again on the first frame below.
    video_format_warnings = []
    if abs(fps - EXPECTED_VIDEO_FPS) > VIDEO_FPS_TOLERANCE:
        video_format_warnings.append(
            f"Očekáváno {EXPECTED_VIDEO_FPS:.2f} fps, načteno {fps:.3f} fps."
        )

    pose = None
    hands = None
    try:
        marker_detector, marker_dictionary, marker_parameters = _create_marker_detector()
        pose = mp.solutions.pose.Pose(
            model_complexity=1,
            smooth_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        hands = mp.solutions.hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            model_complexity=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )
    except Exception:
        cap.release()
        if hands is not None:
            hands.close()
        if pose is not None:
            pose.close()
        raise

    cfg = get_activity_config(activity_id) if mode == "single" else {
        "track_left_hand": True,
        "track_right_hand": True,
        "track_elbows": True,
    }

    board_polygon = None
    activity_polygon = None
    board_hull_np = None
    activity_hull_np = None
    homography = None
    roi_bounds = None
    identity_tracker = None
    measurement_state = {"L": None, "R": None}
    marker_samples = {marker_id: [] for marker_id in MARKER_IDS}

    frame_index = 0
    previous_timestamp = -1.0
    segments = []
    active_records = []
    started = False
    exit_pending_since = None
    # Source hysteresis state prevents rapid index-tip/palm switching near the
    # extension threshold. The first accepted detection starts with a palm fallback.
    tracking_source_state = {"L": "palm_center", "R": "palm_center"}
    source_debouncers = {"L": _SourceDebouncer(), "R": _SourceDebouncer()}
    cancelled = False
    stop_reason = None

    try:
        while cap.isOpened():
            if cancel_event is not None and cancel_event.is_set():
                cancelled = True
                break

            ret, frame = cap.read()
            if not ret:
                break

            frame_height, frame_width = frame.shape[:2]
            if frame_index == 0:
                # All coordinates, overlays, and exported frames use the actual
                # decoded portrait frame, never the container's nominal dimensions.
                width, height = int(frame_width), int(frame_height)
                video_format_warnings = [
                    warning for warning in video_format_warnings
                    if "fps" in warning.lower()
                ]
                if (width, height) not in ALLOWED_VIDEO_SIZES:
                    video_format_warnings.append(
                        f"Očekáváno portrétní video {EXPECTED_VIDEO_WIDTH}x{EXPECTED_VIDEO_HEIGHT}, "
                        f"dekódováno {width}x{height}."
                    )
            elif (frame_width, frame_height) != (width, height):
                raise RuntimeError(
                    "Rozměry snímků se během videa změnily: "
                    f"očekáváno {width}x{height}, načteno {frame_width}x{frame_height}."
                )

            timestamp = float(cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0)
            if (
                not np.isfinite(timestamp)
                or timestamp < 0
                or (frame_index > 0 and timestamp <= previous_timestamp)
            ):
                timestamp = frame_index / fps
            previous_timestamp = timestamp

            display_frame = frame.copy()

            # The board is static, so marker detection is only needed until a
            # stable calibration has been accumulated.
            if board_polygon is None:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                corners, ids, _ = _detect_markers(
                    gray, marker_detector, marker_dictionary, marker_parameters
                )
                if ids is not None:
                    for marker_index, marker_id in enumerate(ids.flatten()):
                        marker_id = int(marker_id)
                        if marker_id not in MARKER_IDS:
                            continue
                        marker_corners = corners[marker_index][0]
                        center = np.mean(marker_corners, axis=0)
                        marker_samples[marker_id].append(center.astype(float))
                        if len(marker_samples[marker_id]) > MARKER_SAMPLE_WINDOW:
                            marker_samples[marker_id].pop(0)

                samples_ready = all(
                    len(marker_samples[marker_id]) >= MARKER_CALIBRATION_SAMPLES
                    for marker_id in MARKER_IDS
                )
                if samples_ready:
                    centers_by_id = {
                        marker_id: np.median(
                            np.asarray(marker_samples[marker_id], dtype=float), axis=0
                        )
                        for marker_id in MARKER_IDS
                    }
                    raw_points = np.asarray(
                        [centers_by_id[i] for i in sorted(MARKER_IDS)],
                        dtype=np.float32,
                    )
                    ordered_points = _order_quad_points(raw_points)
                    candidate = Polygon(
                        [tuple(map(float, point)) for point in ordered_points]
                    )
                    contour = np.round(ordered_points).astype(np.int32).reshape(-1, 1, 2)
                    is_convex = cv2.isContourConvex(contour)

                    if (
                        candidate.is_valid
                        and not candidate.is_empty
                        and candidate.area > 100.0
                        and is_convex
                    ):
                        inset = float(MARKER_CENTER_INSET_CM)
                        if inset < 0 or inset >= BOARD_SIDE_CM / 2.0:
                            raise ValueError(
                                "MARKER_CENTER_INSET_CM musí být >= 0 a menší než polovina desky."
                            )
                        target_pts = np.array(
                            [
                                [inset, inset],
                                [BOARD_SIDE_CM - inset, inset],
                                [BOARD_SIDE_CM - inset, BOARD_SIDE_CM - inset],
                                [inset, BOARD_SIDE_CM - inset],
                            ],
                            dtype=np.float32,
                        )
                        candidate_homography = cv2.getPerspectiveTransform(
                            ordered_points, target_pts
                        )
                        determinant = float(np.linalg.det(candidate_homography))
                        if (
                            np.all(np.isfinite(candidate_homography))
                            and np.isfinite(determinant)
                            and abs(determinant) > 1e-12
                        ):
                            board_polygon = candidate
                            activity_polygon = _make_activity_polygon(board_polygon)
                            board_hull_np = _polygon_outline_np(board_polygon)
                            activity_hull_np = _polygon_outline_np(activity_polygon)
                            homography = candidate_homography
                            board_diagonal_px = float(np.linalg.norm(
                                ordered_points[0] - ordered_points[2]
                            ))
                            identity_tracker = TemporalHandIdentityTracker(
                                board_diagonal_px=board_diagonal_px,
                                fps=fps,
                            )
                            _, roi_x, roi_y, roi_w, roi_h = _crop_to_polygon_bounds(
                                frame, activity_polygon
                            )
                            roi_bounds = (roi_x, roi_y, roi_w, roi_h)

            if board_polygon is None:
                # Keep the user preview and progress moving while waiting for tags.
                if frame_callback is not None:
                    fraction = min(1.0, frame_index / float(total_frames))
                    progress = (
                        (video_number - 1 + fraction) / max(1, video_count)
                        if mode == "batch" else fraction
                    )
                    frame_callback(display_frame, progress)
                frame_index += 1
                continue

            # Draw the outer activity zone and the actual calibrated board separately.
            if activity_hull_np is not None:
                cv2.polylines(
                    display_frame, [activity_hull_np], True, (0, 215, 255), 2
                )
            if board_hull_np is not None:
                cv2.polylines(display_frame, [board_hull_np], True, (0, 255, 0), 3)

            x0, y0, roi_w, roi_h = roi_bounds
            roi = frame[y0:y0 + roi_h, x0:x0 + roi_w]
            roi_rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
            hands_res = hands.process(roi_rgb)
            record = _empty_record(frame_index, timestamp, width, height)

            if hands_res.multi_hand_landmarks and hands_res.multi_handedness:
                candidates = []
                for hand_landmarks, handedness in zip(
                    hands_res.multi_hand_landmarks,
                    hands_res.multi_handedness,
                ):
                    classification = handedness.classification[0]
                    label = str(classification.label)
                    confidence = float(classification.score)

                    # MediaPipe Hands labels assume mirrored selfie input. For raw
                    # unmirrored recordings, swap the labels as configured above.
                    if not INPUT_VIDEO_IS_MIRRORED:
                        label = "Right" if label == "Left" else "Left"
                    label_side = "L" if label == "Left" else "R"
                    palm_center = _palm_center_from_landmarks(
                        hand_landmarks, x0, y0, roi_w, roi_h
                    )
                    if palm_center is None:
                        continue
                    candidates.append({
                        "hand_landmarks": hand_landmarks,
                        "label_side": label_side,
                        "raw_label": str(classification.label),
                        "confidence": confidence,
                        "palm_center": palm_center,
                    })

                assigned_candidates = (
                    identity_tracker.assign(candidates, timestamp)
                    if identity_tracker is not None else []
                )

                for candidate in assigned_candidates:
                    side = candidate["assigned_side"]
                    if not cfg[f"track_{'left' if side == 'L' else 'right'}_hand"]:
                        continue

                    palm_center = candidate["palm_center"]
                    palm_in_zone = bool(
                        activity_polygon.covers(Point(palm_center[0], palm_center[1]))
                    )
                    prefix = side
                    record[f"{prefix}_raw_detection"] = True
                    record[f"{prefix}_handedness_label_raw"] = candidate["raw_label"]
                    record[f"{prefix}_identity_corrected"] = candidate["identity_corrected"]
                    record[f"{prefix}_identity_ambiguous"] = candidate["identity_ambiguous"]
                    record[f"{prefix}_association_distance_px"] = candidate["association_distance_px"]
                    record[f"{prefix}_palm_center_raw"] = palm_center
                    record[f"{prefix}_detection_confidence"] = candidate["confidence"]
                    record[f"{prefix}_in_zone"] = palm_in_zone

                    (
                        selected_point,
                        selected_palm,
                        index_tip,
                        source,
                        index_extended,
                        pip_angle,
                        dip_angle,
                    ) = _select_tracking_reference(
                        candidate["hand_landmarks"],
                        x0, y0, roi_w, roi_h,
                        previous_source=tracking_source_state[side],
                    )
                    # Debounce the fingertip <-> palm decision (see _SourceDebouncer). The raw
                    # per-frame decision stays in the diagnostics; the measured point follows
                    # the stable source. The fingertip landmark exists even while the finger
                    # is bent, so keeping it for a few frames is anatomically valid.
                    raw_source = source
                    if selected_palm is not None:
                        stable_source = source_debouncers[side].update(raw_source, timestamp)
                        if stable_source == "index_tip" and index_tip is not None:
                            selected_point, source = index_tip, "index_tip"
                        else:
                            selected_point, source = selected_palm, "palm_center"
                            if stable_source == "index_tip":
                                source_debouncers[side].force("palm_center")
                    if selected_point is not None:
                        record[f"{prefix}_hand_center_raw"] = selected_point
                        record[f"{prefix}_tracking_point_raw"] = selected_point
                        record[f"{prefix}_index_tip_raw"] = index_tip
                        record[f"{prefix}_raw_tracking_source"] = raw_source

                    # An implausible palm jump is not allowed to relabel a hand's
                    # trajectory. Keep the raw detection in diagnostics, but do
                    # not feed it into interpolation, velocity or jerk.
                    if not candidate["association_accepted"]:
                        record[f"{prefix}_tracking_point_rejected"] = True
                        record[f"{prefix}_rejection_reason"] = (
                            "identity_ambiguous"
                            if candidate["identity_ambiguous"] else "identity_jump"
                        )
                        cv2.putText(
                            display_frame, f"{side} REJECT identity jump",
                            (int(palm_center[0]) + 8, int(palm_center[1]) - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 0, 255), 1, cv2.LINE_AA,
                        )
                        continue

                    if selected_point is None or selected_palm is None:
                        record[f"{prefix}_tracking_point_rejected"] = True
                        record[f"{prefix}_rejection_reason"] = "invalid_landmark"
                        continue

                    # Reject impossible jumps of the measurement. The palm centre is a stable
                    # anatomical point shared by both sources, so it is compared on EVERY
                    # frame (also across a fingertip <-> palm switch, which used to be a
                    # blind spot); the selected point itself is compared like-for-like.
                    mapped_point = cv2.perspectiveTransform(
                        np.asarray([[selected_point]], dtype=np.float32), homography
                    ).reshape(2)
                    mapped_palm = cv2.perspectiveTransform(
                        np.asarray([[selected_palm]], dtype=np.float32), homography
                    ).reshape(2)
                    previous_measurement = measurement_state[side]
                    if previous_measurement is not None:
                        elapsed = timestamp - previous_measurement["timestamp"]
                        if elapsed > 1e-6:
                            displacement_cm = float(np.linalg.norm(
                                mapped_palm - previous_measurement["palm_xy"]
                            ))
                            if previous_measurement["source"] == source:
                                displacement_cm = max(displacement_cm, float(np.linalg.norm(
                                    mapped_point - previous_measurement["board_xy"]
                                )))
                            point_speed_cm_s = displacement_cm / elapsed
                            record[f"{prefix}_observed_speed_cm_s"] = point_speed_cm_s
                            if point_speed_cm_s > MAX_MEASURED_POINT_SPEED_CM_S:
                                record[f"{prefix}_tracking_point_rejected"] = True
                                record[f"{prefix}_rejection_reason"] = "measurement_jump"
                                cv2.putText(
                                    display_frame,
                                    f"{side} REJECT {point_speed_cm_s:.0f} cm/s",
                                    (int(selected_point[0]) + 8, int(selected_point[1]) - 8),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 0, 255), 1, cv2.LINE_AA,
                                )
                                continue

                    tracking_source_state[side] = source
                    measurement_state[side] = {
                        "timestamp": float(timestamp),
                        "board_xy": mapped_point.astype(float),
                        "palm_xy": mapped_palm.astype(float),
                        "source": source,
                    }

                    selected_in_zone = bool(
                        activity_polygon.covers(Point(selected_point[0], selected_point[1]))
                    )
                    is_in_zone = bool(palm_in_zone or selected_in_zone)
                    point_int = (
                        int(round(selected_point[0])), int(round(selected_point[1]))
                    )
                    record[f"{prefix}_hand_center_raw"] = selected_point
                    record[f"{prefix}_hand_center"] = selected_point if is_in_zone else None
                    record[f"{prefix}_tracking_point_raw"] = selected_point
                    record[f"{prefix}_tracking_point"] = selected_point if is_in_zone else None
                    record[f"{prefix}_tracking_source"] = source
                    record[f"{prefix}_index_extended"] = index_extended
                    record[f"{prefix}_index_pip_angle_deg"] = pip_angle
                    record[f"{prefix}_index_dip_angle_deg"] = dip_angle
                    record[f"{prefix}_detected"] = True
                    record[f"{prefix}_in_zone"] = is_in_zone

                    color = (255, 80, 20) if side == "L" else (0, 255, 255)
                    cv2.circle(display_frame, point_int, 9, color, -1)
                    palm_int = (int(round(palm_center[0])), int(round(palm_center[1])))
                    cv2.circle(display_frame, palm_int, 5, (255, 255, 255), 1)
                    label_text = f"{side} {'index' if source == 'index_tip' else 'palm fallback'}"
                    if candidate["identity_corrected"]:
                        label_text += " [identity corrected]"
                    cv2.putText(
                        display_frame, label_text,
                        (point_int[0] + 8, point_int[1] - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.48, color, 1, cv2.LINE_AA
                    )

            any_in_zone = _record_has_relevant_hand_in_zone(record, cfg)
            should_start = (not started) and any_in_zone

            # Pose is required for elbow-extension metrics. It is run only while
            # an activity is active; all frames are processed at their native size.
            if cfg["track_elbows"] and (started or should_start):
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pose_res = pose.process(frame_rgb)
                if pose_res.pose_landmarks:
                    lm = pose_res.pose_landmarks.landmark

                    world_lm = (
                        pose_res.pose_world_landmarks.landmark
                        if getattr(pose_res, "pose_world_landmarks", None) else None
                    )

                    def visible_elbow_angle(shoulder_idx, elbow_idx, wrist_idx):
                        joints = (lm[shoulder_idx], lm[elbow_idx], lm[wrist_idx])
                        if min(float(getattr(joint, "visibility", 1.0)) for joint in joints) < POSE_VISIBILITY_THRESHOLD:
                            return None
                        if ELBOW_ANGLE_MODE == "world_3d":
                            # Metric 3-D landmarks: the image-plane angle of an oblique
                            # camera is foreshortened and depends on the viewing angle.
                            if world_lm is None:
                                return None
                            points_3d = [
                                np.asarray(
                                    [world_lm[i].x, world_lm[i].y, world_lm[i].z], dtype=float
                                )
                                for i in (shoulder_idx, elbow_idx, wrist_idx)
                            ]
                            if not np.all(np.isfinite(points_3d)):
                                return None
                            return _joint_angle_degrees(*points_3d)
                        points = [
                            (float(joint.x) * width, float(joint.y) * height)
                            for joint in joints
                        ]
                        if not np.all(np.isfinite(points)):
                            return None
                        return calculate_angle(*points)

                    if cfg["track_left_hand"] and record["L_in_zone"]:
                        record["L_elbow_angle"] = visible_elbow_angle(11, 13, 15)
                    if cfg["track_right_hand"] and record["R_in_zone"]:
                        record["R_elbow_angle"] = visible_elbow_angle(12, 14, 16)

            finish_after_frame = False
            if should_start:
                started = True
                exit_pending_since = None
                active_records = []
                active_records.append(record)
            elif started:
                active_records.append(record)
                if any_in_zone:
                    exit_pending_since = None
                else:
                    if exit_pending_since is None:
                        exit_pending_since = timestamp
                    if timestamp - exit_pending_since >= EXIT_CONFIRM_SECONDS:
                        _trim_trailing_outside_records(active_records, cfg)
                        if active_records:
                            segments.append(active_records)
                        active_records = []
                        started = False
                        stop_reason = (
                            "Aktivita ukončena: všechny sledované ruce zůstaly mimo "
                            f"rozšířenou herní zónu po dobu {EXIT_CONFIRM_SECONDS:.2f} s."
                        )
                        finish_after_frame = True

            if frame_callback is not None:
                fraction = min(1.0, frame_index / float(total_frames))
                progress = (
                    (video_number - 1 + fraction) / max(1, video_count)
                    if mode == "batch" else fraction
                )
                frame_callback(display_frame, progress)

            frame_index += 1
            if finish_after_frame:
                break

        # End-of-file is also a valid activity boundary. Remove trailing frames
        # without a relevant hand reference point/palm in the buffered activity zone.
        if started and active_records:
            _trim_trailing_outside_records(active_records, cfg)
            if active_records:
                segments.append(active_records)

        if board_polygon is None:
            stop_reason = (
                "Nepodařilo se zkalibrovat herní zónu. Ověřte, že jsou v záběru "
                f"značky ID {sorted(MARKER_IDS)} a že odpovídají slovníku "
                f"{MARKER_DICTIONARY_NAME}."
            )
        elif not segments and not cancelled:
            if mode == "single" and activity_id in (8, 9):
                stop_reason = "V rozšířené herní zóně nebyla zachycena příslušná ruka."
            else:
                stop_reason = "V rozšířené herní zóně nebyla zachycena žádná sledovaná ruka."

    finally:
        cap.release()
        if hands is not None:
            hands.close()
        if pose is not None:
            pose.close()

    return {
        "segments": segments,
        "polygon": board_polygon,
        "activity_polygon": activity_polygon,
        "hull": board_hull_np,
        "activity_hull": activity_hull_np,
        "homography": homography,
        "roi_bounds": roi_bounds,
        "zone_margin_px": float(ACTIVITY_ZONE_MARGIN_PX),
        "marker_dictionary": MARKER_DICTIONARY_NAME,
        "marker_center_inset_cm": float(MARKER_CENTER_INSET_CM),
        "fps": fps,
        "width": width,
        "height": height,
        "total_frames": total_frames,
        "cancelled": cancelled,
        "stop_reason": stop_reason,
        "frames_read": frame_index,
        "video_format_warnings": video_format_warnings,
        "expected_video_size": [EXPECTED_VIDEO_WIDTH, EXPECTED_VIDEO_HEIGHT],
        "expected_video_fps": EXPECTED_VIDEO_FPS,
    }



def _find_jump_outliers(
    timestamps: np.ndarray,
    selected_cm: np.ndarray,
    palm_cm: np.ndarray,
    sources: np.ndarray,
    speed_limit_cm_s: float = MAX_MEASURED_POINT_SPEED_CM_S,
    max_excursion_seconds: float = JUMP_MAX_EXCURSION_SECONDS,
):
    """Offline detection of physically impossible position jumps (false detections).

    ``selected_cm`` is the measured point, ``palm_cm`` the palm centre of the same frame
    (both board coordinates in cm, NaN = no observation) and ``sources`` the reference
    point label of every frame. Two consecutive observations are a *jump* when the palm
    centre, or the selected point if the source did not change, moved faster than
    ``speed_limit_cm_s`` (computed from the real time difference, so gaps are fine).

    The jumps cut the observations into pieces. A short piece (<= max_excursion_seconds)
    whose two neighbours join up again is a false excursion (e.g. the tracker latched onto
    something on the other side of the board for a few frames) and is dropped; a short
    piece at the start/end of the segment is dropped when its only neighbour is much
    longer. Jumps that remain (a genuine relocation, e.g. the hand re-entered elsewhere)
    only split the trajectory, so nothing is ever differentiated across them.

    Returns ``(drop_mask, piece_ids)``; ``piece_ids`` is -1 for frames without data.
    """
    t = np.asarray(timestamps, dtype=float)
    n = len(t)
    selected = np.asarray(selected_cm, dtype=float).reshape(n, 2)
    palm = np.asarray(palm_cm, dtype=float).reshape(n, 2)
    labels = np.asarray(sources, dtype=object)
    drop = np.zeros(n, dtype=bool)
    piece_ids = np.full(n, -1, dtype=int)
    base_valid = np.isfinite(t) & np.isfinite(selected).all(axis=1)
    if int(base_valid.sum()) < 2:
        piece_ids[base_valid] = 0
        return drop, piece_ids

    positive_dt = np.diff(t[base_valid])
    positive_dt = positive_dt[positive_dt > 0]
    frame_dt = float(np.median(positive_dt)) if positive_dt.size else 1.0 / 30.0

    def pair_speed(a: int, b: int) -> float:
        elapsed = max(float(t[b] - t[a]), 1.0 / 240.0)
        displacement = 0.0
        if np.isfinite(palm[a]).all() and np.isfinite(palm[b]).all():
            displacement = max(displacement, float(np.linalg.norm(palm[b] - palm[a])))
        if labels[a] == labels[b]:
            displacement = max(displacement, float(np.linalg.norm(selected[b] - selected[a])))
        return displacement / elapsed

    def build_pieces():
        idx = np.flatnonzero(base_valid & ~drop)
        pieces = []
        if idx.size:
            start_pos = 0
            for k in range(1, idx.size):
                if pair_speed(int(idx[k - 1]), int(idx[k])) > speed_limit_cm_s:
                    pieces.append((start_pos, k))
                    start_pos = k
            pieces.append((start_pos, int(idx.size)))
        return idx, pieces

    for _ in range(n):  # each pass drops at least one piece, so this always terminates
        idx, pieces = build_pieces()
        if len(pieces) < 2:
            break

        def duration(piece):
            first, last = piece
            return float(t[idx[last - 1]] - t[idx[first]]) + frame_dt

        best = None
        for k, piece in enumerate(pieces):
            piece_duration = duration(piece)
            if piece_duration > max_excursion_seconds:
                continue
            if 0 < k < len(pieces) - 1:
                before_last = int(idx[pieces[k - 1][1] - 1])
                after_first = int(idx[pieces[k + 1][0]])
                removable = pair_speed(before_last, after_first) <= speed_limit_cm_s
            else:
                neighbour = pieces[1] if k == 0 else pieces[-2]
                removable = duration(neighbour) >= 2.0 * piece_duration
            if removable and (best is None or piece_duration < best[0]):
                best = (piece_duration, piece)
        if best is None:
            break
        first, last = best[1]
        drop[idx[first:last]] = True

    idx, pieces = build_pieces()
    for piece_id, (first, last) in enumerate(pieces):
        piece_ids[idx[first:last]] = piece_id
    return drop, piece_ids


def _records_to_dataframe(
    records,
    active_hand: Optional[str] = None,
    homography: Optional[np.ndarray] = None,
) -> pd.DataFrame:
    """Build hybrid index-tip/palm-fallback tracks and calibrated board-plane tracks.

    Offline stages, in this order: (1) false-detection jump removal, (2) interpolation of
    short dropouts, (3) zero-lag smoothing, (4) homography to board centimetres.
    """
    if not records:
        return pd.DataFrame()

    rows = []
    for record in records:
        row = {k: v for k, v in record.items() if k != "raw_frame"}
        for side in ("L", "R"):
            point = row.get(f"{side}_tracking_point")
            if point is None:
                point = row.get(f"{side}_hand_center")
            if point is None:  # Backward compatibility with old saved/intermediate records.
                point = row.get(f"{side}_index_tip")
            row[f"{side}_x"] = float(point[0]) if point is not None else np.nan
            row[f"{side}_y"] = float(point[1]) if point is not None else np.nan
        rows.append(row)

    df = pd.DataFrame(rows)
    n_frames = len(df)
    timestamps = pd.to_numeric(df["timestamp"], errors="coerce").to_numpy(dtype=float)
    quality = {}

    def to_board_cm(points_px: np.ndarray) -> np.ndarray:
        mapped_all = np.full((len(points_px), 2), np.nan, dtype=float)
        ok = np.isfinite(points_px).all(axis=1)
        if homography is not None and ok.any():
            mapped = cv2.perspectiveTransform(
                points_px[ok].astype(np.float32).reshape(-1, 1, 2), homography
            ).reshape(-1, 2)
            mapped_all[ok] = mapped
        return mapped_all

    def interpolate_short_time_gaps(arr, tracking_sources, piece_ids, detected):
        """Interpolate internal missing runs only when their time span is <= 200 ms."""
        valid = np.isfinite(arr) & np.isfinite(timestamps)
        result = arr.copy()
        valid_indices = np.flatnonzero(valid)
        for left, right in zip(valid_indices[:-1], valid_indices[1:]):
            if right - left <= 1:
                continue
            # A detected hand outside the activity polygon is a real outside-zone
            # observation, not a tracking dropout: never interpolate across it.
            if np.any(detected[left + 1:right]):
                continue
            # Do not bridge a transition between two different anatomical reference
            # points, nor a jump that separates two trajectory pieces.
            if tracking_sources[left] != tracking_sources[right] or piece_ids[left] != piece_ids[right]:
                continue
            gap_duration = float(timestamps[right] - timestamps[left])
            if not np.isfinite(gap_duration) or gap_duration <= 0:
                continue
            # Use the elapsed time across the bracket, independent of nominal FPS.
            if gap_duration <= INTERPOLATE_GAP_LIMIT_SECONDS:
                missing = np.arange(left + 1, right)
                result[missing] = np.interp(
                    timestamps[missing],
                    [timestamps[left], timestamps[right]],
                    [arr[left], arr[right]],
                )
        return result

    for side in ("L", "R"):
        if f"{side}_tracking_source" in df:
            tracking_sources = df[f"{side}_tracking_source"].fillna("unknown").astype(str).to_numpy(dtype=object)
        else:
            tracking_sources = np.full(n_frames, "legacy_single_point", dtype=object)

        x_observed = pd.to_numeric(df[f"{side}_x"], errors="coerce").to_numpy(dtype=float).copy()
        y_observed = pd.to_numeric(df[f"{side}_y"], errors="coerce").to_numpy(dtype=float).copy()
        valid_observed = np.isfinite(x_observed) & np.isfinite(y_observed) & np.isfinite(timestamps)
        rejected_online = (
            int(df[f"{side}_tracking_point_rejected"].fillna(False).astype(bool).sum())
            if f"{side}_tracking_point_rejected" in df else 0
        )

        # (1) Offline false-detection removal on the whole segment.
        dropped = np.zeros(n_frames, dtype=bool)
        piece_ids = np.where(valid_observed, 0, -1).astype(int)
        if homography is not None and valid_observed.any():
            selected_cm = to_board_cm(np.column_stack((x_observed, y_observed)))
            palm_px = np.full((n_frames, 2), np.nan, dtype=float)
            for i, record in enumerate(records):
                palm = record.get(f"{side}_palm_center_raw")
                if palm is not None and valid_observed[i]:
                    palm_px[i] = (float(palm[0]), float(palm[1]))
            dropped, piece_ids = _find_jump_outliers(
                timestamps, selected_cm, to_board_cm(palm_px), tracking_sources
            )
            if dropped.any():
                x_observed[dropped] = np.nan
                y_observed[dropped] = np.nan
                tracking_sources = tracking_sources.copy()
                tracking_sources[dropped] = "unknown"
                df.loc[dropped, f"{side}_tracking_point_rejected"] = True
                df.loc[dropped, f"{side}_rejection_reason"] = "offline_jump_excursion"
                df.loc[dropped, f"{side}_detected"] = False
                df.loc[dropped, f"{side}_tracking_source"] = None
                valid_observed = valid_observed & ~dropped

        # Private source/piece labels for filtering and kinematics. During a short
        # detection dropout the label is bridged only when both detected endpoints use
        # the same reference point and belong to the same trajectory piece; the exported
        # observed-source column stays missing for those frames so the report is honest.
        detected_observed = (
            df[f"{side}_detected"].fillna(False).to_numpy(dtype=bool)
            if f"{side}_detected" in df else np.zeros(n_frames, dtype=bool)
        )
        metric_sources = tracking_sources.copy()
        piece_for_metrics = piece_ids.copy()
        observed_indices = np.flatnonzero(valid_observed)
        for left, right in zip(observed_indices[:-1], observed_indices[1:]):
            if right - left <= 1:
                continue
            if np.any(detected_observed[left + 1:right]):
                continue
            source = tracking_sources[left]
            if source not in ("index_tip", "palm_center") or tracking_sources[right] != source:
                continue
            if piece_ids[left] != piece_ids[right]:
                continue
            gap_duration = float(timestamps[right] - timestamps[left])
            if 0 < gap_duration <= INTERPOLATE_GAP_LIMIT_SECONDS:
                metric_sources[left + 1:right] = source
                piece_for_metrics[left + 1:right] = piece_ids[left]
        metric_labels = np.asarray(
            [f"{s}|{int(p)}" for s, p in zip(metric_sources, piece_for_metrics)], dtype=object
        )
        df[f"{side}_tracking_source_for_metrics"] = metric_labels
        df[f"{side}_trajectory_piece"] = piece_ids

        # (2) interpolation of short dropouts, (3) zero-lag smoothing per continuous run.
        x_interp = interpolate_short_time_gaps(x_observed, tracking_sources, piece_ids, detected_observed)
        y_interp = interpolate_short_time_gaps(y_observed, tracking_sources, piece_ids, detected_observed)
        df[f"{side}_x"] = x_interp
        df[f"{side}_y"] = y_interp
        df[f"{side}_x_filtered"] = _filter_missing_series(timestamps, x_interp, metric_labels)
        df[f"{side}_y_filtered"] = _filter_missing_series(timestamps, y_interp, metric_labels)

        final_valid = np.isfinite(x_interp) & np.isfinite(y_interp)
        switch_count = 0
        valid_sources = tracking_sources[valid_observed]
        if valid_sources.size > 1:
            switch_count = int(np.sum(valid_sources[1:] != valid_sources[:-1]))
        quality[side] = {
            "snimku_celkem": int(n_frames),
            "snimku_s_platnym_bodem": int(final_valid.sum()),
            "podil_platnych_snimku_percent": round(float(final_valid.mean() * 100.0), 2) if n_frames else 0.0,
            "snimku_doplneno_interpolaci": int(final_valid.sum() - valid_observed.sum()),
            "zamitnuto_online": rejected_online,
            "zamitnuto_offline_skoky": int(dropped.sum()),
            "zamitnuto_celkem_percent": round(
                float((rejected_online + int(dropped.sum())) / n_frames * 100.0), 2
            ) if n_frames else 0.0,
            "preruseni_trajektorie_skokem": int(max(0, (piece_ids.max() + 1) - 1)) if (piece_ids >= 0).any() else 0,
            "prepnuti_zdroje_prst_dlan": switch_count,
        }

        # (4) calibrated board-plane coordinates (cm).
        if homography is not None:
            point_x = df[f"{side}_x_filtered"].to_numpy(dtype=float)
            point_y = df[f"{side}_y_filtered"].to_numpy(dtype=float)
            board_xy = to_board_cm(np.column_stack((point_x, point_y)))
            df[f"{side}_x_board_filtered"] = board_xy[:, 0]
            df[f"{side}_y_board_filtered"] = board_xy[:, 1]
            df[f"{side}_x_metric_filtered"] = board_xy[:, 0]
            df[f"{side}_y_metric_filtered"] = board_xy[:, 1]
        else:
            df[f"{side}_x_metric_filtered"] = df[f"{side}_x_filtered"]
            df[f"{side}_y_metric_filtered"] = df[f"{side}_y_filtered"]

        if (active_hand == "L" and side == "R") or (active_hand == "R" and side == "L"):
            quality[side] = None
            for col in (
                f"{side}_x_filtered", f"{side}_y_filtered",
                f"{side}_x_board_filtered", f"{side}_y_board_filtered",
                f"{side}_x_metric_filtered", f"{side}_y_metric_filtered",
            ):
                if col in df:
                    df[col] = np.nan
            df[f"{side}_elbow_angle"] = np.nan
            df[f"{side}_detected"] = False
            if f"{side}_tracking_source" in df:
                df[f"{side}_tracking_source"] = None
            if f"{side}_in_zone" in df:
                df[f"{side}_in_zone"] = False

    for side in ("L", "R"):
        angle_col = f"{side}_elbow_angle"
        if angle_col not in df:
            df[angle_col] = np.nan
        df[angle_col] = pd.to_numeric(df[angle_col], errors="coerce")
        detected_col = f"{side}_detected"
        if detected_col not in df:
            df[detected_col] = df.get(f"{side}_detected", False)
        df[f"{side}_detected"] = df[f"{side}_detected"].fillna(False).astype(bool)
        in_zone_col = f"{side}_in_zone"
        if in_zone_col not in df:
            df[in_zone_col] = False
        df[in_zone_col] = df[in_zone_col].fillna(False).astype(bool)

    # Keep the explicit detection columns under the names consumed by the metrics.
    if "L_detected" not in df:
        df["L_detected"] = False
    if "R_detected" not in df:
        df["R_detected"] = False
    df.attrs["coordinate_units"] = "cm" if homography is not None else "px"
    df.attrs["tracking_quality"] = quality
    return df



def _point_json_value(point):
    if point is None:
        return None
    values = np.asarray(point, dtype=float).reshape(-1)
    if values.size < 2 or not np.all(np.isfinite(values[:2])):
        return None
    return [round(float(values[0]), 2), round(float(values[1]), 2)]


def _json_safe(value):
    if value is None:
        return None
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    return value


def _draw_finite_path(frame, df, side, color):
    x_col, y_col = f"{side}_x_filtered", f"{side}_y_filtered"
    if x_col not in df or y_col not in df:
        return
    source_col = f"{side}_tracking_source_for_metrics"
    if source_col not in df:
        source_col = f"{side}_tracking_source"
    if source_col in df:
        sources = df[source_col].fillna("unknown").astype(str).to_numpy(dtype=object)
    else:
        sources = np.full(len(df), "legacy_single_point", dtype=object)

    points = []
    previous_source = None
    for x, y, source in zip(
        df[x_col].to_numpy(dtype=float),
        df[y_col].to_numpy(dtype=float),
        sources,
    ):
        valid = np.isfinite(x) and np.isfinite(y)
        if valid and previous_source is not None and source != previous_source:
            if len(points) >= 2:
                cv2.polylines(frame, [np.asarray(points, dtype=np.int32)], False, color, 3)
            points = []
        if valid:
            points.append([int(round(x)), int(round(y))])
            previous_source = source
        else:
            if len(points) >= 2:
                cv2.polylines(frame, [np.asarray(points, dtype=np.int32)], False, color, 3)
            points = []
            previous_source = None
    if len(points) >= 2:
        cv2.polylines(frame, [np.asarray(points, dtype=np.int32)], False, color, 3)




def _write_single_outputs(
    video_path, records, activity_id, polygon, hull, fps, width, height,
    homography=None, activity_hull=None, zone_margin_px=ACTIVITY_ZONE_MARGIN_PX,
    marker_dictionary=MARKER_DICTIONARY_NAME, marker_center_inset_cm=MARKER_CENTER_INSET_CM,
    video_format_warnings=None,
):
    video_dir = os.path.dirname(os.path.abspath(video_path))
    video_base_name = os.path.splitext(os.path.basename(video_path))[0]
    output_dir = os.path.join(video_dir, f"{video_base_name}_out")
    os.makedirs(output_dir, exist_ok=True)
    stale_diagnostics = os.path.join(output_dir, "tracking_diagnostics.csv")
    if os.path.exists(stale_diagnostics):
        os.remove(stale_diagnostics)

    cfg = get_activity_config(activity_id)
    df = _records_to_dataframe(records, homography=homography)
    center_x = (
        BOARD_SIDE_CM / 2.0
        if homography is not None
        else (float(polygon.centroid.x) if polygon is not None else None)
    )
    output_metrics = calculate_all_metrics(df=df, activity_id=activity_id, center_x=center_x)
    output_metrics["metadata"].update(
        _build_video_metadata(width, height, fps, homography, video_format_warnings)
    )
    output_metrics["metadata"].update({
        "ocekavane_video_rozliseni_px": [EXPECTED_VIDEO_WIDTH, EXPECTED_VIDEO_HEIGHT],
        "ocekavane_fps": EXPECTED_VIDEO_FPS,
        "aktivni_zona_rozsireni_px": float(zone_margin_px),
    })
    classification = _classify_batch_segment(records)
    if activity_id == 8:
        classification.update(activity_type="jednorucni", active_hand="R")
    elif activity_id == 9:
        classification.update(activity_type="jednorucni", active_hand="L")
    output_metrics["metadata"]["typ_aktivity"] = classification["activity_type"]

    video_trim_path = os.path.join(output_dir, "annotated.mp4")
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    video_writer = None

    source_cap = cv2.VideoCapture(video_path)
    if not source_cap.isOpened():
        raise RuntimeError(f"Zdrojové video nelze otevřít: {video_path}")

    trajectory_export = []
    first_source_frame = None
    last_valid_frame = None
    last_read_frame_id = None
    next_source_frame = 0

    try:
        for record_index, record in enumerate(records):
            target_frame = int(record["frame_id"])

            if last_read_frame_id == target_frame and last_valid_frame is not None:
                source_frame = last_valid_frame.copy()
            else:
                if target_frame < next_source_frame:
                    source_cap.release()
                    source_cap = cv2.VideoCapture(video_path)
                    if not source_cap.isOpened():
                        raise RuntimeError(f"Zdrojové video nelze znovu otevřít: {video_path}")
                    next_source_frame = 0

                eof_reached = False
                while next_source_frame < target_frame:
                    if not source_cap.grab():
                        eof_reached = True
                        break
                    next_source_frame += 1

                if eof_reached:
                    ok, source_frame = False, None
                else:
                    ok, source_frame = source_cap.read()
                    next_source_frame += 1

                if not ok or source_frame is None:
                    if last_valid_frame is not None:
                        source_frame = last_valid_frame.copy()
                    else:
                        source_frame = np.zeros((height, width, 3), dtype=np.uint8)
                else:
                    last_valid_frame = source_frame.copy()
                    last_read_frame_id = target_frame

            source_height, source_width = source_frame.shape[:2]
            if (source_width, source_height) != (width, height):
                raise RuntimeError(
                    "Rozměry při exportu neodpovídají rozměrům při analýze: "
                    f"analýza {width}x{height}, export {source_width}x{source_height}."
                )

            if video_writer is None:
                video_writer = cv2.VideoWriter(
                    video_trim_path, fourcc, fps, (source_width, source_height)
                )
                if not video_writer.isOpened():
                    video_writer.release()
                    video_writer = None
                    raise RuntimeError(
                        f"Nepodařilo se vytvořit výstupní video: {video_trim_path}"
                    )

            if first_source_frame is None:
                first_source_frame = source_frame.copy()

            annotated_frame = source_frame
            if activity_hull is not None:
                cv2.polylines(annotated_frame, [activity_hull], True, (0, 215, 255), 2)
            if hull is not None:
                cv2.polylines(annotated_frame, [hull], True, (0, 255, 0), 3)

            for side, color, enabled in (
                ("L", (255, 0, 0), cfg["track_left_hand"]),
                ("R", (0, 255, 255), cfg["track_right_hand"]),
            ):
                if not enabled:
                    continue
                x = df.iloc[record_index][f"{side}_x_filtered"]
                y = df.iloc[record_index][f"{side}_y_filtered"]
                if np.isfinite(x) and np.isfinite(y):
                    cv2.circle(annotated_frame, (int(round(x)), int(round(y))), 9, color, -1)
                if record.get(f"{side}_tracking_point_rejected"):
                    raw = record.get(f"{side}_hand_center_raw")
                    if raw is not None:
                        p = (int(round(raw[0])), int(round(raw[1])))
                        cv2.drawMarker(
                            annotated_frame, p, (0, 0, 255), cv2.MARKER_TILTED_CROSS,
                            18, 2, cv2.LINE_AA,
                        )

            video_writer.write(annotated_frame)

            def point_json(side, enabled):
                x = df.iloc[record_index][f"{side}_x_filtered"]
                y = df.iloc[record_index][f"{side}_y_filtered"]
                if not enabled or not (np.isfinite(x) and np.isfinite(y)):
                    return None
                return [round(float(x), 2), round(float(y), 2)]

            def board_point_json(side, enabled):
                x = df.iloc[record_index].get(f"{side}_x_board_filtered", np.nan)
                y = df.iloc[record_index].get(f"{side}_y_board_filtered", np.nan)
                if homography is None or not enabled or not (np.isfinite(x) and np.isfinite(y)):
                    return None
                return [round(float(x), 3), round(float(y), 3)]

            def raw_point_json(side, field, enabled):
                point = record.get(f"{side}_{field}")
                if not enabled or point is None:
                    return None
                return [round(float(point[0]), 2), round(float(point[1]), 2)]

            trajectory_export.append({
                "frame_id": int(record["frame_id"]),
                "timestamp": float(record["timestamp"]),
                "L_hand_center": point_json("L", cfg["track_left_hand"]),
                "R_hand_center": point_json("R", cfg["track_right_hand"]),
                "L_hand_center_raw": raw_point_json("L", "hand_center_raw", cfg["track_left_hand"]),
                "R_hand_center_raw": raw_point_json("R", "hand_center_raw", cfg["track_right_hand"]),
                "L_tracking_point": point_json("L", cfg["track_left_hand"]),
                "R_tracking_point": point_json("R", cfg["track_right_hand"]),
                "L_tracking_source": record.get("L_tracking_source") if cfg["track_left_hand"] else None,
                "R_tracking_source": record.get("R_tracking_source") if cfg["track_right_hand"] else None,
                "L_raw_tracking_source": record.get("L_raw_tracking_source") if cfg["track_left_hand"] else None,
                "R_raw_tracking_source": record.get("R_raw_tracking_source") if cfg["track_right_hand"] else None,
                "L_index_tip_raw": _point_json_value(record.get("L_index_tip_raw")) if cfg["track_left_hand"] else None,
                "R_index_tip_raw": _point_json_value(record.get("R_index_tip_raw")) if cfg["track_right_hand"] else None,
                "L_palm_center_raw": _point_json_value(record.get("L_palm_center_raw")) if cfg["track_left_hand"] else None,
                "R_palm_center_raw": _point_json_value(record.get("R_palm_center_raw")) if cfg["track_right_hand"] else None,
                "L_hand_center_board_cm": board_point_json("L", cfg["track_left_hand"]),
                "R_hand_center_board_cm": board_point_json("R", cfg["track_right_hand"]),
                "L_tracking_point_board_cm": board_point_json("L", cfg["track_left_hand"]),
                "R_tracking_point_board_cm": board_point_json("R", cfg["track_right_hand"]),
                "L_index_extended": record.get("L_index_extended") if cfg["track_left_hand"] else None,
                "R_index_extended": record.get("R_index_extended") if cfg["track_right_hand"] else None,
                "L_index_pip_angle_deg": _json_safe(record.get("L_index_pip_angle_deg")) if cfg["track_left_hand"] else None,
                "R_index_pip_angle_deg": _json_safe(record.get("R_index_pip_angle_deg")) if cfg["track_right_hand"] else None,
                "L_index_dip_angle_deg": _json_safe(record.get("L_index_dip_angle_deg")) if cfg["track_left_hand"] else None,
                "R_index_dip_angle_deg": _json_safe(record.get("R_index_dip_angle_deg")) if cfg["track_right_hand"] else None,
                "L_detected": bool(record.get("L_detected", False)) if cfg["track_left_hand"] else False,
                "R_detected": bool(record.get("R_detected", False)) if cfg["track_right_hand"] else False,
                "L_raw_detection": bool(record.get("L_raw_detection", False)) if cfg["track_left_hand"] else False,
                "R_raw_detection": bool(record.get("R_raw_detection", False)) if cfg["track_right_hand"] else False,
                "L_tracking_point_rejected": bool(record.get("L_tracking_point_rejected", False)) if cfg["track_left_hand"] else False,
                "R_tracking_point_rejected": bool(record.get("R_tracking_point_rejected", False)) if cfg["track_right_hand"] else False,
                "L_rejection_reason": record.get("L_rejection_reason") if cfg["track_left_hand"] else None,
                "R_rejection_reason": record.get("R_rejection_reason") if cfg["track_right_hand"] else None,
                "L_identity_corrected": bool(record.get("L_identity_corrected", False)) if cfg["track_left_hand"] else False,
                "R_identity_corrected": bool(record.get("R_identity_corrected", False)) if cfg["track_right_hand"] else False,
                "L_identity_ambiguous": bool(record.get("L_identity_ambiguous", False)) if cfg["track_left_hand"] else False,
                "R_identity_ambiguous": bool(record.get("R_identity_ambiguous", False)) if cfg["track_right_hand"] else False,
                "L_handedness_label_raw": record.get("L_handedness_label_raw") if cfg["track_left_hand"] else None,
                "R_handedness_label_raw": record.get("R_handedness_label_raw") if cfg["track_right_hand"] else None,
                "L_association_distance_px": _json_safe(record.get("L_association_distance_px")) if cfg["track_left_hand"] else None,
                "R_association_distance_px": _json_safe(record.get("R_association_distance_px")) if cfg["track_right_hand"] else None,
                "L_observed_speed_cm_s": _json_safe(record.get("L_observed_speed_cm_s")) if cfg["track_left_hand"] else None,
                "R_observed_speed_cm_s": _json_safe(record.get("R_observed_speed_cm_s")) if cfg["track_right_hand"] else None,
                "L_in_zone": bool(record.get("L_in_zone", False)) if cfg["track_left_hand"] else False,
                "R_in_zone": bool(record.get("R_in_zone", False)) if cfg["track_right_hand"] else False,
                "L_detection_confidence": _json_safe(record.get("L_detection_confidence")) if cfg["track_left_hand"] else None,
                "R_detection_confidence": _json_safe(record.get("R_detection_confidence")) if cfg["track_right_hand"] else None,
                "L_elbow_angle": _json_safe(record.get("L_elbow_angle")) if cfg["track_left_hand"] else None,
                "R_elbow_angle": _json_safe(record.get("R_elbow_angle")) if cfg["track_right_hand"] else None,
            })
    finally:
        if video_writer is not None:
            video_writer.release()
        if source_cap is not None:
            source_cap.release()

    with open(os.path.join(output_dir, "trajectory.json"), "w", encoding="utf-8") as f:
        json.dump({
            "metadata": {
                "fps": fps,
                "activity_type": classification["activity_type"],
                "active_hand": classification["active_hand"],
                "expected_video_size_px": [EXPECTED_VIDEO_WIDTH, EXPECTED_VIDEO_HEIGHT],
                "decoded_video_size_px": [int(width), int(height)],
                "video_orientation": "portrait" if height > width else "landscape",
                "video_display_aspect_ratio": "9:16" if height > width else "16:9",
                "expected_fps": EXPECTED_VIDEO_FPS,
                "board_side_cm": BOARD_SIDE_CM,
                "coordinates_calibrated_to_board": homography is not None,
                "board_coordinate_units": "cm" if homography is not None else None,
                "tracked_point": "index_tip_when_extended_else_palm_center_fallback",
                "index_tip_landmark": 8,
                "index_extension_test": {
                    "pip_enter_threshold_deg": INDEX_EXTENSION_ENTER_PIP_DEG,
                    "dip_enter_threshold_deg": INDEX_EXTENSION_ENTER_DIP_DEG,
                    "pip_exit_threshold_deg": INDEX_EXTENSION_EXIT_PIP_DEG,
                    "dip_exit_threshold_deg": INDEX_EXTENSION_EXIT_DIP_DEG,
                },
                "identity_assignment": "temporal_palm_center_association_with_handedness_prior",
                "identity_ambiguity_policy": "reject_ambiguous_frames_instead_of_switching_hand_tracks",
                "max_identity_speed_cm_s": MAX_IDENTITY_SPEED_CM_S,
                "max_measured_point_speed_cm_s": MAX_MEASURED_POINT_SPEED_CM_S,
                "jerk_estimator": (
                    f"local_polynomial_degree_{JERK_POLY_DEGREE}_window_{JERK_WINDOW_SECONDS:g}s_run_edges_excluded"
                ),
                "position_smoothing": f"zero_lag_local_polynomial_degree_{SMOOTHING_POLY_DEGREE}_window_{SMOOTHING_WINDOW_SECONDS:g}s",
                "normalized_jerk_mode": NORM_JERK_MODE,
                "elbow_angle_mode": ELBOW_ANGLE_MODE,
                "source_switch_debounce_s": SOURCE_SWITCH_MIN_SECONDS,
                "jump_speed_limit_cm_s": MAX_MEASURED_POINT_SPEED_CM_S,
                "video_format_warnings": list(video_format_warnings or []),
                "activity_zone_margin_px": float(zone_margin_px),
                "marker_dictionary": marker_dictionary,
                "marker_center_inset_cm": float(marker_center_inset_cm),
                "input_video_is_mirrored": bool(INPUT_VIDEO_IS_MIRRORED),
            },
            "frames": trajectory_export,
        }, f, indent=4, ensure_ascii=False, allow_nan=False)

    if first_source_frame is None:
        raise RuntimeError("Nebyl načten žádný zdrojový snímek pro vizualizaci trajektorie.")

    bg_image = first_source_frame.copy()
    if activity_hull is not None:
        cv2.polylines(bg_image, [activity_hull], True, (0, 215, 255), 2)
    if hull is not None:
        cv2.polylines(bg_image, [hull], True, (0, 255, 0), 3)
    _draw_finite_path(bg_image, df, "L", (255, 0, 0))
    _draw_finite_path(bg_image, df, "R", (0, 255, 255))

    plot_path = os.path.join(output_dir, "trajectory_plot.png")
    ok_encode, encoded_img = cv2.imencode(".png", bg_image)
    if not ok_encode:
        raise RuntimeError("Nepodařilo se zakódovat trajectory_plot.png")
    with open(plot_path, "wb") as f:
        f.write(encoded_img.tobytes())

    with open(os.path.join(output_dir, "data.json"), "w", encoding="utf-8") as f:
        json.dump(output_metrics, f, indent=4, ensure_ascii=False, allow_nan=False)

    return output_dir
def run_pipeline(activity_id: int, video_path: str, frame_callback=None, cancel_event=None):
    """Analyze one video and export native-resolution frames by rereading the source."""
    if activity_id not in range(1, 11):
        raise ValueError("ID aktivity musí být v rozsahu 1 až 10.")

    collected = _collect_segments(
        video_path=video_path,
        mode="single",
        activity_id=activity_id,
        frame_callback=frame_callback,
        cancel_event=cancel_event,
    )
    if collected["cancelled"]:
        return False, "Analýza byla přerušena uživatelem.", None
    if not collected["segments"]:
        return False, collected["stop_reason"] or "Nebyla detekována žádná aktivita.", None

    output_dir = _write_single_outputs(
        video_path,
        collected["segments"][0],
        activity_id,
        collected["polygon"],
        collected["hull"],
        collected["fps"],
        collected["width"],
        collected["height"],
        homography=collected["homography"],
        activity_hull=collected.get("activity_hull"),
        zone_margin_px=collected.get("zone_margin_px", ACTIVITY_ZONE_MARGIN_PX),
        marker_dictionary=collected.get("marker_dictionary", MARKER_DICTIONARY_NAME),
        marker_center_inset_cm=collected.get("marker_center_inset_cm", MARKER_CENTER_INSET_CM),
        video_format_warnings=collected.get("video_format_warnings", []),
    )
    return True, f"Zpracování úspěšně dokončeno!\n\nVýstupy uloženy do:\n{output_dir}", output_dir


# =========================================================================
# 4. BATCH PROCESSING & CSV EXPORT
# =========================================================================
def _classify_batch_segment(records):
    """Classify only the number of participating hands, not the game/task identity."""
    frame_count = max(1, len(records))
    left_share = sum(bool(row.get("L_in_zone")) for row in records) / frame_count
    right_share = sum(bool(row.get("R_in_zone")) for row in records) / frame_count
    both_share = sum(
        bool(row.get("L_in_zone")) and bool(row.get("R_in_zone"))
        for row in records
    ) / frame_count

    if left_share >= 0.5 and right_share >= 0.5 and both_share >= 0.25:
        activity_type, active_hand, compare_hands = "obourucni", None, True
    elif left_share >= 0.5 and right_share < 0.5:
        activity_type, active_hand, compare_hands = "jednorucni", "L", False
    elif right_share >= 0.5 and left_share < 0.5:
        activity_type, active_hand, compare_hands = "jednorucni", "R", False
    else:
        activity_type, active_hand, compare_hands = "neurceno", None, False

    return {
        "activity_type": activity_type,
        "active_hand": active_hand,
        "compare_hands": compare_hands,
        "left_share": left_share,
        "right_share": right_share,
        "both_share": both_share,
    }


def _batch_report_value(metrics: dict, path_template: str):
    """Resolve one schema path in a segment's metrics and convert it for the CSV."""
    coordinate_units = str(_nested_value(metrics, "metadata.jednotky_souradnic") or "px")
    coord_unit = "cm" if coordinate_units == "cm" else "px"
    jerk_unit = "cm_s3" if coord_unit == "cm" else "px_s3"
    path = path_template.format(coord_unit=coord_unit, jerk_unit=jerk_unit)
    value = _nested_value(metrics, path)
    if isinstance(value, list):
        return " | ".join(map(str, value)) if value else "Bez upozornění"
    if value is None:
        return ""
    if isinstance(value, (np.bool_, bool)):
        return "Ano" if bool(value) else "Ne"
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, (float, np.floating)):
        return float(value) if np.isfinite(value) else ""
    if path.endswith("typ_aktivity"):
        return {"obourucni": "Obouruční", "jednorucni": "Jednoruční", "neurceno": "Neurčeno"}.get(str(value), value)
    if path.startswith("kinematika.plynulost_norm_jerk_zdroj"):
        return {"tahy": "Medián přes tahy", "cely_usek": "Celý úsek (orientační)"}.get(str(value), value)
    return value


def _report_unit(unit_type: str, metadata: dict) -> str:
    """Unit label for a schema row; coordinate-dependent units follow the calibration."""
    coordinate_units = str(metadata.get("jednotky_souradnic") or "px")
    speed_units = str(metadata.get("jednotky_rychlosti") or f"{coordinate_units}/s")
    jerk_units = str(metadata.get("jednotky_jerku") or ("cm/s³" if coordinate_units == "cm" else "px/s³"))
    units = {
        "coord": coordinate_units, "speed": speed_units, "jerk": jerk_units,
        "degree": "°", "percent": "%", "count": "ks", "bool": "Ano/Ne",
        "correlation": "r", "none": "–", "text": "–", "seconds": "s",
        "pixels": "px", "fps": "fps", "ratio": "–", "resolution": "px",
    }
    return units.get(unit_type, "–")


def _video_schema_value(video_metadata: dict, status: str, path: str):
    """Value of a per-video (not per-segment) schema row."""
    if path == "video.resolution":
        width, height = video_metadata.get("video_width_px"), video_metadata.get("video_height_px")
        return f"{int(width)} × {int(height)}" if width and height else ""
    if path == "video.status":
        return status
    if path == "video.warnings":
        warnings = video_metadata.get("upozorneni_formatu_videa", []) or []
        return " | ".join(map(str, warnings)) if warnings else "Bez upozornění"
    value = video_metadata.get(path.split(".", 1)[1])
    if value is None:
        return ""
    if path == "video.video_orientation":
        return {"portrait": "Na výšku", "landscape": "Na šířku", "square": "Čtvercové"}.get(str(value), value)
    return value


def _csv_cell(value):
    """Format one CSV cell (decimal separator is configurable, see CSV_DECIMAL_SEPARATOR)."""
    if isinstance(value, (float, np.floating)):
        text = repr(round(float(value), 4))
        return text.replace(".", CSV_DECIMAL_SEPARATOR) if CSV_DECIMAL_SEPARATOR != "." else text
    return value


def _open_report_file(path: str):
    """Open the CSV for writing; if it is locked (e.g. open in Excel) use a timestamped name."""
    try:
        return open(path, "w", newline="", encoding="utf-8-sig"), path
    except PermissionError:
        base, extension = os.path.splitext(path)
        alternative = f"{base}_{datetime.now():%Y%m%d_%H%M%S}{extension}"
        return open(alternative, "w", newline="", encoding="utf-8-sig"), alternative


def _build_report_table(video_results: dict) -> list:
    """Build the report as rows of raw values (numbers stay numbers).

    Layout: Kategorie; Oblast; Metrika; Jednotka; <video 1>; <video 2>; ... Segment metrics
    are written once per activity-window slot, then the per-video format rows follow.
    """
    video_names = list(video_results.keys())
    max_segments = max((len(result.get("segments", [])) for result in video_results.values()), default=0)
    segment_slots = max(1, max_segments) if REPORT_MAX_SEGMENTS is None else REPORT_MAX_SEGMENTS

    rows = [["Kategorie", "Oblast", "Metrika", "Jednotka"] + video_names]
    for segment_index in range(segment_slots):
        # Units come from the first video that has this segment slot.
        unit_source_metadata = {}
        for candidate_name in video_names:
            candidate_segments = video_results[candidate_name].get("segments", [])
            if segment_index < len(candidate_segments):
                unit_source_metadata = candidate_segments[segment_index].get("metrics", {}).get("metadata", {})
                break
        for category in REPORT_CATEGORIES:
            for schema_category, area, label, unit_type, path_template in SEGMENT_REPORT_SCHEMA:
                if schema_category != category:
                    continue
                values = []
                for video_name in video_names:
                    segments = video_results[video_name].get("segments", [])
                    if segment_index >= len(segments):
                        values.append("")
                        continue
                    values.append(_batch_report_value(segments[segment_index].get("metrics", {}), path_template))
                rows.append([category, area, label, _report_unit(unit_type, unit_source_metadata)] + values)
    for area, label, unit_type, path in VIDEO_REPORT_SCHEMA:
        values = []
        for video_name in video_names:
            result = video_results[video_name]
            video_metadata = result.get("video_metadata", {}) or {}
            values.append(_video_schema_value(video_metadata, result.get("status", "N/A"), path))
        unit = "px" if unit_type == "resolution" else "fps" if unit_type == "fps" else "–"
        rows.append(["obecné parametry videa", area, label, unit] + values)
    return rows


def _write_batch_csv(folder_path: str, video_results: dict, csv_name: str = "batch_results.csv") -> str:
    """Write the batch report as CSV (semicolon separated, UTF-8 with BOM)."""
    rows = _build_report_table(video_results)
    handle, output_path = _open_report_file(os.path.join(folder_path, csv_name))
    with handle as f:
        writer = csv.writer(f, delimiter=";")
        for row in rows:
            writer.writerow([_csv_cell(value) for value in row])
    return output_path


def _write_batch_xlsx(folder_path: str, video_results: dict, xlsx_name: str = "batch_results.xlsx") -> Optional[str]:
    """Write the same report as .xlsx with real numeric cells; None if openpyxl is missing.

    Unlike the CSV this file does not depend on the decimal separator of the computer
    that opens it, so it is the recommended file for reading and sorting by hand.
    """
    try:
        from openpyxl import Workbook
        from openpyxl.styles import Font
        from openpyxl.utils import get_column_letter
    except ImportError:
        return None
    rows = _build_report_table(video_results)
    workbook = Workbook()
    sheet = workbook.active
    sheet.title = "Výsledky"
    for row in rows:
        sheet.append([None if value == "" else value for value in row])
    for cell in sheet[1]:
        cell.font = Font(bold=True)
    sheet.freeze_panes = "E2"
    widths = {1: 24, 2: 24, 3: 38, 4: 10}
    for column in range(1, len(rows[0]) + 1):
        sheet.column_dimensions[get_column_letter(column)].width = widths.get(column, 22)
    path = os.path.join(folder_path, xlsx_name)
    try:
        workbook.save(path)
    except PermissionError:  # file open in Excel: keep the results under a new name
        base, extension = os.path.splitext(path)
        path = f"{base}_{datetime.now():%Y%m%d_%H%M%S}{extension}"
        workbook.save(path)
    return path


def _log_batch_error(folder_path: str, video_name: str) -> None:
    """Append the full traceback of a failed video to batch_errors.log (never raises)."""
    try:
        with open(os.path.join(folder_path, "batch_errors.log"), "a", encoding="utf-8") as log:
            log.write(f"[{datetime.now():%Y-%m-%d %H:%M:%S}] {video_name}\n{traceback.format_exc()}\n")
    except OSError:
        pass


def _probe_video_metadata(video_path: str) -> dict:
    """Format information readable without processing (used when a video fails early)."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        cap.release()
        return {"pipeline_version": PIPELINE_VERSION}
    try:
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        if not np.isfinite(fps) or fps <= 0:
            fps = None
        result = {
            "video_width_px": width or None,
            "video_height_px": height or None,
            "video_fps": round(fps, 3) if fps is not None else None,
        }
        result.update(get_video_format(width, height))
        result.update({
            "jednotky_souradnic": "px",
            "jednotky_rychlosti": "px/s",
            "jednotky_jerku": "px/s³",
            "kalibrace_desky": "neznámá",
            "upozorneni_formatu_videa": [],
            "pipeline_version": PIPELINE_VERSION,
        })
        return result
    finally:
        cap.release()


def run_batch_pipeline(
    folder_path: str,
    frame_callback=None,
    progress_callback=None,
    cancel_event=None,
):
    """Analyze one activity per video using buffered hybrid-point segmentation."""
    if not os.path.isdir(folder_path):
        raise NotADirectoryError(f"Vstupní složka neexistuje: {folder_path}")

    video_paths = sorted(
        os.path.join(folder_path, name)
        for name in os.listdir(folder_path)
        if os.path.isfile(os.path.join(folder_path, name))
        and os.path.splitext(name)[1].lower() in VIDEO_EXTENSIONS
    )
    if not video_paths:
        return False, "Ve vybrané složce nebyla nalezena žádná podporovaná videa.", None

    results = {}
    cancelled = False
    for video_index, video_path in enumerate(video_paths, start=1):
        video_name = os.path.basename(video_path)
        if cancel_event is not None and cancel_event.is_set():
            cancelled = True
            break

        if progress_callback is not None:
            progress_callback(
                f"Batch {video_index}/{len(video_paths)}: {video_name}",
                (video_index - 1) / len(video_paths),
            )

        video_metadata = _probe_video_metadata(video_path)
        try:
            collected = _collect_segments(
                video_path=video_path,
                mode="batch",
                frame_callback=frame_callback,
                cancel_event=cancel_event,
                video_number=video_index,
                video_count=len(video_paths),
            )
            video_metadata = _build_video_metadata(
                collected["width"], collected["height"], collected["fps"],
                collected["homography"], collected.get("video_format_warnings", []),
            )
            if collected["cancelled"]:
                results[video_name] = {"status": "Přerušeno uživatelem", "segments": [], "video_metadata": video_metadata}
                cancelled = True
                break

            segment_results = []
            center_x = (
                BOARD_SIDE_CM / 2.0 if collected["homography"] is not None
                else (float(collected["polygon"].centroid.x) if collected["polygon"] is not None else None)
            )
            for records in collected["segments"]:
                classification = _classify_batch_segment(records)
                df = _records_to_dataframe(
                    records,
                    active_hand=classification["active_hand"],
                    homography=collected["homography"],
                )
                metrics = calculate_all_metrics(
                    df,
                    activity_id=None,
                    center_x=center_x,
                    compare_hands=classification["compare_hands"],
                )
                metrics["metadata"].pop("activity_id", None)
                metrics["metadata"].pop("aktivni_ruka", None)
                metrics["metadata"]["typ_aktivity"] = classification["activity_type"]
                metrics["metadata"].update({
                    "jednotky_souradnic": video_metadata["jednotky_souradnic"],
                    "jednotky_rychlosti": video_metadata["jednotky_rychlosti"],
                    "jednotky_jerku": video_metadata["jednotky_jerku"],
                    "video_width_px": video_metadata["video_width_px"],
                    "video_height_px": video_metadata["video_height_px"],
                    "video_orientation": video_metadata["video_orientation"],
                    "video_display_aspect_ratio": video_metadata["video_display_aspect_ratio"],
                    "video_fps": video_metadata["video_fps"],
                })
                segment_results.append({"metrics": metrics})

            if collected["stop_reason"] and not segment_results:
                status = collected["stop_reason"]
            elif segment_results:
                status = "OK"
            else:
                status = "Nebyla nalezena žádná aktivita."
            results[video_name] = {"status": status, "segments": segment_results, "video_metadata": video_metadata}

        except Exception as exc:
            # An individual corrupt/unsupported video must not abort the whole batch.
            _log_batch_error(folder_path, video_name)
            results[video_name] = {
                "status": f"CHYBA: {type(exc).__name__}: {exc}",
                "segments": [],
                "video_metadata": video_metadata,
            }

        if progress_callback is not None:
            progress_callback(
                f"Dokončeno {video_index}/{len(video_paths)}: {video_name}",
                video_index / len(video_paths),
            )

    csv_path = _write_batch_csv(folder_path, results)
    xlsx_path = _write_batch_xlsx(folder_path, results)
    xlsx_note = f"\nXLSX (číselné buňky, doporučeno): {xlsx_path}" if xlsx_path else ""
    completed = len(results)
    activity_window_count = sum(len(result.get("segments", [])) for result in results.values())
    error_count = sum(str(result.get("status", "")).startswith("CHYBA:") for result in results.values())
    no_activity_count = sum(
        not result.get("segments")
        and not str(result.get("status", "")).startswith("CHYBA:")
        and result.get("status") != "Přerušeno uživatelem"
        for result in results.values()
    )

    if cancelled:
        message = (
            "Batch processing byl přerušen. Částečné výsledky z již zpracovaných videí byly uloženy do:\n"
            f"{csv_path}{xlsx_note}\n\nZpracováno videí: {completed}/{len(video_paths)}; nalezených úseků aktivity: {activity_window_count}."
        )
        return False, message, csv_path

    message = (
        "Batch processing dokončen.\n\n"
        f"Zpracováno videí: {completed}\n"
        f"Nalezených úseků aktivity: {activity_window_count}\n"
        f"Videa s chybou: {error_count}\n"
        f"Videa bez detekované aktivity: {no_activity_count}\n\n"
        f"CSV výsledky: {csv_path}"
        f"{xlsx_note}"
    )
    return True, message, csv_path
