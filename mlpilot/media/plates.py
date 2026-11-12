"""Licence plates: find them, read them (OCR), check how reading holds up in fog / rain / night,
and blur them for privacy.

Needs ``pip install "dataspoof-mlpilot[ocr]"`` (OpenCV + RapidOCR; the OCR models ship inside the
package, so nothing is downloaded at run time). Without OCR installed, ``detect_plates(read=False)``
still finds plate-shaped regions with OpenCV alone.

>>> from mlpilot.media import plates
>>> plates.detect_plates(img)                          # [Plate(box=(x, y, w, h), text='MH12AB1234', confidence=0.99)]
>>> plates.read_plate(img, box, enhance=True)          # dehaze / brighten / denoise first, then read
>>> plates.plate_report("traffic_cam/", pattern="IN")  # one row per image, with weather / light conditions
>>> plates.clean_plate_text("MH l2 AB I234", pattern="IN")
'MH12AB1234'

How plates are found: (1) OCR text boxes whose text looks like a plate (4-12 letters and digits,
with both, or matching ``pattern``); (2) OpenCV's plate detector; (3) bright, plate-shaped
rectangles. (2) and (3) are a fallback, tried (best ``max_candidates`` first) only when the
full-image pass reads no plate, and kept only when they read as plate text. With ``read=False``
every OCR plate and every detector hit is returned, which is what blurring wants.
"""

from __future__ import annotations

import dataclasses
import itertools
import logging
import re

import numpy as np
import polars as pl

from .faces import _cascade, _cv2, _rgb_u8
from .image import _items

PLATE_PATTERNS = {
    "IN": r"^[A-Z]{2}\d{1,2}[A-Z]{0,3}\d{1,4}$",         # MH12AB1234, DL3CAB1234, 22BH1234AA is below
    "IN_BH": r"^\d{2}BH\d{4}[A-Z]{1,2}$",
    "US": r"^[A-Z0-9]{2,8}$",
    "EU": r"^[A-Z]{1,3}[A-Z0-9]{1,6}$",
    "UK": r"^[A-Z]{2}\d{2}[A-Z]{3}$",
}
_TO_DIGIT = {"O": "0", "Q": "0", "D": "0", "I": "1", "L": "1", "Z": "2", "S": "5", "B": "8", "G": "6", "T": "7"}
_TO_LETTER = {"0": "O", "1": "I", "2": "Z", "5": "S", "8": "B", "6": "G", "7": "T", "4": "A"}


@dataclasses.dataclass
class Plate:
    box: tuple                    # (x, y, w, h)
    text: str = ""
    confidence: float = 0.0
    source: str = ""              # "ocr", "cascade" or "shape"

    def __repr__(self) -> str:
        return f"Plate(box={self.box}, text={self.text!r}, confidence={self.confidence:.2f}, source={self.source!r})"


# ---- OCR engine --------------------------------------------------------------------------------
_ENGINE: list = []


def _ocr():
    """The RapidOCR engine (created once; its start-up logging is silenced)."""
    if not _ENGINE:
        try:
            from rapidocr import RapidOCR
        except ImportError as exc:
            raise ImportError('Plate reading needs OCR: pip install "dataspoof-mlpilot[ocr]"') from exc
        previous = logging.root.manager.disable
        logging.disable(logging.WARNING)
        try:
            _ENGINE.append(RapidOCR())
        finally:
            logging.disable(previous)
        logging.getLogger("RapidOCR").setLevel(logging.ERROR)
    return _ENGINE[0]


def ocr_available() -> bool:
    try:
        import rapidocr  # noqa: F401
    except ImportError:
        return False
    return True


def ocr_text(image) -> list:
    """Every text line in an image: ``[(box, text, confidence)]`` with ``box = (x, y, w, h)``."""
    rgb = _rgb_u8(image)
    result = _ocr()(rgb)
    if result.boxes is None or result.txts is None:
        return []
    out = []
    for quad, text, score in zip(result.boxes, result.txts, result.scores):
        q = np.asarray(quad)
        x0, y0 = q.min(axis=0)
        x1, y1 = q.max(axis=0)
        out.append(((int(x0), int(y0), int(x1 - x0), int(y1 - y0)), str(text), float(score)))
    return out


# ---- text ---------------------------------------------------------------------------------------
def _pattern(pattern):
    if pattern is None:
        return None
    if isinstance(pattern, re.Pattern):
        return pattern
    return re.compile(PLATE_PATTERNS.get(pattern, pattern))


def clean_plate_text(text: str, *, pattern=None) -> str:
    """Upper-case letters and digits only. With ``pattern`` (a regex or one of ``PLATE_PATTERNS``),
    fixes OCR look-alikes (O/0, I/1, S/5, B/8 ...) until the text matches, or returns it unchanged."""
    s = re.sub(r"[^A-Z0-9]", "", text.upper())
    rx = _pattern(pattern)
    if rx is None or rx.match(s) or not s:
        return s
    options = [sorted({c, _TO_DIGIT.get(c, c), _TO_LETTER.get(c, c)}) for c in s]
    if np.prod([len(o) for o in options], dtype=float) > 4096:     # too many to try
        return s
    for combo in itertools.product(*options):
        cand = "".join(combo)
        if rx.match(cand):
            return cand
    return s


def looks_like_plate(text: str, *, pattern=None) -> bool:
    s = re.sub(r"[^A-Z0-9]", "", text.upper())
    rx = _pattern(pattern)
    if rx is not None:
        return bool(rx.match(clean_plate_text(s, pattern=rx)))
    return 4 <= len(s) <= 12 and any(c.isdigit() for c in s) and any(c.isalpha() for c in s)


# ---- detection ----------------------------------------------------------------------------------
def _iou(a: tuple, b: tuple) -> float:
    ax1, ay1, bx1, by1 = a[0] + a[2], a[1] + a[3], b[0] + b[2], b[1] + b[3]
    iw = max(0, min(ax1, bx1) - max(a[0], b[0]))
    ih = max(0, min(ay1, by1) - max(a[1], b[1]))
    inter = iw * ih
    return inter / float(a[2] * a[3] + b[2] * b[3] - inter + 1e-9)


def _shape_candidates(rgb: np.ndarray, *, min_width: int) -> list:
    """Bright, plate-shaped rectangles (aspect 2-6) with dark marks inside."""
    cv2 = _cv2()
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    gray = cv2.bilateralFilter(gray, 7, 40, 40)
    blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, cv2.getStructuringElement(cv2.MORPH_RECT, (15, 5)))
    _, bw = cv2.threshold(blackhat, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    bw = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (17, 3)))
    contours, _ = cv2.findContours(bw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    scored = []
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        if w < min_width or h < 8 or not 2.0 <= w / h <= 6.5:
            continue
        roi = gray[y:y + h, x:x + w]
        if roi.mean() < 70 or roi.std() < 25:              # plates are light with dark characters
            continue
        scored.append((float(roi.std()) * np.sqrt(w * h), (int(x), int(y), int(w), int(h))))
    scored.sort(key=lambda t: -t[0])                        # most plate-like (big, high contrast) first
    return [b for _, b in scored]


def _pad(box: tuple, shape: tuple, frac: float) -> tuple:
    x, y, w, h = box
    px, py = int(w * frac), int(h * frac * 2)
    x0, y0 = max(0, x - px), max(0, y - py)
    x1, y1 = min(shape[1], x + w + px), min(shape[0], y + h + py)
    return (x0, y0, x1 - x0, y1 - y0)


def read_plate(image, box: tuple | None = None, *, pattern=None, enhance: bool = False) -> tuple:
    """OCR one plate (the whole image, or ``box``). Returns ``(text, confidence)``.

    ``enhance=True`` first removes haze / low light / noise / blur that the condition detector finds
    (see :func:`mlpilot.media.conditions.enhance`) and keeps whichever read is more confident."""
    rgb = _rgb_u8(image)
    if box is not None:
        x, y, w, h = _pad(box, rgb.shape, 0.06)
        rgb = rgb[y:y + h, x:x + w]
    if rgb.size == 0:
        return "", 0.0
    if enhance:
        from .conditions import enhance as _enhance
        restored, steps = _enhance(rgb)
        if steps:
            reads = [read_plate(rgb, pattern=pattern), read_plate(np.asarray(restored.convert("RGB")), pattern=pattern)]
            return max(reads, key=lambda r: r[1])
    if rgb.shape[0] < 48:                                   # OCR reads small crops better upscaled
        cv2 = _cv2()
        f = 48 / rgb.shape[0]
        rgb = cv2.resize(rgb, (max(1, int(rgb.shape[1] * f)), 48), interpolation=cv2.INTER_CUBIC)
    lines = [t for t in ocr_text(rgb) if t[1].strip()]
    if not lines:
        return "", 0.0
    lines.sort(key=lambda t: (t[0][1], t[0][0]))             # two-line plates: top line first
    text = clean_plate_text("".join(t[1] for t in lines), pattern=pattern)
    conf = float(np.mean([t[2] for t in lines]))
    return text, conf


def detect_plates(image, *, read: bool = True, pattern=None, min_width: int = 40, enhance: bool = False,
                  min_confidence: float = 0.5, max_candidates: int = 6) -> list:
    """Licence plates in an image, best first.

    ``read=True`` (needs ``[ocr]``) reads each plate and keeps only regions that read as plate text.
    ``read=False`` returns every plate-like region (for blurring); OCR text boxes are still used
    when OCR is installed. ``enhance=True`` also reads a restored copy (dehazed / brightened /
    denoised as needed) and keeps the more confident read of each plate (about twice the time)."""
    rgb = _rgb_u8(image)
    if enhance:
        from .conditions import enhance as _enhance
        restored, steps = _enhance(rgb)
        kw = dict(read=read, pattern=pattern, min_width=min_width, min_confidence=min_confidence,
                  max_candidates=max_candidates)
        as_taken = detect_plates(rgb, **kw)
        if not steps:
            return as_taken
        # Restoration can also hurt (denoising snow, sharpening blur), so read both images and keep
        # the more confident read of each plate: OCR confidence tracks correctness well here.
        both = sorted(as_taken + detect_plates(np.asarray(restored.convert("RGB")), **kw),
                      key=lambda p: (-p.confidence, -p.box[2] * p.box[3]))
        merged: list = []
        for p in both:
            if all(_iou(p.box, k.box) < 0.3 for k in merged):
                merged.append(p)
        return merged
    use_ocr = read or ocr_available()
    found: list = []
    text_boxes: list = []
    if use_ocr:
        for box, text, conf in ocr_text(rgb):
            text_boxes.append(box)
            if box[2] >= min_width * 0.6 and box[2] >= box[3] and looks_like_plate(text, pattern=pattern):
                found.append(Plate(_pad(box, rgb.shape, 0.04), clean_plate_text(text, pattern=pattern), conf, "ocr"))
    cv2 = _cv2()
    gray = cv2.equalizeHist(cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY))
    candidates = [("cascade", tuple(int(v) for v in b)) for b in _cascade("haarcascade_russian_plate_number.xml")
                  .detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4, minSize=(min_width, min_width // 4))]
    if read:
        # Reading the whole image is the most reliable detector; region proposals are a fallback
        # (small or low-contrast plates the full-image pass missed), and each one costs an OCR call.
        if found:
            candidates = []
        else:
            candidates += [("shape", b) for b in _shape_candidates(rgb, min_width=min_width)]
            candidates = candidates[:max_candidates]
    elif not use_ocr:
        candidates += [("shape", b) for b in _shape_candidates(rgb, min_width=min_width)]
    for source, box in candidates:
        if any(_iou(box, p.box) > 0.2 for p in found):
            continue
        if read:
            text, conf = read_plate(rgb, box, pattern=pattern)
            if not looks_like_plate(text, pattern=pattern):
                continue
            found.append(Plate(box, text, conf, source))
        elif not use_ocr or any(_iou(box, t) > 0.1 for t in text_boxes):
            # with OCR at hand, a detector hit must contain text (even text too degraded to read)
            found.append(Plate(box, "", 0.0, source))
    if read:
        found = [p for p in found if p.confidence >= min_confidence]
    found.sort(key=lambda p: (-p.confidence, -p.box[2] * p.box[3]))
    keep: list = []
    for p in found:
        if all(_iou(p.box, k.box) < 0.3 for k in keep):
            keep.append(p)
    return keep


def plate_report(source, *, pattern=None, enhance: bool = False, conditions: bool = True) -> pl.DataFrame:
    """One row per image: plates found, their text and confidence, and the weather / light
    conditions in the image (so you can see *why* a plate was missed)."""
    rows = []
    for path, im in _items(source):
        img = im if im is not None else path
        try:
            found = detect_plates(img, pattern=pattern, enhance=enhance)
            row = {"path": str(path), "plates": len(found), "text": ", ".join(p.text for p in found),
                   "confidence": round(float(np.mean([p.confidence for p in found])), 3) if found else None,
                   "error": None}
            if conditions:
                from .conditions import detect_conditions
                row["conditions"] = ", ".join(detect_conditions(img).tags)
        except Exception as exc:  # noqa: BLE001 - one bad file should not stop the report
            row = {"path": str(path), "plates": 0, "text": "", "confidence": None, "error": str(exc)}
            if conditions:
                row["conditions"] = ""
        rows.append(row)
    return pl.DataFrame(rows, schema_overrides={"confidence": pl.Float64, "error": pl.Utf8})
