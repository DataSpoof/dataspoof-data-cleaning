"""Faces: find them, measure how much of each is covered (hair, mask, sunglasses), simulate
coverings for training, and blur faces for privacy.

Needs OpenCV: ``pip install "dataspoof-mlpilot[vision]"`` (uses the face and eye detectors that
ship with OpenCV 4, so nothing is downloaded at run time).

>>> from mlpilot.media import faces
>>> found = faces.detect_faces(img)                    # FaceBox(x, y, w, h, eyes=[...])
>>> faces.face_occlusion(img)                          # how much of the face is covered, and by what
OcclusionResult(score=0.46, occluded=True, likely='hair', regions={'forehead': 0.9, 'eyes': 0.6, 'lower': 0.0})
>>> faces.occlusion_report("id_photos/")               # one row per image: faces, coverage, eyes visible
>>> faces.add_occluder(img, "hair", severity=4)        # training augmentation (hair, mask, sunglasses, cutout)
>>> faces.anonymize(img)                               # blur every face (and licence plate)

How coverage is measured: the face is split into forehead, eye band and lower face; in each, the
share of skin-coloured pixels is compared with the *best-visible* region of the same face (so
skin tone and lighting cancel out), and the eye detector checks whether the eyes are visible.
A dark covering over the forehead and eyes reads as hair, a dark band over the eyes as
sunglasses, a covering over the lower face as a mask. These are classical detectors: good for
auditing datasets and flagging images for review, not a replacement for a trained face model.
"""

from __future__ import annotations

import dataclasses

import numpy as np
import polars as pl

from .image import _items, _pil

REGIONS = {"forehead": (0.02, 0.27), "eyes": (0.27, 0.52), "lower": (0.58, 0.97)}
WEIGHTS = {"forehead": 0.25, "eyes": 0.4, "lower": 0.35}
FLAT_OBJECT = 0.6               # share of untextured covered pixels above which a covering is an object


def _cv2():
    try:
        import cv2
    except ImportError as exc:
        raise ImportError('Face and plate detection need OpenCV: pip install "dataspoof-mlpilot[vision]"') from exc
    return cv2


_CASCADES: dict = {}


def _cascade(name: str):
    cv2 = _cv2()
    if name not in _CASCADES:
        path = cv2.data.haarcascades + name
        c = cv2.CascadeClassifier(path)
        if c.empty():
            raise RuntimeError(f"OpenCV cascade {name} not found: install opencv-python-headless<5 "
                               "(OpenCV 5 no longer bundles the Haar models)")
        _CASCADES[name] = c
    return _CASCADES[name]


def _rgb_u8(image) -> np.ndarray:
    """A PIL image, a path or an RGB array -> uint8 RGB array."""
    if isinstance(image, np.ndarray):
        arr = image if image.dtype == np.uint8 else (np.clip(image, 0, 1) * 255).astype(np.uint8)
        return np.stack([arr] * 3, axis=2) if arr.ndim == 2 else arr[..., :3]
    Image, _ = _pil()
    if not isinstance(image, Image.Image):
        from .image import open_image
        image = open_image(image, "RGB")
    return np.asarray(image.convert("RGB"))


@dataclasses.dataclass
class FaceBox:
    x: int
    y: int
    w: int
    h: int
    eyes: list = dataclasses.field(default_factory=list)       # [(x, y, w, h)] in image coordinates

    @property
    def box(self) -> tuple:
        return (self.x, self.y, self.w, self.h)


def detect_faces(image, *, min_size: int = 40, scale: float = 1.1, neighbors: int = 5, eyes: bool = True,
                 relaxed: bool = False) -> list:
    """Frontal faces (largest first), each with the eyes found inside it.

    ``relaxed=True`` adds looser passes when nothing is found, which catch faces that are heavily
    covered (hair over the eyes, sunglasses) at the cost of occasional false faces. The coverage
    report and :func:`anonymize` use it: there a missed face costs more than a false one."""
    cv2 = _cv2()
    rgb = _rgb_u8(image)
    gray = cv2.equalizeHist(cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY))
    passes = [("haarcascade_frontalface_default.xml", neighbors), ("haarcascade_frontalface_alt2.xml", neighbors)]
    if relaxed:
        passes += [("haarcascade_frontalface_alt2.xml", min(neighbors, 3)), ("haarcascade_frontalface_alt.xml", 2)]
    found: list = []
    for model, nb in passes:     # the alternative models catch some faces the default misses
        found = list(_cascade(model).detectMultiScale(gray, scaleFactor=scale, minNeighbors=nb,
                                                      minSize=(min_size, min_size)))
        if found:
            break
    out = []
    for x, y, w, h in sorted((tuple(int(v) for v in f) for f in found), key=lambda f: -f[2] * f[3]):
        face = FaceBox(x, y, w, h)
        if eyes:
            face.eyes = _eyes(gray, face)
        out.append(face)
    return out


def _eyes(gray: np.ndarray, face: FaceBox) -> list:
    y1 = face.y + int(face.h * 0.62)
    roi = gray[face.y:y1, face.x:face.x + face.w]
    found = _cascade("haarcascade_eye.xml").detectMultiScale(roi, scaleFactor=1.05, minNeighbors=4,
                                                            minSize=(max(8, face.w // 10),) * 2)
    eyes = [(int(face.x + ex), int(face.y + ey), int(ew), int(eh)) for ex, ey, ew, eh in found
            if ey + eh / 2 < face.h * 0.6]
    return sorted(eyes, key=lambda e: -e[2] * e[3])[:2]


# ── coverage / occlusion ────────────────────────────────────────────────────
@dataclasses.dataclass
class OcclusionResult:
    face: FaceBox | None
    score: float                 # 0 = fully visible, 1 = fully covered
    occluded: bool
    likely: str | None           # hair | mask | sunglasses | object | None
    regions: dict                # region -> share covered (0-1)
    eyes_visible: int
    skin_reliable: bool

    def __repr__(self) -> str:
        if self.face is None:
            return "OcclusionResult(no face found)"
        regs = {k: round(v, 2) for k, v in self.regions.items()}
        return (f"OcclusionResult(score={self.score:.2f}, occluded={self.occluded}, likely={self.likely!r}, "
                f"regions={regs}, eyes_visible={self.eyes_visible})")


def _skin_mask(rgb: np.ndarray) -> np.ndarray:
    cv2 = _cv2()
    ycc = cv2.cvtColor(rgb, cv2.COLOR_RGB2YCrCb)
    lum, cr, cb = ycc[..., 0], ycc[..., 1], ycc[..., 2]
    # hue alone also accepts dark brown (hair, shadows): skin is never that dark in a usable photo
    return (lum >= 55) & (cr >= 133) & (cr <= 178) & (cb >= 72) & (cb <= 130)


def face_occlusion(image, face: FaceBox | None = None, *, threshold: float = 0.25) -> OcclusionResult:
    """How much of ``face`` (default: the largest face found) is covered, and by what."""
    rgb = _rgb_u8(image)
    if face is None:
        found = detect_faces(rgb, relaxed=True)
        if not found:
            return OcclusionResult(None, float("nan"), False, None, {}, 0, False)
        face = found[0]
    else:                         # a known face location: the eyes must be looked for in *this* image
        cv2 = _cv2()
        face = dataclasses.replace(face, eyes=_eyes(cv2.equalizeHist(cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)), face))
    x, y, w, h = face.box
    crop = rgb[max(0, y):y + h, max(0, x):x + w]
    skin = _skin_mask(crop)
    lum = crop.astype(np.float32).mean(axis=2) / 255.0
    x0, x1 = int(0.18 * w), int(0.82 * w)
    slices = {name: (slice(int(a * h), int(b * h)), slice(x0, x1)) for name, (a, b) in REGIONS.items()}

    def fractions(mask):
        return {n: float(mask[sl].mean()) if mask[sl].size else 0.0 for n, sl in slices.items()}
    frac = fractions(skin)
    # Brown hair has the same hue as skin, but it is darker than this person's lit skin: keep only
    # skin-hued pixels close to the skin brightness of the best-visible region.
    # The reference is the *brightest* skin-hued region: covered regions (hair) are darker.
    medians = [float(np.median(lum[sl][skin[sl]])) for sl in slices.values() if skin[sl].sum() >= 20]
    if medians:
        ref = max(medians)
        skin = skin & (lum >= 0.7 * ref)
        frac = fractions(skin)
    # Texture of what covers the face: hair is strands (strong local gradients), a sticker, hand
    # or block is flat.
    gy, gx = np.gradient(lum)
    grad = np.hypot(gx, gy)
    flat = {n: float((grad[sl][~skin[sl]] < 0.015).mean()) if (~skin[sl]).sum() >= 20 else 0.0
            for n, sl in slices.items()}
    dark = {n: float((lum[sl] < 0.22).mean()) if lum[sl].size else 0.0 for n, sl in slices.items()}
    best = max(frac.values())
    reliable = best >= 0.3
    covered = {k: float(np.clip(1 - v / best, 0, 1)) if reliable else 0.0 for k, v in frac.items()}
    eyes_visible = len(face.eyes)
    if eyes_visible == 0:
        covered["eyes"] = max(covered["eyes"], 0.7)
    elif eyes_visible == 1:
        covered["eyes"] = max(covered["eyes"], 0.35)
    score = float(sum(WEIGHTS[k] * v for k, v in covered.items()))
    likely = None
    if score >= threshold:
        top, eyes_c, low = covered["forehead"], covered["eyes"], covered["lower"]
        if low >= 0.4 and low >= max(top, eyes_c):
            likely = "mask" if dark["lower"] < 0.5 else "object"
        elif eyes_c >= 0.4 and dark["eyes"] >= 0.25 and dark["eyes"] >= dark["forehead"] + 0.15:
            likely = "sunglasses"            # a very dark band at eye level, darker than above it
        elif max(top, eyes_c) >= 0.3:
            # stringy, darker-than-skin covering over the forehead / eyes; a flat one is an object
            flatness = max(flat[r] for r in ("forehead", "eyes") if covered[r] >= 0.3)
            likely = "object" if flatness >= FLAT_OBJECT else "hair"
        else:
            likely = "object"
    return OcclusionResult(face, score, score >= threshold, likely, covered, eyes_visible, reliable)


def occlusion_report(source, *, threshold: float = 0.25) -> pl.DataFrame:
    """One row per image: faces found, coverage of the largest face, what covers it, eyes visible.
    Images where no face is found are reported too (often the most heavily covered ones)."""
    rows = []
    for name, im in _items(source):
        try:
            rgb = _rgb_u8(im if im is not None else name)
            found = detect_faces(rgb, relaxed=True)
            r = face_occlusion(rgb, found[0], threshold=threshold) if found else None
        except Exception as exc:
            rows.append({"path": name, "faces": None, "error": f"{type(exc).__name__}: {exc}"[:200]})
            continue
        rows.append({"path": name, "faces": len(found), "error": None,
                     "coverage": round(r.score, 4) if r else None, "occluded": r.occluded if r else None,
                     "likely": r.likely if r else None, "eyes_visible": r.eyes_visible if r else None,
                     **({f"covered_{k}": round(v, 4) for k, v in r.regions.items()} if r else {})})
    return pl.DataFrame(rows, infer_schema_length=None)


# ── simulated coverings (augmentation) ──────────────────────────────────────
OCCLUDERS = ("hair", "mask", "sunglasses", "cutout")


def add_occluder(image, kind: str, *, severity: float = 3, seed: int = 0, face: FaceBox | None = None):
    """A copy of ``image`` with a face partly covered: ``hair`` (strands falling over the forehead
    and eyes), ``mask`` (a face mask over nose and mouth), ``sunglasses`` or ``cutout`` (a block).
    The face is found automatically when OpenCV is installed, otherwise the centre is used."""
    Image, _ = _pil()
    from PIL import ImageDraw, ImageFilter
    if kind not in OCCLUDERS:
        raise ValueError(f"kind must be one of {OCCLUDERS}")
    s = float(severity)
    if not 1 <= s <= 5:
        raise ValueError("severity runs from 1 (mild) to 5 (extreme)")
    rng = np.random.default_rng(seed)
    im = Image.fromarray(_rgb_u8(image)).copy()
    W, H = im.size
    if face is None:
        try:
            found = detect_faces(im)
            face = found[0] if found else None
        except ImportError:
            face = None
    if face is None:
        fw = min(W, H) // 3
        face = FaceBox((W - fw) // 2, (H - fw) // 2, fw, fw)
    x, y, w, h = face.box
    d = ImageDraw.Draw(im)
    if kind == "hair":
        # a fringe of thin, curved, semi-transparent strands falling from above the hairline
        palette = [(25, 18, 14), (45, 30, 20), (70, 48, 30), (15, 12, 12), (95, 70, 45)]
        base = palette[int(rng.integers(0, len(palette)))]
        layer = Image.new("RGBA", im.size, (0, 0, 0, 0))
        ld = ImageDraw.Draw(layer)
        reach = y + h * (0.2 + 0.09 * s)                           # how far down the fringe falls
        half = w * (0.16 + 0.06 * s)
        side = rng.choice([-1.0, 1.0])                              # swept to one side, like a real fringe
        for _ in range(int(350 + 150 * s)):
            cx, cy = x + w * 0.5 + rng.uniform(-half, half), y - h * rng.uniform(0.1, 0.3)
            end = y + (reach - y) * rng.uniform(0.75, 1.05)
            sweep, curl = side * rng.uniform(0.1, 0.45), side * rng.uniform(0.0, 0.03)
            m = rng.uniform(0.75, 1.3)                              # one brightness per strand, same hue
            shade = tuple(int(np.clip(c * m, 0, 255)) for c in base)
            pts = []
            while cy < end:
                pts.append((cx, cy))
                cy += h * 0.02
                sweep += curl
                cx += h * 0.02 * sweep
            for i in range(len(pts) - 1):                           # taper: thinner and fainter at the tip
                t = i / max(len(pts) - 1, 1)
                ld.line([pts[i], pts[i + 1]], fill=shade + (int(250 - 70 * t),),
                        width=max(1, int(round((2.0 + 0.3 * s) * (1 - 0.5 * t)))))
        im = Image.alpha_composite(im.convert("RGBA"), layer.filter(ImageFilter.GaussianBlur(0.35))).convert("RGB")
        d = ImageDraw.Draw(im)
    elif kind == "mask":
        top = y + h * (0.6 - 0.04 * s)
        poly = [(x + w * 0.1, top), (x + w * 0.9, top), (x + w * 0.85, y + h * 1.02), (x + w * 0.5, y + h * 1.1),
                (x + w * 0.15, y + h * 1.02)]
        d.polygon(poly, fill=(170, 200, 230))
        for side in (0.1, 0.9):
            d.line([(x + w * side, top + 2), (x + w * (side - 0.08 if side < 0.5 else side + 0.08), y + h * 0.45)],
                   fill=(230, 230, 230), width=2)
    elif kind == "sunglasses":
        if len(face.eyes) == 2:
            centres = [(ex + ew / 2, ey + eh / 2) for ex, ey, ew, eh in face.eyes]
        else:
            centres = [(x + w * 0.32, y + h * 0.42), (x + w * 0.68, y + h * 0.42)]
        rx, ry = w * (0.12 + 0.015 * s), h * (0.08 + 0.012 * s)
        for cx, cy in centres:
            d.ellipse([cx - rx, cy - ry, cx + rx, cy + ry], fill=(12, 12, 16))
        (ax, ay), (bx, by) = sorted(centres)
        d.line([(ax + rx * 0.8, ay), (bx - rx * 0.8, by)], fill=(12, 12, 16), width=max(2, int(w * 0.03)))
    else:  # cutout
        side = int(w * (0.25 + 0.1 * s))
        x0 = int(x + rng.uniform(0, max(1, w - side)))
        y0 = int(y + rng.uniform(0, max(1, h - side)))
        d.rectangle([x0, y0, x0 + side, y0 + side], fill=tuple(int(c) for c in rng.integers(0, 255, 3)))
    return im.filter(ImageFilter.GaussianBlur(0.4))


# ── privacy ─────────────────────────────────────────────────────────────────
def blur_regions(image, boxes, *, method: str = "blur", strength: float = 1.0, pad: float = 0.1):
    """Blur (or pixelate) rectangular ``boxes`` ``[(x, y, w, h)]`` so they can't be recognised."""
    Image, _ = _pil()
    from PIL import ImageFilter
    im = Image.fromarray(_rgb_u8(image)).copy()
    for x, y, w, h in boxes:
        px, py = int(w * pad), int(h * pad)
        box = (max(0, x - px), max(0, y - py), min(im.width, x + w + px), min(im.height, y + h + py))
        region = im.crop(box)
        if method == "pixelate":
            k = max(4, int(min(region.size) / (6 / strength)))
            small = region.resize((max(1, region.width // k), max(1, region.height // k)), Image.Resampling.BILINEAR)
            region = small.resize(region.size, Image.Resampling.NEAREST)
        else:
            region = region.filter(ImageFilter.GaussianBlur(max(region.size) * 0.12 * strength))
        im.paste(region, box[:2])
    return im


def anonymize(image, *, faces: bool = True, plates: bool = True, method: str = "blur") -> tuple:
    """Blur every face and licence plate. Returns ``(image, {"faces": [...], "plates": [...]})``."""
    rgb = _rgb_u8(image)
    found: dict = {"faces": [], "plates": []}
    if faces:
        found["faces"] = [f.box for f in detect_faces(rgb, eyes=False, relaxed=True)]
    if plates:
        from .plates import detect_plates
        found["plates"] = [p.box for p in detect_plates(rgb, read=False)]
    Image, _ = _pil()
    out = blur_regions(Image.fromarray(rgb), found["faces"] + found["plates"], method=method)
    return out, found
