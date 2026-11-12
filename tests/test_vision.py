"""Hard-condition vision: weather / light detection and restoration, face coverage (hair, mask,
sunglasses), licence plates, privacy blurring and the robustness benchmark.

The two photos in tests/assets are public-domain scikit-image samples (NASA's astronaut portrait
and a cup of coffee), resized."""

from pathlib import Path

import numpy as np
import polars as pl
import pytest

Image = pytest.importorskip("PIL.Image")
from PIL import ImageDraw, ImageFont  # noqa: E402

from mlpilot.media import benchmark, conditions as cond  # noqa: E402

ASSETS = Path(__file__).parent / "assets"
try:
    import cv2  # noqa: F401
    HAS_CV2 = bool(cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml").empty() is False)
except Exception:  # noqa: BLE001
    HAS_CV2 = False
try:
    import rapidocr  # noqa: F401
    HAS_OCR = HAS_CV2
except ImportError:
    HAS_OCR = False
needs_cv2 = pytest.mark.skipif(not HAS_CV2, reason='needs OpenCV 4 ([vision] extra)')
needs_ocr = pytest.mark.skipif(not HAS_OCR, reason='needs RapidOCR ([ocr] extra)')


def _open(name):
    im = Image.open(ASSETS / name)
    im.load()
    return im.convert("RGB")


@pytest.fixture(scope="module")
def astronaut():
    return _open("astronaut.jpg")


@pytest.fixture(scope="module")
def coffee():
    return _open("coffee.jpg")


def plate_scene(text="MH12AB1234", plate_h=52, face=None):
    """A street-ish scene: road, a car and a white plate with ``text``."""
    rng = np.random.default_rng(0)
    img = Image.fromarray((np.clip(rng.normal(0.45, 0.08, (520, 800, 3)), 0, 1) * 255).astype(np.uint8))
    d = ImageDraw.Draw(img)
    d.rectangle([0, 322, 800, 520], fill=(70, 70, 75))
    d.rounded_rectangle([150, 150, 650, 420], 30, fill=(150, 20, 25))
    d.rectangle([220, 180, 580, 260], fill=(40, 60, 80))
    font = ImageFont.load_default(size=int(plate_h * 0.66))
    tw = d.textlength(text, font=font)
    x0, y0 = 400 - tw / 2 - 12, 320
    d.rectangle([x0, y0, x0 + tw + 24, y0 + plate_h], fill=(245, 245, 240), outline=(0, 0, 0), width=3)
    d.text((x0 + 12, y0 + plate_h * 0.14), text, fill=(10, 10, 10), font=font)
    if face is not None:
        img.paste(face.resize((300, 300)), (0, 0))
    return img


# ── conditions ──────────────────────────────────────────────────────────────
def test_clean_photos_have_no_conditions(astronaut, coffee):
    for im in (astronaut, coffee):
        res = cond.detect_conditions(im)
        assert res.tags == [], res.scores
        assert set(res.scores) == set(cond.CONDITIONS)


@pytest.mark.parametrize("condition", cond.CONDITIONS)
def test_every_simulated_condition_is_detected(astronaut, coffee, condition):
    for im in (astronaut, coffee):
        for s in (3, 5):
            res = cond.detect_conditions(cond.apply_condition(im, condition, severity=s, seed=1))
            assert condition in res.tags, (condition, s, res.scores)


def test_apply_condition_is_reproducible_and_validated(astronaut):
    a = np.asarray(cond.apply_condition(astronaut, "rain", severity=3, seed=5))
    b = np.asarray(cond.apply_condition(astronaut, "rain", severity=3, seed=5))
    c = np.asarray(cond.apply_condition(astronaut, "rain", severity=3, seed=6))
    assert np.array_equal(a, b) and not np.array_equal(a, c)
    for extra in ("shadow", "cutout"):                      # simulate-only conditions
        assert cond.apply_condition(astronaut, extra).size == astronaut.size
    with pytest.raises(ValueError):
        cond.apply_condition(astronaut, "fog", severity=9)
    # arrays work as well as PIL images
    arr = np.asarray(astronaut)
    assert cond.apply_condition(arr, "fog").size == astronaut.size
    assert isinstance(cond.detect_conditions(arr).tags, list)


def _psnr(a, b) -> float:
    a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    return float(10 * np.log10(255 ** 2 / np.mean((a - b) ** 2)))


def test_enhance_removes_fog_and_low_light(astronaut, coffee):
    for im in (astronaut, coffee):
        foggy = cond.apply_condition(im, "fog", severity=4, seed=2)
        fixed, steps = cond.enhance(foggy)
        assert "dehaze" in steps
        assert cond.condition_scores(fixed)[0]["fog"] < cond.condition_scores(foggy)[0]["fog"]
        assert _psnr(fixed, im) > _psnr(foggy, im)
        dark = cond.apply_condition(im, "low_light", severity=4, seed=2)
        bright, steps = cond.enhance(dark)
        assert "brighten" in steps
        assert np.asarray(bright).mean() > np.asarray(dark).mean() + 20
    _, steps = cond.enhance(astronaut)
    assert steps == []                                      # nothing to fix on a clean photo
    with pytest.raises(ValueError):
        cond.enhance(astronaut, steps=["magic"])


def test_condition_report_and_coverage(astronaut, coffee):
    imgs = [astronaut, coffee] + [cond.apply_condition(astronaut, "fog", severity=4, seed=i) for i in range(2)]
    per_image, coverage = cond.condition_report(imgs)
    assert per_image.height == 4
    assert isinstance(coverage, pl.DataFrame)
    fog = coverage.filter(pl.col("condition") == "fog")
    assert fog.height == 1 and fog["images"][0] == 2


# ── faces ───────────────────────────────────────────────────────────────────
@needs_cv2
def test_face_detection_and_clean_coverage(astronaut):
    from mlpilot.media import faces
    found = faces.detect_faces(astronaut)
    assert len(found) == 1 and len(found[0].eyes) == 2
    res = faces.face_occlusion(astronaut)
    assert not res.occluded and res.likely is None and res.eyes_visible == 2


@needs_cv2
@pytest.mark.parametrize("kind,severity", [("hair", 3), ("hair", 5), ("mask", 3), ("sunglasses", 4)])
def test_face_coverings_are_measured_and_named(astronaut, kind, severity):
    from mlpilot.media import faces
    face = faces.detect_faces(astronaut)[0]
    clean = faces.face_occlusion(astronaut, face).score
    for seed in (2, 7):
        covered = faces.add_occluder(astronaut, kind, severity=severity, seed=seed, face=face)
        res = faces.face_occlusion(covered, face)
        assert res.occluded and res.score > clean + 0.15, res
        assert res.likely == kind, res


@needs_cv2
def test_flat_block_is_not_called_hair(astronaut):
    from mlpilot.media import faces
    face = faces.detect_faces(astronaut)[0]
    res = faces.face_occlusion(faces.add_occluder(astronaut, "cutout", severity=3, seed=2, face=face), face)
    assert res.occluded and res.likely != "hair"


@needs_cv2
def test_occlusion_report_rows(astronaut, coffee):
    from mlpilot.media import faces
    hair = faces.add_occluder(astronaut, "hair", severity=4, seed=1)
    rep = faces.occlusion_report([astronaut, hair, coffee])
    assert rep.height == 3
    assert rep["faces"].to_list()[2] == 0                   # no face in a coffee cup
    assert rep["occluded"].to_list()[:2] == [False, True]
    assert rep["likely"][1] == "hair"


@needs_cv2
def test_anonymize_hides_faces(astronaut):
    from mlpilot.media import faces
    out, found = faces.anonymize(astronaut, plates=False)
    assert len(found["faces"]) == 1
    assert faces.detect_faces(out) == []
    px = faces.anonymize(astronaut, plates=False, method="pixelate")[0]
    assert faces.detect_faces(px) == []


# ── plates ──────────────────────────────────────────────────────────────────
def test_plate_text_cleanup():
    from mlpilot.media import plates
    assert plates.clean_plate_text("mh-12 ab 1234") == "MH12AB1234"
    assert plates.clean_plate_text("MH l2 AB I234", pattern="IN") == "MH12AB1234"
    assert plates.clean_plate_text("DL3CAF0O7I", pattern="IN") == "DL3CAF0071"
    assert plates.looks_like_plate("KA05MN4821") and not plates.looks_like_plate("COFFEE")
    assert plates.clean_plate_text("HELLOWORLD", pattern="IN") == "HELLOWORLD"      # left alone if it cannot match


@needs_ocr
def test_plates_are_found_and_read():
    from mlpilot.media import plates
    for text, h in (("MH12AB1234", 52), ("KA05MN4821", 34)):
        found = plates.detect_plates(plate_scene(text, plate_h=h), pattern="IN")
        assert [p.text for p in found] == [text]
        x, y, w, hh = found[0].box
        assert 250 < x + w / 2 < 550 and 300 < y + hh / 2 < 400
        assert found[0].confidence > 0.9


@needs_ocr
def test_no_plates_in_ordinary_photos(astronaut, coffee):
    from mlpilot.media import plates
    for im in (astronaut, coffee):
        assert plates.detect_plates(im) == []


@needs_ocr
@pytest.mark.parametrize("condition", ["fog", "rain", "low_light", "noise"])
def test_plates_read_in_bad_weather(condition):
    from mlpilot.media import plates
    img = cond.apply_condition(plate_scene("DL3CAF0071"), condition, severity=4, seed=0)
    found = plates.detect_plates(img, pattern="IN", enhance=True)
    assert found and found[0].text == "DL3CAF0071"


@needs_ocr
def test_plate_report_and_anonymize(astronaut):
    from mlpilot.media import faces, plates
    scene = plate_scene("MH12AB1234", face=astronaut)
    rep = plates.plate_report([scene, cond.apply_condition(scene, "fog", severity=4)], pattern="IN")
    assert rep["plates"].to_list() == [1, 1]
    assert rep["text"].to_list() == ["MH12AB1234", "MH12AB1234"]
    assert "fog" in rep["conditions"][1]
    out, found = faces.anonymize(scene)
    assert len(found["plates"]) == 1 and len(found["faces"]) == 1
    assert plates.detect_plates(out) == []                  # the plate can no longer be read
    from mlpilot.media.plates import _iou
    assert not any(_iou(f.box, found["faces"][0]) > 0.2 for f in faces.detect_faces(out, relaxed=True))


# ── benchmark ───────────────────────────────────────────────────────────────
def _brightness_classifier(imgs):
    return ["bright" if np.asarray(im).mean() > 100 else "dark" for im in imgs]


def test_condition_benchmark_structure(astronaut, coffee):
    imgs = [astronaut, coffee, astronaut.transpose(Image.Transpose.FLIP_LEFT_RIGHT)]
    labels = _brightness_classifier(imgs)
    rep = benchmark.condition_benchmark(_brightness_classifier, imgs, labels,
                                        conditions=["low_light", "fog", "jpeg"], severities=(1, 5))
    assert rep.clean == 1.0
    assert rep.table.height == 6 and set(rep.table.columns) == {"condition", "severity", "score", "drop", "n"}
    assert rep.weakest[0] == "low_light"                    # darkening breaks a brightness classifier
    assert rep.by_condition.filter(pl.col("condition") == "jpeg")["worst_drop"][0] == 0
    assert not rep.passed(max_drop=0.1)
    assert "low_light" in rep.summary() and "BenchmarkReport" in repr(rep)
    with pytest.raises(ValueError):
        benchmark.condition_benchmark(_brightness_classifier, imgs, labels[:2])


def test_condition_augment_keeps_labels_aligned(astronaut, coffee):
    imgs, labels, used = benchmark.condition_augment([astronaut, coffee], ["a", "c"], copies=2, seed=1)
    assert len(imgs) == len(labels) == len(used) == 6
    assert labels == ["a", "c", "a", "c", "a", "c"]
    assert used[:2] == ["clean", "clean"] and all(u in cond.SIMULATED for u in used[2:])


@needs_cv2
def test_benchmark_face_coverings(astronaut):
    rep = benchmark.condition_benchmark(benchmark.face_counter(), [astronaut], [1], conditions=[],
                                        severities=(5,), faces=True)
    assert set(rep.by_condition["condition"]) == set(benchmark.FACE_OCCLUDERS)
    assert rep.errors == {}


@needs_ocr
def test_benchmark_plate_reader():
    scenes = [plate_scene("MH12AB1234"), plate_scene("KA05MN4821")]
    rep = benchmark.condition_benchmark(benchmark.plate_reader("IN"), scenes, ["MH12AB1234", "KA05MN4821"],
                                        conditions=["fog"], severities=(3,))
    assert rep.clean == 1.0 and rep.table.height == 1
