# Weather, faces & plates: API reference

Generated from the source docstrings. For a task-oriented introduction, read the [Weather, faces & plates guide](../guide/vision.md).

**On this page:**

- [`mlpilot.media.benchmark`](#mlpilotmediabenchmark): `BenchmarkReport`, `condition_benchmark`, `condition_augment`, `image_classifier`, `plate_reader`, `face_counter`
- [`mlpilot.media.conditions`](#mlpilotmediaconditions): `dark_channel`, `noise_sigma`, `ConditionResult`, `condition_scores`, `detect_conditions`, `condition_report`, `apply_condition`, `dehaze`, `brighten`, `denoise`, `sharpen`, `enhance`
- [`mlpilot.media.faces`](#mlpilotmediafaces): `FaceBox`, `detect_faces`, `OcclusionResult`, `face_occlusion`, `occlusion_report`, `add_occluder`, `blur_regions`, `anonymize`
- [`mlpilot.media.plates`](#mlpilotmediaplates): `Plate`, `ocr_available`, `ocr_text`, `clean_plate_text`, `looks_like_plate`, `read_plate`, `detect_plates`, `plate_report`

---

## `mlpilot.media.benchmark`

<sub>Source: [`mlpilot/media/benchmark.py`](https://github.com/DataSpoof/dataspoof-mlpilot/blob/main/mlpilot/media/benchmark.py)</sub>

Robustness benchmark for image models: weather, low light, blur, noise and covered faces.

How well does a model hold up in fog, rain, snow, at night, through a dirty lens, or when faces
are partly covered? Re-score it on degraded copies of a test set, one condition and severity
at a time.

```python
from mlpilot.media import benchmark
predict = benchmark.image_classifier(model)                 # a fitted AutoML on image_features
rep = benchmark.condition_benchmark(predict, test_images, test_labels)
print(rep.summary())                                       # clean score, then the worst conditions
rep.table                                                  # condition x severity -> score, drop
rep.passed(max_drop=0.10)                                  # gate a release on robustness
rep = benchmark.condition_benchmark(benchmark.plate_reader("IN"), cars, plate_texts)
```

Works with anything that maps a list of PIL images to a list of predictions: an AutoML model on
`image_features`, a plate reader, a face detector (`labels` = expected face counts), or your
own deep model. `score` defaults to exact-match accuracy; pass any `score(y_true, y_pred)`.

`condition_augment` produces the same degraded copies for *training*, so the model sees the
conditions it is weakest at.

### class `BenchmarkReport`

```python
class BenchmarkReport:
```

| Field | Type | Default |
|---|---|---|
| `clean` | `float` | required |
| `table` | `pl.DataFrame` | required |
| `by_condition` | `pl.DataFrame` | required |
| `errors` | `dict` | required |

#### `BenchmarkReport.weakest` *(property)*

```python
weakest: list
```

Conditions ordered from the largest worst-case drop down.

#### `BenchmarkReport.passed()`

```python
passed(*, max_drop: float=0.1) -> bool
```

#### `BenchmarkReport.summary()`

```python
summary(top: int=5) -> str
```

### `condition_benchmark()`

```python
def condition_benchmark(predict: Callable, images, labels, *, conditions=None, severities=(1, 3, 5), score: Callable | None=None, faces: bool=False, seed: int=0) -> BenchmarkReport
```

Score `predict` on clean and degraded copies of `images`.

`predict(list_of_pil_images) -> predictions`; `labels` are the expected predictions.
`conditions` default to every simulated condition (fog, rain, snow, low_light, glare,
motion_blur, noise, jpeg, shadow, cutout); `faces=True` adds hair / mask / sunglasses /
cutout coverings over detected faces (needs `[vision]`).

### `condition_augment()`

```python
def condition_augment(images, labels=None, *, conditions=None, severities=(2, 4), copies: int=1, seed: int=0) -> tuple
```

Training augmentation: `copies` degraded versions of every image, each with a random
condition and severity. Returns `(images, labels, conditions_used)` *including* the originals,
so `labels` stay aligned (`None` labels are passed through as `None`).

### `image_classifier()`

```python
def image_classifier(model, *, features: Callable | None=None, keep_path: bool=False, **feature_kwargs) -> Callable
```

Wrap a model trained on `image_features` (an `AutoML` or anything with
`predict(DataFrame)`) as `predict(images)`. Pass the same `feature_kwargs` used in training;
the `path` column is dropped unless `keep_path=True`.

### `plate_reader()`

```python
def plate_reader(pattern=None, *, enhance: bool=False) -> Callable
```

`predict(images)` -> the text of the most confident plate in each image ("" if none).

### `face_counter()`

```python
def face_counter(**kwargs) -> Callable
```

`predict(images)` -> number of faces found in each image (labels = expected counts).

---

## `mlpilot.media.conditions`

<sub>Source: [`mlpilot/media/conditions.py`](https://github.com/DataSpoof/dataspoof-mlpilot/blob/main/mlpilot/media/conditions.py)</sub>

Weather, lighting and capture conditions: detect them, simulate them, and undo them.

Real-world images are taken in fog, rain, snow, at night, against the sun, from a moving car.
A model trained only on clear daylight images fails silently on the rest. This module:

```python
from mlpilot.media import conditions as cond
cond.detect_conditions("camera_042.jpg")
```
ConditionResult(tags=['fog', 'low_light'], scores={'fog': 0.71, 'low_light': 0.64, ...})
```python
cond.condition_report("traffic_cams/")          # one row per image + coverage per condition
foggy = cond.apply_condition(img, "fog", severity=4, seed=1)        # training augmentation
clear = cond.enhance(foggy)                                         # dehaze / brighten / denoise
```

Detected: `fog` (haze), `rain`, `snow`, `low_light`, `glare` (sun flare, headlights),
`motion_blur` (any strong blur; the `anisotropy` statistic separates motion from focus blur),
`noise`, `jpeg` (compression artefacts). `shadow` and `cutout` can be *simulated* but are
not detected: brightness statistics cannot tell a cast shadow from a dark object. Each
detector returns a 0-1 score; a condition is *tagged* when its score passes a threshold
(`THRESHOLDS`, adjustable). The detectors are classical image statistics: dark-channel haze
estimation, an Immerkaer noise estimate, gradient anisotropy for motion blur, oriented
bright-streak analysis for rain, small bright blobs for snow, 8x8 block-edge energy for JPEG.
They work with Pillow + numpy + scipy alone (no OpenCV, no GPU) and are meant for auditing and
routing data, not as a substitute for a trained weather classifier.

### `dark_channel()`

```python
def dark_channel(rgb: np.ndarray, patch: int=15) -> np.ndarray
```

Per-pixel minimum over colour channels and a local patch (He et al.'s haze prior):
close to 0 in clear outdoor images, high where haze adds a bright veil.

### `noise_sigma()`

```python
def noise_sigma(gray: np.ndarray) -> float
```

Immerkaer's fast noise estimate (standard deviation of additive noise, 0-1 scale).

### class `ConditionResult`

```python
class ConditionResult:
```

| Field | Type | Default |
|---|---|---|
| `scores` | `dict` | required |
| `tags` | `list` | required |
| `stats` | `dict` | required |

#### `ConditionResult.clear` *(property)*

```python
clear: bool
```

### `condition_scores()`

```python
def condition_scores(image) -> tuple
```

`(scores, stats)`: a 0-1 score per condition, and the raw statistics behind them.

### `detect_conditions()`

```python
def detect_conditions(image, *, thresholds: dict | None=None) -> ConditionResult
```

Score and tag the conditions an image was taken in (see the module docstring).

### `condition_report()`

```python
def condition_report(source, *, thresholds: dict | None=None, min_share: float=0.05) -> tuple
```

`(per_image, coverage)` for a folder / glob / list of paths or PIL images.

`coverage` counts images per condition and flags conditions that are present but rarer
than `min_share` of the data, and conditions with no examples at all: a model will not
learn to handle what it has (almost) never seen.

### `apply_condition()`

```python
def apply_condition(image, condition: str, *, severity: float=3, seed: int=0)
```

A copy of `image` (PIL) as if taken in `condition` at `severity` 1-5. Reproducible
for a given `seed`. Use it to augment training data or to stress-test a model
(`media.benchmark.condition_benchmark`).

### `dehaze()`

```python
def dehaze(image, *, strength: float=0.9)
```

Remove fog / haze with the dark-channel prior and a guided-filter transmission map.

### `brighten()`

```python
def brighten(image, *, target: float=0.45)
```

Low-light enhancement: adaptive gamma towards `target` mean brightness, then CLAHE when
OpenCV is available (local contrast without blowing out highlights).

### `denoise()`

```python
def denoise(image, *, strength: float | None=None)
```

Reduce sensor noise: non-local means with OpenCV, otherwise an edge-preserving median.

### `sharpen()`

```python
def sharpen(image, *, amount: float=1.0, radius: float=2.0)
```

Unsharp mask: recovers edge detail lost to mild blur (motion or focus).

### `enhance()`

```python
def enhance(image, steps='auto')
```

Restore an image. `steps="auto"` picks from the detected conditions (fog -> dehaze,
low light -> brighten, noise -> denoise, blur -> sharpen); or pass a list such as
`["dehaze", "denoise"]`. Returns `(image, steps_applied)`.

---

## `mlpilot.media.faces`

<sub>Source: [`mlpilot/media/faces.py`](https://github.com/DataSpoof/dataspoof-mlpilot/blob/main/mlpilot/media/faces.py)</sub>

Faces: find them, measure how much of each is covered (hair, mask, sunglasses), simulate
coverings for training, and blur faces for privacy.

Needs OpenCV: `pip install "dataspoof-mlpilot[vision]"` (uses the face and eye detectors that
ship with OpenCV 4, so nothing is downloaded at run time).

```python
from mlpilot.media import faces
found = faces.detect_faces(img)                    # FaceBox(x, y, w, h, eyes=[...])
faces.face_occlusion(img)                          # how much of the face is covered, and by what
```
OcclusionResult(score=0.46, occluded=True, likely='hair', regions={'forehead': 0.9, 'eyes': 0.6, 'lower': 0.0})
```python
faces.occlusion_report("id_photos/")               # one row per image: faces, coverage, eyes visible
faces.add_occluder(img, "hair", severity=4)        # training augmentation (hair, mask, sunglasses, cutout)
faces.anonymize(img)                               # blur every face (and licence plate)
```

How coverage is measured: the face is split into forehead, eye band and lower face; in each, the
share of skin-coloured pixels is compared with the *best-visible* region of the same face (so
skin tone and lighting cancel out), and the eye detector checks whether the eyes are visible.
A dark covering over the forehead and eyes reads as hair, a dark band over the eyes as
sunglasses, a covering over the lower face as a mask. These are classical detectors: good for
auditing datasets and flagging images for review, not a replacement for a trained face model.

### class `FaceBox`

```python
class FaceBox:
```

| Field | Type | Default |
|---|---|---|
| `x` | `int` | required |
| `y` | `int` | required |
| `w` | `int` | required |
| `h` | `int` | required |
| `eyes` | `list` | `list()` |

#### `FaceBox.box` *(property)*

```python
box: tuple
```

### `detect_faces()`

```python
def detect_faces(image, *, min_size: int=40, scale: float=1.1, neighbors: int=5, eyes: bool=True, relaxed: bool=False) -> list
```

Frontal faces (largest first), each with the eyes found inside it.

`relaxed=True` adds looser passes when nothing is found, which catch faces that are heavily
covered (hair over the eyes, sunglasses) at the cost of occasional false faces. The coverage
report and `anonymize` use it: there a missed face costs more than a false one.

### class `OcclusionResult`

```python
class OcclusionResult:
```

| Field | Type | Default |
|---|---|---|
| `face` | `FaceBox \| None` | required |
| `score` | `float` | required |
| `occluded` | `bool` | required |
| `likely` | `str \| None` | required |
| `regions` | `dict` | required |
| `eyes_visible` | `int` | required |
| `skin_reliable` | `bool` | required |

### `face_occlusion()`

```python
def face_occlusion(image, face: FaceBox | None=None, *, threshold: float=0.25) -> OcclusionResult
```

How much of `face` (default: the largest face found) is covered, and by what.

### `occlusion_report()`

```python
def occlusion_report(source, *, threshold: float=0.25) -> pl.DataFrame
```

One row per image: faces found, coverage of the largest face, what covers it, eyes visible.
Images where no face is found are reported too (often the most heavily covered ones).

### `add_occluder()`

```python
def add_occluder(image, kind: str, *, severity: float=3, seed: int=0, face: FaceBox | None=None)
```

A copy of `image` with a face partly covered: `hair` (strands falling over the forehead
and eyes), `mask` (a face mask over nose and mouth), `sunglasses` or `cutout` (a block).
The face is found automatically when OpenCV is installed, otherwise the centre is used.

### `blur_regions()`

```python
def blur_regions(image, boxes, *, method: str='blur', strength: float=1.0, pad: float=0.1)
```

Blur (or pixelate) rectangular `boxes` `[(x, y, w, h)]` so they can't be recognised.

### `anonymize()`

```python
def anonymize(image, *, faces: bool=True, plates: bool=True, method: str='blur') -> tuple
```

Blur every face and licence plate. Returns `(image, {"faces": [...], "plates": [...]})`.

---

## `mlpilot.media.plates`

<sub>Source: [`mlpilot/media/plates.py`](https://github.com/DataSpoof/dataspoof-mlpilot/blob/main/mlpilot/media/plates.py)</sub>

Licence plates: find them, read them (OCR), check how reading holds up in fog / rain / night,
and blur them for privacy.

Needs `pip install "dataspoof-mlpilot[ocr]"` (OpenCV + RapidOCR; the OCR models ship inside the
package, so nothing is downloaded at run time). Without OCR installed, `detect_plates(read=False)`
still finds plate-shaped regions with OpenCV alone.

```python
from mlpilot.media import plates
plates.detect_plates(img)                          # [Plate(box=(x, y, w, h), text='MH12AB1234', confidence=0.99)]
plates.read_plate(img, box, enhance=True)          # dehaze / brighten / denoise first, then read
plates.plate_report("traffic_cam/", pattern="IN")  # one row per image, with weather / light conditions
plates.clean_plate_text("MH l2 AB I234", pattern="IN")
```
'MH12AB1234'

How plates are found: (1) OCR text boxes whose text looks like a plate (4-12 letters and digits,
with both, or matching `pattern`); (2) OpenCV's plate detector; (3) bright, plate-shaped
rectangles. (2) and (3) are a fallback, tried (best `max_candidates` first) only when the
full-image pass reads no plate, and kept only when they read as plate text. With `read=False`
every OCR plate and every detector hit is returned, which is what blurring wants.

### class `Plate`

```python
class Plate:
```

| Field | Type | Default |
|---|---|---|
| `box` | `tuple` | required |
| `text` | `str` | `''` |
| `confidence` | `float` | `0.0` |
| `source` | `str` | `''` |

### `ocr_available()`

```python
def ocr_available() -> bool
```

### `ocr_text()`

```python
def ocr_text(image) -> list
```

Every text line in an image: `[(box, text, confidence)]` with `box = (x, y, w, h)`.

### `clean_plate_text()`

```python
def clean_plate_text(text: str, *, pattern=None) -> str
```

Upper-case letters and digits only. With `pattern` (a regex or one of `PLATE_PATTERNS`),
fixes OCR look-alikes (O/0, I/1, S/5, B/8 ...) until the text matches, or returns it unchanged.

### `looks_like_plate()`

```python
def looks_like_plate(text: str, *, pattern=None) -> bool
```

### `read_plate()`

```python
def read_plate(image, box: tuple | None=None, *, pattern=None, enhance: bool=False) -> tuple
```

OCR one plate (the whole image, or `box`). Returns `(text, confidence)`.

`enhance=True` first removes haze / low light / noise / blur that the condition detector finds
(see `enhance`) and keeps whichever read is more confident.

### `detect_plates()`

```python
def detect_plates(image, *, read: bool=True, pattern=None, min_width: int=40, enhance: bool=False, min_confidence: float=0.5, max_candidates: int=6) -> list
```

Licence plates in an image, best first.

`read=True` (needs `[ocr]`) reads each plate and keeps only regions that read as plate text.
`read=False` returns every plate-like region (for blurring); OCR text boxes are still used
when OCR is installed. `enhance=True` also reads a restored copy (dehazed / brightened /
denoised as needed) and keeps the more confident read of each plate (about twice the time).

### `plate_report()`

```python
def plate_report(source, *, pattern=None, enhance: bool=False, conditions: bool=True) -> pl.DataFrame
```

One row per image: plates found, their text and confidence, and the weather / light
conditions in the image (so you can see *why* a plate was missed).
