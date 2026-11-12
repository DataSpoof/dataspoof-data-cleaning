# Hard conditions: weather, covered faces, licence plates

Models trained on clean, well-lit photos fail quietly on the images that matter: a camera in fog
or rain, a night shot, a face half-hidden by hair or a mask. This module finds those images in a
dataset, simulates them for training, restores them before inference, and measures how much
each one costs your model.

```bash
pip install "dataspoof-mlpilot[image]"     # conditions + benchmark (Pillow; nothing else)
pip install "dataspoof-mlpilot[vision]"    # + faces (OpenCV 4)
pip install "dataspoof-mlpilot[ocr]"       # + licence plates (OpenCV + RapidOCR, models bundled)
```

Everything runs on CPU, offline: no model downloads, no GPU.

## Weather and lighting conditions

```python
from mlpilot.media import conditions as cond

cond.detect_conditions(img)            # ConditionResult(tags=['fog', 'low_light'], scores={...})
per_image, coverage = cond.condition_report("camera_frames/")
print(coverage)                        # images per condition; "rare" / "missing" ones flagged

foggy = cond.apply_condition(img, "fog", severity=4, seed=0)    # augmentation, severity 1-5
fixed, steps = cond.enhance(foggy)     # steps == ['dehaze']; or enhance(img, ["denoise", "brighten"])
```

| Condition | Detected by | Restored by (`enhance`) |
|---|---|---|
| `fog` / haze | dark-channel prior (a bright veil lifts the darkest pixels) | `dehaze`: dark channel + guided filter |
| `rain` | many thin bright streaks pointing the same way | — |
| `snow` | many small round bright flakes spread over the whole frame | — |
| `low_light` | mean and 95th-percentile brightness | `brighten`: adaptive gamma + CLAHE |
| `glare` | share of near-white, blown-out pixels | — |
| `motion_blur` | edge energy relative to contrast | `sharpen`: unsharp mask |
| `noise` | Immerkaer noise estimate on a full-resolution crop | `denoise`: non-local means |
| `jpeg` | 8×8 block-edge strength | — |
| `shadow`, `cutout` | simulate only | — |

Each score runs from 0 to 1, and an image is tagged at 0.5. You can change that per condition with
`thresholds={"fog": 0.4}`. In testing on four sample photos, every clean photo scored below 0.5,
and every condition was detected at severities 3 and 5.

**Why coverage matters:** `condition_report` tells you that only 1% of your frames are at
night, or that none are in snow. A model can't learn to handle conditions it has never seen.
Fill the gap with `apply_condition`, or with `benchmark.condition_augment`, which adds random
conditions and keeps labels aligned.

## Faces: partly covered by hair, a mask, sunglasses

```python
from mlpilot.media import faces

faces.detect_faces(img)                # [FaceBox(x, y, w, h, eyes=[...])]
faces.face_occlusion(img)              # OcclusionResult(score=0.56, occluded=True, likely='hair', ...)
faces.occlusion_report("id_photos/")   # one row per photo: faces, coverage, likely cause, eyes visible
faces.add_occluder(img, "hair", severity=3)   # also "mask", "sunglasses", "cutout"
```

How it works:
- **Regions:** the face is split into forehead, eye band and lower face.
- **Coverage:** in each region, coverage is the share of pixels that are *not* this person's visible skin. Skin is matched by hue and by brightness close to the best-lit skin on the same face, so skin tone and lighting cancel out, and brown hair (skin-like hue, but darker) counts as a covering.
- **Eyes:** if the eye detector can't find the eyes, the eye band counts as covered.
- **Naming the covering:**
  - **mask:** a covering over the lower face;
  - **sunglasses:** a very dark band at eye level;
  - **hair:** a textured, stringy covering over the forehead and eyes;
  - **object:** a flat covering.

On the test portrait, clean faces score 0.14–0.16 (the threshold is 0.25). Hair, masks and
sunglasses are each scored as covered and named correctly at severities 1–5.

Heavy coverings can hide a face from the detector itself. `occlusion_report`, `face_occlusion`
and `anonymize` therefore use `detect_faces(relaxed=True)`, which adds looser passes. Images
where no face is found at all are still listed in the report, and they are often the most
heavily covered.

## Licence plates

```python
from mlpilot.media import plates

plates.detect_plates(img, pattern="IN")      # [Plate(box=(x, y, w, h), text='MH12AB1234', confidence=0.99)]
plates.detect_plates(img, enhance=True)      # dehaze / brighten / denoise first, original as fallback
plates.read_plate(img, box)                  # OCR one region -> (text, confidence)
plates.plate_report("traffic_cam/", pattern="IN")   # plates, text, confidence, and the conditions per image
plates.clean_plate_text("MH l2 AB I234", pattern="IN")   # 'MH12AB1234' (fixes O/0, I/1, S/5, B/8 ...)
```

`pattern` is a regex or one of `PLATE_PATTERNS`: `IN`, `IN_BH`, `UK`, `EU`, `US`. With a
pattern, OCR look-alike characters are swapped until the text matches.

How plates are found:
1. **Full-image OCR:** RapidOCR reads the whole image, and text lines that look like a plate are kept.
2. **Fallback:** if that pass reads nothing, OpenCV's plate detector and bright, plate-shaped rectangles are read one by one.

Expect about 1–2 s per image on a laptop CPU.

The tests ran on synthetic street scenes, with plates 34–52 px tall and 3 plates per row:

| Condition, severity 5 | Read correctly | With `enhance=True` |
|---|---|---|
| fog, rain, snow, low light, noise, JPEG, shadow | 3/3 | 3/3 |
| glare | 2/3 | 3/3 |
| motion blur (severity 3 / 5) | 2/3 / 0/3 | 2/3 / 0/3 |

`enhance=True` reads both the image as taken and a restored copy, and keeps the more confident
read of each plate. Restoration on its own sometimes hurts: denoising can smear snow-covered
characters, for example. Reading both costs about twice the time.

Strong motion blur is the limit. For moving traffic, use a faster shutter or a deblurring
model. These are classical detectors plus a compact OCR model. They are good for auditing data,
flagging frames and anonymising, but they don't replace a detector trained on your own cameras.
Benchmark on your own frames with `condition_benchmark`.

## Privacy: blur faces and plates

```python
out, found = faces.anonymize(img)                  # {"faces": [boxes], "plates": [boxes]}
out, _ = faces.anonymize(img, method="pixelate")
faces.blur_regions(img, [(x, y, w, h)])            # any boxes you already have
```

`anonymize` leans towards blurring too much. It uses the relaxed face passes, and it blurs
plate-shaped regions that contain text even when that text can't be read.

## Robustness benchmark

Re-score any image model on degraded copies of a test set, one condition and severity at a time:

```python
from mlpilot.media import benchmark

predict = benchmark.image_classifier(model)        # an AutoML trained on media.image_features
rep = benchmark.condition_benchmark(predict, test_images, test_labels, faces=True)
print(rep.summary())
# (illustrative output)
# Clean score: 0.942
#   low_light        worst 0.611 (drop +0.331), mean 0.802
#   fog              worst 0.705 (drop +0.237), mean 0.851
rep.table              # condition x severity -> score, drop
rep.weakest            # conditions ordered by worst-case drop
assert rep.passed(max_drop=0.10)                   # gate a release

# plates: score = exact-match accuracy of the read text
rep = benchmark.condition_benchmark(benchmark.plate_reader("IN"), car_images, plate_texts)
# faces: labels are the expected face counts
rep = benchmark.condition_benchmark(benchmark.face_counter(), portraits, [1] * len(portraits), faces=True)
```

The benchmark needs only one thing from your model: `predict(list_of_PIL_images) -> predictions`.
That covers your own PyTorch or ONNX models too. Pass `score=` for any metric other than
accuracy. To fix the weak spots, retrain on
`benchmark.condition_augment(images, labels, conditions=rep.weakest[:3])`.
