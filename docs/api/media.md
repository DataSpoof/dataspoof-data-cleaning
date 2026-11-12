# Image & audio data: API reference

Generated from the source docstrings. For a task-oriented introduction, read the [Image & audio data guide](../guide/media.md).

**On this page:**

- [`mlpilot.media`](#mlpilotmedia)
- [`mlpilot.media.audio`](#mlpilotmediaaudio): `read_audio`, `resample`, `load_audio`, `scan_audio`, `audio_quality_report`, `AudioPreprocessor`, `mel_filterbank`, `spectral_features`, `audio_features`
- [`mlpilot.media.image`](#mlpilotmediaimage): `open_image`, `dhash`, `sharpness`, `scan_images`, `MediaIssue`, `MediaQualityReport`, `image_quality_report`, `ImagePreprocessor`, `image_features`, `augment`

---

## `mlpilot.media`

<sub>Source: [`mlpilot/media/__init__.py`](https://github.com/DataSpoof/dataspoof-mlpilot/blob/main/mlpilot/media/__init__.py)</sub>

Image and audio data: quality audits, consistent preprocessing, and features for AutoML.

* images: `scan_images`, `image_quality_report`, `ImagePreprocessor`, `image_features`,
  `augment` (needs Pillow: `pip install "dataspoof-mlpilot[image]"`)
* conditions: detect / simulate / remove fog, rain, snow, low light, glare, blur, noise, JPEG
  (`detect_conditions`, `condition_report`, `apply_condition`, `enhance`)
* faces: detection, coverage by hair / mask / sunglasses, simulated coverings, blurring
  (`detect_faces`, `face_occlusion`, `occlusion_report`, `add_occluder`, `anonymize`;
  needs `[vision]`)
* plates: licence-plate detection and reading (`detect_plates`, `read_plate`,
  `plate_report`; needs `[ocr]`)
* benchmark: re-score any image model under every condition and severity
  (`condition_benchmark`, `condition_augment`)
* audio:  `scan_audio`, `audio_quality_report`, `AudioPreprocessor`, `audio_features`,
  `load_audio` (WAV built in; other formats need `[audio]`)

The feature extractors return DataFrames, so `AutoML(target).fit(features.join(labels, on="path"))`
trains image / audio classifiers with the same tooling as tabular data.

---

## `mlpilot.media.audio`

<sub>Source: [`mlpilot/media/audio.py`](https://github.com/DataSpoof/dataspoof-mlpilot/blob/main/mlpilot/media/audio.py)</sub>

Audio data: audit recordings, preprocess them consistently, and extract features for AutoML.

WAV works out of the box (scipy); FLAC / OGG / MP3 … need `pip install "dataspoof-mlpilot[audio]"`
(soundfile). Everything else — resampling, silence trimming, loudness normalisation, MFCCs and
spectral features — is implemented with numpy/scipy (no librosa needed).

* `load_audio(path, sr=16000)`            mono float32 in [-1, 1], resampled
* `scan_audio("calls/**/*.wav")`          duration, sample rate, channels, loudness, clipping,
  silence, DC offset; unreadable files kept with `error`
* `audio_quality_report(scan)`            corrupt, clipped, silent / mostly silent, too short,
  mixed sample rates, DC offset, duplicates
* `AudioPreprocessor(sr=16000, duration=3)`  mono, resample, trim silence, normalise loudness,
  pre-emphasis, pad/trim to a fixed length; JSON config
* `audio_features(paths)`                 MFCC / delta statistics, spectral centroid, bandwidth,
  roll-off, flatness, zero-crossing rate, RMS -> DataFrame

### `read_audio()`

```python
def read_audio(path)
```

`(samples float32 (n, channels) in [-1, 1], sample_rate)` without any processing.

### `resample()`

```python
def resample(signal: np.ndarray, sr: int, target: int) -> np.ndarray
```

### `load_audio()`

```python
def load_audio(path, sr: int | None=16000, mono: bool=True)
```

Read, mix to mono and resample. Returns `(signal, sample_rate)`.

### `scan_audio()`

```python
def scan_audio(source, *, labels: dict | None=None, label_from_folder: bool=False, silence_db: float=-50.0) -> pl.DataFrame
```

### `audio_quality_report()`

```python
def audio_quality_report(scan: pl.DataFrame, *, min_duration: float=0.5, clip_ratio: float=0.001, max_silence: float=0.8, dc_limit: float=0.05) -> MediaQualityReport
```

### class `AudioPreprocessor`

```python
class AudioPreprocessor:
```

Mono -> resample -> trim leading/trailing silence -> loudness normalise -> pre-emphasis ->
pad/trim to `duration` seconds. Stateless apart from its settings (saved as JSON).

**Constructor**

```python
AudioPreprocessor(sr: int=16000, *, duration: float | None=None, trim_db: float | None=-40.0, normalize: str | None='rms', target_dbfs: float=-20.0, pre_emphasis: float | None=None)
```

#### `AudioPreprocessor.process()`

```python
process(signal: np.ndarray, sr: int) -> np.ndarray
```

#### `AudioPreprocessor.transform()`

```python
transform(paths) -> list | np.ndarray
```

Processed signals: an `(n, samples)` array if `duration` is set, else a list.

#### `AudioPreprocessor.to_dict()`

```python
to_dict() -> dict
```

#### `AudioPreprocessor.save()`

```python
save(path) -> Path
```

#### `AudioPreprocessor.load()` *(classmethod)*

```python
load(path) -> 'AudioPreprocessor'
```

### `mel_filterbank()`

```python
def mel_filterbank(sr: int, n_fft: int, n_mels: int=40, fmin: float=0.0, fmax: float | None=None) -> np.ndarray
```

### `spectral_features()`

```python
def spectral_features(x: np.ndarray, sr: int, *, n_fft: int=512, hop: int=160, n_mels: int=40, n_mfcc: int=13) -> dict
```

Frame-level features summarised by mean and std over time.

### `audio_features()`

```python
def audio_features(source, *, preprocessor: AudioPreprocessor | None=None, **kw) -> pl.DataFrame
```

One row of features per file (`path` + numeric columns) for `AutoML`.

---

## `mlpilot.media.image`

<sub>Source: [`mlpilot/media/image.py`](https://github.com/DataSpoof/dataspoof-mlpilot/blob/main/mlpilot/media/image.py)</sub>

Image data: audit a folder of images, preprocess them consistently, and turn them into
features the tabular AutoML can learn from (no deep-learning dependency).

Needs Pillow: `pip install "dataspoof-mlpilot[image]"`.

* `scan_images("photos/**/*.jpg")`       one row per file: size, mode, format, hashes, brightness,
  sharpness, readable or not
* `image_quality_report(scan)`           corrupt files, exact & near duplicates, tiny / dark /
  over-exposed / blurry images, odd aspect ratios, mixed
  colour modes, label imbalance
* `ImagePreprocessor(size=(224, 224))`   EXIF-orientation fix, colour mode, resize (pad / crop /
  stretch), per-channel normalisation learned in `fit`;
  JSON-serialisable like every pipeline step
* `image_features(paths)`                colour histograms, edge-orientation histograms (HOG-like),
  texture and shape statistics -> a DataFrame for AutoML
* `augment(image, seed=…)`               reproducible flips, rotations, crops, brightness/contrast

### `open_image()`

```python
def open_image(path, mode: str | None='RGB')
```

Open an image with its EXIF orientation applied (phone photos are often stored rotated).

### `dhash()`

```python
def dhash(image, size: int=8) -> str
```

Perceptual difference hash: near-identical images (resized, re-compressed) share most bits.

### `sharpness()`

```python
def sharpness(gray: np.ndarray) -> float
```

Variance of the Laplacian: low values mean a blurry image.

### `scan_images()`

```python
def scan_images(source, *, labels: dict | None=None, label_from_folder: bool=False) -> pl.DataFrame
```

Inspect every image (files that can't be read are kept, with `error` set).

`label_from_folder=True` takes each image's label from its parent folder name
(the usual `train/cat/001.jpg` layout); or pass `labels={path: label}`.

### class `MediaIssue`

```python
class MediaIssue:
```

| Field | Type | Default |
|---|---|---|
| `check` | `str` | required |
| `severity` | `str` | required |
| `message` | `str` | required |
| `files` | `list` | required |

### class `MediaQualityReport`

```python
class MediaQualityReport:
```

| Field | Type | Default |
|---|---|---|
| `kind` | `str` | required |
| `n_files` | `int` | required |
| `issues` | `list` | required |
| `summary_stats` | `dict` | required |

#### `MediaQualityReport.problem_files` *(property)*

```python
problem_files: list
```

#### `MediaQualityReport.summary()`

```python
summary() -> str
```

#### `MediaQualityReport.to_dict()`

```python
to_dict() -> dict
```

### `image_quality_report()`

```python
def image_quality_report(scan: pl.DataFrame, *, min_side: int=32, dark: float=20, bright: float=235, blur_ratio: float=0.1, near_duplicate_bits: int=4, max_aspect: float=4.0) -> MediaQualityReport
```

Problems that hurt image models, from a `scan_images` result.

### class `ImagePreprocessor`

```python
class ImagePreprocessor:
```

Consistent image preparation, learned on training images and replayed on new ones.

`resize`: `"pad"` (keep aspect, letterbox), `"crop"` (keep aspect, centre crop) or
`"stretch"`. `normalize=True` learns per-channel mean/std in `fit`.

**Constructor**

```python
ImagePreprocessor(size=(224, 224), *, mode: str='RGB', resize: str='pad', normalize: bool=True, pad_color=(0, 0, 0))
```

#### `ImagePreprocessor.fit()`

```python
fit(paths) -> 'ImagePreprocessor'
```

#### `ImagePreprocessor.transform()`

```python
transform(paths) -> np.ndarray
```

A float32 array `(n, height, width, channels)`.

#### `ImagePreprocessor.fit_transform()`

```python
fit_transform(paths) -> np.ndarray
```

#### `ImagePreprocessor.to_dict()`

```python
to_dict() -> dict
```

#### `ImagePreprocessor.save()`

```python
save(path) -> Path
```

#### `ImagePreprocessor.load()` *(classmethod)*

```python
load(path) -> 'ImagePreprocessor'
```

### `image_features()`

```python
def image_features(source, *, size: int=64, color_bins: int=8, hog_cells: int=4) -> pl.DataFrame
```

Hand-crafted features per image (`path` + numeric columns) so `AutoML` can classify
images without a GPU: per-channel colour histograms and moments, HSV saturation, edge
density, an edge-orientation (HOG-like) grid, sharpness, brightness and aspect ratio.

`source`: a folder, glob or list of paths, or in-memory PIL images (named `<image i>`).

### `augment()`

```python
def augment(image, *, seed: int=0, flip: bool=True, max_rotate: float=15.0, crop: float=0.1, brightness: float=0.2, contrast: float=0.2)
```

One reproducible random augmentation of a PIL image (same seed -> same result).
