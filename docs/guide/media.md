# Image & audio data


```python
from mlpilot import media

scan = media.scan_images("photos/", label_from_folder=True)      # one row per file
print(media.image_quality_report(scan))   # corrupt, exact & near duplicates, tiny, dark, blurry, mixed modes, imbalance
X = media.ImagePreprocessor((224, 224), resize="pad").fit_transform(scan["path"])   # (n, 224, 224, 3), normalised
feats = media.image_features(scan["path"])                        # colour, edge-orientation, texture features
model = ac.AutoML("label").fit(feats.join(scan.select("path", "label"), on="path").drop("path"))

audio = media.scan_audio("calls/**/*.wav")
print(media.audio_quality_report(audio))  # corrupt, silent, clipped, too short, DC offset, mixed sample rates
clips = media.AudioPreprocessor(16000, duration=3.0).transform(audio["path"])   # mono, resampled, trimmed, levelled
feats = media.audio_features(audio["path"])                       # MFCCs + spectral features for AutoML
```

**Images** (`pip install "mlpilot[image]"`):
- Phone-photo rotation (EXIF) is fixed automatically.
- Near-duplicates are found by perceptual hashing, and blur is measured by the variance of the
  Laplacian.
- Preprocessing settings save as JSON, and augmentation is reproducible.
- In testing, features from synthetic image classes reached ≥ 90% accuracy with a linear model.

**Audio:**
- WAV is built in; FLAC, OGG and MP3 need the `[audio]` extra.
- Resampling, silence trimming, loudness normalisation and MFCCs are implemented here, so
  librosa isn't needed.
- In testing, audio features separated two classes of tones with 100% accuracy.

**Robustness:** a corrupt file is reported, never fatal to a batch.
