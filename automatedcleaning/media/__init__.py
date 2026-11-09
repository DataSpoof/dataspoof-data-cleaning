"""Image and audio data: quality audits, consistent preprocessing, and features for AutoML.

* images: ``scan_images``, ``image_quality_report``, ``ImagePreprocessor``, ``image_features``,
  ``augment`` (needs Pillow: ``pip install "AutomatedCleaning[image]"``)
* audio:  ``scan_audio``, ``audio_quality_report``, ``AudioPreprocessor``, ``audio_features``,
  ``load_audio`` (WAV built in; other formats need ``[audio]``)

The feature extractors return DataFrames, so ``AutoML(target).fit(features.join(labels, on="path"))``
trains image / audio classifiers with the same tooling as tabular data.
"""

from .audio import (AudioPreprocessor, audio_features, audio_quality_report, load_audio, mel_filterbank, read_audio,
                    scan_audio, spectral_features)
from .image import (ImagePreprocessor, MediaIssue, MediaQualityReport, augment, dhash, image_features,
                    image_quality_report, open_image, scan_images)

__all__ = ["ImagePreprocessor", "scan_images", "image_quality_report", "image_features", "augment", "open_image",
           "dhash", "AudioPreprocessor", "scan_audio", "audio_quality_report", "audio_features", "load_audio",
           "read_audio", "spectral_features", "mel_filterbank", "MediaIssue", "MediaQualityReport"]
