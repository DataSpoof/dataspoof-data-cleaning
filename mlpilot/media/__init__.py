"""Image and audio data: quality audits, consistent preprocessing, and features for AutoML.

* images: ``scan_images``, ``image_quality_report``, ``ImagePreprocessor``, ``image_features``,
  ``augment`` (needs Pillow: ``pip install "dataspoof-mlpilot[image]"``)
* conditions: detect / simulate / remove fog, rain, snow, low light, glare, blur, noise, JPEG
  (``detect_conditions``, ``condition_report``, ``apply_condition``, ``enhance``)
* faces: detection, coverage by hair / mask / sunglasses, simulated coverings, blurring
  (``detect_faces``, ``face_occlusion``, ``occlusion_report``, ``add_occluder``, ``anonymize``;
  needs ``[vision]``)
* plates: licence-plate detection and reading (``detect_plates``, ``read_plate``,
  ``plate_report``; needs ``[ocr]``)
* benchmark: re-score any image model under every condition and severity
  (``condition_benchmark``, ``condition_augment``)
* audio:  ``scan_audio``, ``audio_quality_report``, ``AudioPreprocessor``, ``audio_features``,
  ``load_audio`` (WAV built in; other formats need ``[audio]``)

The feature extractors return DataFrames, so ``AutoML(target).fit(features.join(labels, on="path"))``
trains image / audio classifiers with the same tooling as tabular data.
"""

from . import benchmark, conditions, faces, plates
from .audio import (AudioPreprocessor, audio_features, audio_quality_report, load_audio, mel_filterbank, read_audio,
                    scan_audio, spectral_features)
from .benchmark import BenchmarkReport, condition_augment, condition_benchmark
from .conditions import (CONDITIONS, ConditionResult, apply_condition, condition_report, condition_scores,
                         detect_conditions, enhance)
from .faces import FaceBox, OcclusionResult, add_occluder, anonymize, blur_regions, detect_faces, face_occlusion, \
    occlusion_report
from .image import (ImagePreprocessor, MediaIssue, MediaQualityReport, augment, dhash, image_features,
                    image_quality_report, open_image, scan_images)
from .plates import Plate, clean_plate_text, detect_plates, plate_report, read_plate

__all__ = ["ImagePreprocessor", "scan_images", "image_quality_report", "image_features", "augment", "open_image",
           "dhash", "AudioPreprocessor", "scan_audio", "audio_quality_report", "audio_features", "load_audio",
           "read_audio", "spectral_features", "mel_filterbank", "MediaIssue", "MediaQualityReport",
           # conditions
           "conditions", "CONDITIONS", "ConditionResult", "condition_scores", "detect_conditions",
           "condition_report", "apply_condition", "enhance",
           # faces
           "faces", "FaceBox", "OcclusionResult", "detect_faces", "face_occlusion", "occlusion_report",
           "add_occluder", "blur_regions", "anonymize",
           # plates
           "plates", "Plate", "detect_plates", "read_plate", "plate_report", "clean_plate_text",
           # benchmark
           "benchmark", "BenchmarkReport", "condition_benchmark", "condition_augment"]
