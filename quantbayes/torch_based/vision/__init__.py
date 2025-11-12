"""Public façade – keep import surface tiny."""

from .pipeline.segmentation_pipeline import SegmentationPipeline
from .pipeline.classification_pipeline import ClassificationPipeline

__all__ = ["SegmentationPipeline", "ClassificationPipeline"]
