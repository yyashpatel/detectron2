from detectron2.utils.visualizer import Visualizer, VisImage, _create_text_labels, GenericMask
from detectron2.utils.colormap import random_color
from enum import Enum, unique
import numpy as np
import math
import numpy as np
import cv2
import matplotlib as mpl
import matplotlib.colors as mplc
import matplotlib.figure as mplfigure
import torch
from matplotlib.backends.backend_agg import FigureCanvasAgg
from PIL import Image


_KEYPOINT_THRESHOLD = 0.05
_SMALL_OBJECT_AREA_THRESH = 1000

class ColorMode(Enum):
    """
    Enum of different color modes to use for instance visualizations.
    """

    IMAGE = 0
    """
    Picks a random color for every instance and overlay segmentations with low opacity.
    """
    SEGMENTATION = 1
    """
    Let instances of the same category have similar colors
    (from metadata.thing_colors), and overlay them with
    high opacity. This provides more attention on the quality of segmentation.
    """
    IMAGE_BW = 2
    """
    Same as IMAGE, but convert all areas without masks to gray-scale.
    Only available for drawing per-instance mask predictions.
    """

class myVisualizer(Visualizer):

    def __init__(self, img_rgb, metadata=None, scale=1.0, instance_mode=ColorMode.IMAGE):
        super().__init__(img_rgb, metadata=None, scale=1.0, instance_mode=ColorMode.IMAGE)
        """
        Args:
            img_rgb: a numpy array of shape (H, W, C), where H and W correspond to
                the height and width of the image respectively. C is the number of
                color channels. The image is required to be in RGB format since that
                is a requirement of the Matplotlib library. The image is also expected
                to be in the range [0, 255].
            metadata (Metadata): dataset metadata (e.g. class names and colors)
            instance_mode (ColorMode): defines one of the pre-defined style for drawing
                instances on an image.
        """
        self.img = np.asarray(img_rgb).clip(0, 255).astype(np.uint8)
        if metadata is None:
            metadata = MetadataCatalog.get("__nonexist__")
        self.metadata = metadata
        self.output = VisImage(self.img, scale=scale)
        self.cpu_device = torch.device("cpu")

        # too small texts are useless, therefore clamp to 9
        self._default_font_size = max(
            np.sqrt(self.output.height * self.output.width) // 90, 10 // scale
        )
        self._instance_mode = instance_mode
        self.keypoint_threshold = _KEYPOINT_THRESHOLD

    def __call__(self, predictions, score_threshold):

        boxes = predictions.pred_boxes if predictions.has("pred_boxes") else None
        scores = predictions.scores if predictions.has("scores") else None
        score_ind = torch.where(scores>score_threshold)

        classes = predictions.pred_classes if predictions.has("pred_classes") else None
        classes = classes[score_ind]
        classes = classes.tolist()

        labels = _create_text_labels(classes, scores, self.metadata.get("thing_classes", None))
        keypoints = predictions.pred_keypoints if predictions.has("pred_keypoints") else None
        boxes = boxes[score_ind]

        if predictions.has("pred_masks"):
            masks = np.asarray(predictions.pred_masks)
            masks = masks[score_ind]
            masks = [GenericMask(x, self.output.height, self.output.width) for x in masks]
        else:
            masks = None

        if self._instance_mode == ColorMode.SEGMENTATION and self.metadata.get("thing_colors"):
            colors = [
                self._jitter([x / 255 for x in self.metadata.thing_colors[c]]) for c in classes
            ]
            alpha = 0.8
        else:
            colors = None
            alpha = 0.5

        if self._instance_mode == ColorMode.IMAGE_BW:
            self.output.Visualizer.reset_image(
                Visualizer._create_grayscale_image(
                    (predictions.pred_masks.any(dim=0) > 0).numpy()
                    if predictions.has("pred_masks")
                    else None
                )
            )
            alpha = 0.3

        self.overlay_instances(
            boxes=boxes,
            labels=labels,
            masks=masks,
            keypoints=keypoints,
            assigned_colors=colors,
            alpha=alpha,
        )
        return self.output