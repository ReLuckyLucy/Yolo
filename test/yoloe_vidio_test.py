import numpy as np

from ultralytics import YOLOE
from ultralytics.models.yolo.yoloe import YOLOEVPSegPredictor

# Initialize a YOLOE model
model = YOLOE("yoloe-11l-seg.pt")

# Set visual prompt
visuals = dict(
    bboxes=np.array(
        [
            [221.52, 405.8, 344.98, 857.54],  # For person
            [120, 425, 160, 445],  # For glasses
        ],
    ),
    cls=np.array(
        [
            0,  # For person
            1,  # For glasses
        ]
    ),
)

# Execute prediction for specified categories on an image
results = model.predict(
    "img/test.mp4",
    visual_prompts=visuals,
    predictor=YOLOEVPSegPredictor,
)

# Show results
results[0].show()