# Pose keypoints for one detected person.
# keypoints are in COCO-17 order (nose, eyes, ears, shoulders, ... ankles).
float64[4] bbox        # person detection box [x1, y1, x2, y2] in pixels
float64    bbox_score  # person detection confidence [0, 1]
Keypoint2D[] keypoints # length 17 for COCO
