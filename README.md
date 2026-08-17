# Oriented Bounding Box Angle Measurement

Detects objects with a YOLO oriented bounding box (OBB) model and annotates each detection with its tilt angle relative to the horizontal and vertical axes.

## What it does

For each image in `test_images/`, the script:
1. Runs the YOLO OBB model (`oriented_bounding_box.pt`) to detect objects.
2. Draws the oriented bounding box around each detection.
3. Draws horizontal/vertical reference rays and the box's major axis.
4. Computes and labels the angle between the box and both axes (`H:` and `V:`).
5. Saves the annotated image to `results_angles/`.

## Requirements

- Python 3.9+
- Install dependencies:
  ```
  pip install -r requirements.txt
  ```

## Usage

Place images in `test_images/`, then run:

```
python angle_measurement.py
```

Annotated results are written to `results_angles/`.

## Configuration

Key settings at the top of `angle_measurement.py`:

| Variable      | Description                     |
|---------------|----------------------------------|
| `MODEL_PATH`  | Path to the `.pt` OBB model      |
| `IMAGES_DIR`  | Input images folder              |
| `OUTPUT_DIR`  | Output folder for annotated images |
| `CONF_THRESH` | Detection confidence threshold   |
| `IOU_THRESH`  | IoU threshold for NMS            |
| `IMGSZ`       | Inference image size             |
