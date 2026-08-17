import os
import math
import cv2
import numpy as np
from ultralytics import YOLO
from pathlib import Path

# ─── CONFIG ───────────────────────────────────────────────
MODEL_PATH  = "oriented_bounding_box.pt"
IMAGES_DIR  = "test_images"
OUTPUT_DIR  = "results_angles"
CONF_THRESH = 0.25
IOU_THRESH  = 0.45
IMGSZ       = 640

# Colors (BGR)
COLOR_BOX       = (0, 255, 0)     # Green  — bounding box
COLOR_HORIZ_REF = (255, 100, 0)   # Blue   — horizontal reference line
COLOR_VERT_REF  = (0, 100, 255)   # Orange — vertical reference line
COLOR_AXIS      = (0, 220, 255)   # Yellow — box major axis
COLOR_ARC_H     = (255, 200, 0)   # Cyan arc for horizontal angle
COLOR_ARC_V     = (0, 200, 255)   # Orange arc for v jiy7587ij6yy6tttththhthusha deshgan kaushalyaa deshan kajuushal555555555ertical angle
COLOR_LABEL     = (255, 255, 255) # White text
# ──────────────────────────────────────────────────────────


def draw_angle_annotation(img, center, angle_rad, ref_len):
    """
    Draw:
      - Horizontal reference ray (→) in blue
      - Vertical reference ray (↓) in orange
      - Box major-axis ray in yellow
      - Arc + label for horizontal angle (axis vs horizontal)
      - Arc + label for vertical angle (axis vs vertical)
    """
    cx, cy = center

    # Normalise angle to [-90°, 90°] so the "tilt" is always in that range
    angle_deg = math.degrees(angle_rad)
    # Keep within (-90, 90]
    angle_deg = ((angle_deg + 90) % 180) - 90

    horiz_angle = abs(angle_deg)          # angle from horizontal axis
    vert_angle  = 90.0 - horiz_angle      # angle from vertical axis

    # ── Reference rays ────────────────────────────────────
    ray = int(ref_len * 0.7)
    # Horizontal ray (rightward)
    cv2.arrowedLine(img, (cx, cy), (cx + ray, cy),
                    COLOR_HORIZ_REF, 1, cv2.LINE_AA, tipLength=0.12)
    # Vertical ray (downward, image coords)
    cv2.arrowedLine(img, (cx, cy), (cx, cy + ray),
                    COLOR_VERT_REF, 1, cv2.LINE_AA, tipLength=0.12)

    # ── Box major-axis ray ────────────────────────────────
    rad = math.radians(angle_deg)          # use normalised angle
    axis_ex = int(cx + ray * math.cos(rad))
    axis_ey = int(cy + ray * math.sin(rad))
    cv2.arrowedLine(img, (cx, cy), (axis_ex, axis_ey),
                    COLOR_AXIS, 2, cv2.LINE_AA, tipLength=0.12)

    # ── Arcs ──────────────────────────────────────────────
    arc_r = int(ray * 0.45)

    # Horizontal arc: from 0° to angle_deg (image-coordinate convention)
    a_start = min(0.0, angle_deg)
    a_end   = max(0.0, angle_deg)
    cv2.ellipse(img, (cx, cy), (arc_r, arc_r), 0,
                a_start, a_end, COLOR_ARC_H, 1, cv2.LINE_AA)

    # Vertical arc: from 90° toward angle_deg
    v_start = min(90.0, 90.0 + angle_deg if angle_deg < 0 else angle_deg)
    v_end   = max(angle_deg, 90.0)
    cv2.ellipse(img, (cx, cy), (arc_r, arc_r), 0,
                v_start, v_end, COLOR_ARC_V, 1, cv2.LINE_AA)

    # ── Angle labels ──────────────────────────────────────
    # Place horizontal label along the arc bisector (~half of horiz_angle)
    mid_h_rad = math.radians(angle_deg / 2.0)
    lx_h = cx + int((arc_r + 10) * math.cos(mid_h_rad))
    ly_h = cy + int((arc_r + 10) * math.sin(mid_h_rad))
    cv2.putText(img, f"H:{horiz_angle:.1f}",
                (lx_h - 5, ly_h),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, COLOR_ARC_H, 1, cv2.LINE_AA)

    # Place vertical label along the arc bisector (~midway between axis and vertical)
    mid_v_deg = (angle_deg + 90.0) / 2.0
    mid_v_rad = math.radians(mid_v_deg)
    lx_v = cx + int((arc_r + 10) * math.cos(mid_v_rad))
    ly_v = cy + int((arc_r + 10) * math.sin(mid_v_rad))
    cv2.putText(img, f"V:{vert_angle:.1f}",
                (lx_v - 5, ly_v),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, COLOR_ARC_V, 1, cv2.LINE_AA)

    return horiz_angle, vert_angle


def process_image(img_path: Path, model: YOLO, output_dir: Path):
    img = cv2.imread(str(img_path))
    if img is None:
        print(f"  Could not read {img_path.name}, skipping.")
        return

    results = model.predict(
        source=str(img_path),
        conf=CONF_THRESH,
        iou=IOU_THRESH,
        imgsz=IMGSZ,
        verbose=False,
    )

    result = results[0]
    boxes  = result.obb

    if boxes is None or len(boxes) == 0:
        print(f"  [{img_path.name}] — no detections")
        cv2.imwrite(str(output_dir / img_path.name), img)
        return

    print(f"  [{img_path.name}] — {len(boxes)} detection(s)")

    for i, box in enumerate(boxes):
        cls_id   = int(box.cls.item())
        cls_name = model.names[cls_id]
        conf     = float(box.conf.item())

        # xywhr: [cx, cy, w, h, angle_radians]
        xywhr   = box.xywhr[0].cpu().numpy()
        cx, cy, w, h, angle_rad = float(xywhr[0]), float(xywhr[1]), \
                                   float(xywhr[2]), float(xywhr[3]), float(xywhr[4])
        center  = (int(cx), int(cy))
        ref_len = int(max(w, h) * 0.65)

        # 4 corners of the OBB
        corners = box.xyxyxyxy[0].cpu().numpy().reshape(4, 2).astype(np.int32)

        # Draw oriented bounding box
        cv2.polylines(img, [corners], isClosed=True,
                      color=COLOR_BOX, thickness=2, lineType=cv2.LINE_AA)

        # Draw center dot
        cv2.circle(img, center, 4, (0, 0, 255), -1, cv2.LINE_AA)

        # Draw angle annotation
        h_ang, v_ang = draw_angle_annotation(img, center, angle_rad, ref_len)

        # Class + confidence label above the box
        tx = int(corners[:, 0].min())
        ty = int(corners[:, 1].min()) - 8
        if ty < 15:
            ty = int(corners[:, 1].max()) + 18

        label = f"{cls_name} {conf:.2f}"
        (lw, lh), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
        cv2.rectangle(img, (tx - 2, ty - lh - 4), (tx + lw + 2, ty + 2),
                      (0, 0, 0), -1)
        cv2.putText(img, label, (tx, ty),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, COLOR_LABEL, 2, cv2.LINE_AA)

        print(f"    [{i+1}] {cls_name:<15} conf:{conf:.2f}  "
              f"H-angle:{h_ang:6.1f}°  V-angle:{v_ang:6.1f}°")

    cv2.imwrite(str(output_dir / img_path.name), img)


def main():
    output_dir = Path(OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)

    model = YOLO(MODEL_PATH)

    exts = ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.webp")
    image_files = [p for ext in exts for p in Path(IMAGES_DIR).glob(ext)]

    if not image_files:
        print(f"No images found in '{IMAGES_DIR}/'")
        return

    print(f"\nProcessing {len(image_files)} image(s)...\n")
    for img_path in sorted(image_files):
        process_image(img_path, model, output_dir)

    print(f"\nDone! Annotated images saved to '{OUTPUT_DIR}/'")


if __name__ == "__main__":
    main()
