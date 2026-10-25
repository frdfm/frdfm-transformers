"""Geometry only: draws predicted controls; does not read or interpret instructions."""
import math
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw
from text2image.stick_figures.data import SCALES


def joints(controls):
    torso, left_arm, right_arm, left_knee, right_knee = np.asarray(controls).clip(0, 1) * SCALES
    a = math.radians(float(torso))
    hip = np.array([128., 143.])
    shoulder = hip + 57 * np.array([math.sin(a), -math.cos(a)])
    head = shoulder + 21 * np.array([math.sin(a), -math.cos(a)])
    result = {"hip": hip, "shoulder": shoulder, "head": head}
    for label, direction, arm, knee_angle in [("left", -1, left_arm, left_knee), ("right", 1, right_arm, right_knee)]:
        # Left/right are canvas-left/canvas-right, not an anatomical camera convention.
        start = shoulder + np.array([direction * 6., 0])
        arm_angle = math.radians(float(arm))
        elbow = start + 26 * np.array([direction * math.sin(arm_angle), math.cos(arm_angle)])
        wrist = elbow + 25 * np.array([direction * math.sin(arm_angle + 0.12), math.cos(arm_angle + 0.12)])
        knee = hip + np.array([direction * 14., 35.])
        bend = math.radians(float(knee_angle))
        ankle = knee + 34 * np.array([direction * math.sin(bend), math.cos(bend)])
        result.update({label + "_shoulder": start, label + "_elbow": elbow, label + "_wrist": wrist,
                       label + "_knee": knee, label + "_ankle": ankle})
    return result


def draw_figure(controls):
    image = Image.new("RGB", (256, 240), "white")
    pen = ImageDraw.Draw(image)
    p = joints(controls)
    def line(names, color, width=5):
        pen.line([tuple(p[n]) for n in names], fill=color, width=width)
    line(["hip", "shoulder"], "black", 6)
    x, y = p["head"]
    pen.ellipse((x-12, y-12, x+12, y+12), outline="black", width=4)
    for side, color in [("left", "#1872b6"), ("right", "#d47900")]:
        line([side+"_shoulder", side+"_elbow", side+"_wrist"], color)
        line(["hip", side+"_knee", side+"_ankle"], color)
    return image


def save_comparison(rows, path):
    # Every row: caption, expected controls, predicted controls.
    grid = Image.new("RGB", (700, len(rows) * 280), "#eeeeee")
    pen = ImageDraw.Draw(grid)
    for i, (caption, expected, prediction) in enumerate(rows):
        y = i * 280
        # Wrap by clauses so instructions remain legible.
        clauses = caption.split(" ; ")
        pen.text((8, y+4), " ; ".join(clauses[:3]), fill="black")
        pen.text((8, y+18), " ; ".join(clauses[3:]), fill="black")
        grid.paste(draw_figure(expected), (40, y+38))
        grid.paste(draw_figure(prediction), (370, y+38))
        pen.text((45, y+40), "Expected", fill="black")
        pen.text((375, y+40), "Predicted", fill="black")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    grid.save(path)
