"""Offline-only connected skeleton rasterizer; never used by neural inference.

Draw at 256px then area-filter to 48px. Minimum stroke width preserves joints.
"""
import math
import numpy as np
from PIL import Image, ImageDraw

def raster(combo, rng, canonical=False, side=48):
    """Randomize anatomy/style while preserving discrete pose states and left/right."""
    from text2image.stick_figures.data import ANGLES
    def uniform(low, high):
        return (low + high) / 2 if canonical else rng.uniform(low, high)
    angles = np.array([ANGLES[i][state] for i,state in enumerate(combo)], dtype=float)
    if not canonical:
        # At most three degrees around the requested prototype; states stay separated.
        angles += rng.uniform(-3, 3, 5)
        angles = np.maximum(angles, 0)
    torso, left, right, left_knee, right_knee = np.deg2rad(angles)
    hip = np.array([0., 0.])
    torso_length = uniform(46, 66)
    direction = np.array([math.sin(torso), -math.cos(torso)])
    shoulder = hip + torso_length * direction
    radius = uniform(9, 15)
    neck = uniform(5, 11)
    head = shoulder + (neck + radius) * direction
    shoulder_half = uniform(5, 11)
    hip_half = uniform(9, 17)
    upper_arm, lower_arm = uniform(22, 32), uniform(21, 31)
    upper_leg, lower_leg = uniform(28, 42), uniform(26, 40)
    elbow_bend = uniform(0.02, 0.18)
    points = {"hip": hip, "shoulder": shoulder, "head": head}
    for label, sign, arm, knee_angle in [("left",-1,left,left_knee),("right",1,right,right_knee)]:
        start = shoulder + np.array([sign*shoulder_half, 0])
        elbow = start + upper_arm*np.array([sign*math.sin(arm), math.cos(arm)])
        wrist = elbow + lower_arm*np.array([sign*math.sin(arm+elbow_bend),math.cos(arm+elbow_bend)])
        knee = hip + np.array([sign*hip_half, upper_leg])
        ankle = knee + lower_leg*np.array([sign*math.sin(knee_angle),math.cos(knee_angle)])
        points.update({label+"_shoulder":start,label+"_elbow":elbow,label+"_wrist":wrist,
                       label+"_knee":knee,label+"_ankle":ankle})
    # Normalize the full drawing bounds, including the head, so no limb is cropped.
    all_points=np.stack([*points.values(),head+radius,head-radius])
    lower,upper=all_points.min(0),all_points.max(0)
    scale=uniform(0.84,1.04)*min(202/(upper[0]-lower[0]),208/(upper[1]-lower[1]))
    center=(lower+upper)/2
    offset=np.array([128+uniform(-9,9),128+uniform(-9,9)])
    points={name:(xy-center)*scale+offset for name,xy in points.items()}
    radius*=scale
    width=max(int(round(256/side*1.6)),int(round(uniform(3.5,5.0)*scale)))
    image=Image.new("L",(256,256),0)
    pen=ImageDraw.Draw(image)
    def line(names):
        xy=[tuple(points[name]) for name in names]
        pen.line(xy,fill=255,width=width)
        for x,y in xy:
            r=width/2
            pen.ellipse((x-r,y-r,x+r,y+r),fill=255)
    line(["hip","shoulder"])
    line(["shoulder","head"])
    # Clear the center of the head so it remains an outlined circle.
    x,y=points["head"]
    pen.ellipse((x-radius,y-radius,x+radius,y+radius),fill=0,outline=255,width=width)
    for limb in ("left","right"):
        line(["shoulder",limb+"_shoulder",limb+"_elbow",limb+"_wrist"])
        line(["hip",limb+"_knee",limb+"_ankle"])
    pixels=(np.asarray(image.resize((side,side),Image.Resampling.LANCZOS))>=80).astype(np.uint8)
    return pixels.reshape(-1)



def is_connected(pixels, side=48):
    """Eight-connected ink; diagonal lines count as continuous strokes."""
    ink=set(map(tuple,np.argwhere(np.asarray(pixels).reshape(side,side))))
    if not ink:return False
    pending=[ink.pop()]
    while pending:
        y,x=pending.pop()
        for dy,dx in ((-1,-1),(-1,0),(-1,1),(0,-1),(0,1),(1,-1),(1,0),(1,1)):
            point=(y+dy,x+dx)
            if point in ink:
                ink.remove(point);pending.append(point)
    return not ink
