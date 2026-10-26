"""Fixed output pixel colors. Palette IDs are actual image pixels, not pose controls."""
import numpy as np
NAMES=('background','head','torso','left arm','right arm','left leg','right leg')
RGB=np.array([(255,255,255),(230,180,0),(135,65,190),(220,45,45),(35,100,220),(30,155,75),(240,110,20)],dtype=np.uint8)
COLORS=len(NAMES)
