"""Read-only pixel diagnostics. These never modify generated images."""
import numpy as np
from text2image.stick_colors.palette import NAMES
from text2image.stick_colors.tokenization import SIDE


def component_sizes(mask):
    remaining=set(map(tuple,np.argwhere(mask)))
    sizes=[]
    while remaining:
        pending=[remaining.pop()];size=0
        while pending:
            y,x=pending.pop();size+=1
            for dy,dx in ((-1,-1),(-1,0),(-1,1),(0,-1),(0,1),(1,-1),(1,0),(1,1)):
                point=(y+dy,x+dx)
                if point in remaining:remaining.remove(point);pending.append(point)
        sizes.append(size)
    return sizes


def near(a,b,tolerance=2):
    aa=np.argwhere(a);bb=np.argwhere(b)
    if not len(aa) or not len(bb):return False
    return bool((np.abs(aa[:,None,:]-bb[None,:,:]).max(-1)<=tolerance).any())


def diagnose(pixels,reference=None):
    images=np.asarray(pixels).reshape(-1,SIDE,SIDE)
    targets=None if reference is None else np.asarray(reference).reshape(-1,SIDE,SIDE)
    parts={}
    for color,name in enumerate(NAMES[1:],1):
        present=[];largest=[];contacts=[];direct=[];ious=[]
        for i,image in enumerate(images):
            mask=image==color;sizes=component_sizes(mask)
            present.append(bool(sizes));largest.append(max(sizes,default=0)/max(1,int(mask.sum())))
            if color!=2:
                contacts.append(near(mask,image==2,2));direct.append(near(mask,image==2,1))
            if targets is not None:
                other=targets[i]==color
                ious.append(float((mask&other).sum()/max(1,(mask|other).sum())))
        result=dict(present_fraction=float(np.mean(present)),mean_largest_component_fraction=float(np.mean(largest)))
        if color!=2:
            result.update(torso_contact_within_2px_fraction=float(np.mean(contacts)),direct_torso_contact_fraction=float(np.mean(direct)))
        if targets is not None:result['mean_color_iou']=float(np.mean(ious))
        parts[name]=result
    all_present=[all((image==c).any() for c in range(1,7)) for image in images]
    connected=[len(component_sizes(image!=0))==1 for image in images]
    result=dict(count=len(images),all_six_colors_present_fraction=float(np.mean(all_present)),
                connected_foreground_fraction=float(np.mean(connected)),parts=parts)
    if targets is not None:result['mean_foreground_color_iou']=float(np.mean([d['mean_color_iou'] for d in parts.values()]))
    return result
