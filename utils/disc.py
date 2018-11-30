# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt

def str2bool(s):
    if s.lower() in ['t', 'true', 'y', 'yes', '1']:
        return True
    elif s.lower() in ['f', 'false', 'n', 'no', '0']:
        return False
    else:
        assert False, 'unexpected value for bool type'
        
def show_images(images,titles=None,vmin=None,vmax=None):
    fig, axes = plt.subplots(2, (len(images)+1)//2, figsize=(7, 6), sharex=True, sharey=True)
    ax = axes.ravel()

    for i in range(len(images)):
        ax[i].imshow(images[i],vmin=vmin,vmax=vmax)
        if titles is None:
            ax[i].set_title("image %d"%i)
        else:
            ax[i].set_title(titles[i])

    plt.show()
    
def point_yx2xy(yx_pts):
    """
    convert [[y1,x1],[y2,x2],...] to pts for cv2.polylines
    """
    xy_pts=yx_pts.copy()
    xy_pts[:,0]=yx_pts[:,1]
    xy_pts[:,1]=yx_pts[:,0]
    points=[xy_pts.reshape((-1,1,2)).astype(np.int32)]
    return points