# -*- coding: utf-8 -*-

from train import get_parser
from dataset.dataset import dataset
import torch.utils.data as TD
from utils.train_tools import get_ckpt_path,get_model
from utils.disc import show_images,point_yx2xy
import torch
import numpy as np
import os
import cv2

def show_offset(imgs,img_path,points_offset,points_perturb_true,outputs):
    """
    imgs: the 128x128 image patch
    img_path: origin image, remember resize to 320,240
    points_offset: true offset for perturb
    points_perturb_true: true cordinate for 4 perturb points
    
    points_offset_predict: predict offset for perturb
    points_origin: true cordinate for 4 origin points 
    points_perturb_predict: predict cordinate for 4 perturb points
    """
    points_offset_predict=outputs.data.cpu().numpy()
    np_imgs=imgs.data.cpu().numpy()
    # convert np_imgs for [-1,1] to [0,255]
    np_imgs=128*(np_imgs+1)
    np_imgs=np_imgs.astype(np.uint8)
    
    # true cordinate for 4 origin points 
    points_origin=points_perturb_true - points_offset
    
    batch_size=np_imgs.shape[0]
    for idx in range(batch_size):
        # gray image 128x128
        img1_patch=np_imgs[idx,0,:,:]
        img2_patch=np_imgs[idx,1,:,:]
        show_images([img1_patch,img2_patch],
                    ['origin patch','homo patch'])
        
        pts_img_origin=points_origin[idx,:].reshape(4,2)
        pts_img_perturb_predict=pts_img_origin+points_offset_predict[idx,:].reshape(4,2)
        pts_img_perturb_true=points_perturb_true[idx,:].reshape(4,2)
        h_predict, status = cv2.findHomography(pts_img_origin, pts_img_perturb_predict, cv2.RANSAC)
        h_true, status = cv2.findHomography(pts_img_origin, pts_img_perturb_true, cv2.RANSAC)
        # rgb image 320x240
        img_origin=cv2.resize(cv2.imread(img_path[idx]),(320,240))
        img_homo_predict=cv2.warpPerspective(img_origin,h_predict,dsize=(320,240))
        img_homo_true=cv2.warpPerspective(img_origin,h_true,dsize=(320,240))
        
        img_origin=cv2.polylines(img_origin,
                                 point_yx2xy(pts_img_perturb_predict),
                                 isClosed=True,
                                 color=(255,0,0),
                                 thickness=5)
        img_origin=cv2.polylines(img_origin,
                                 point_yx2xy(pts_img_perturb_true),
                                 isClosed=True,
                                 color=(255,255,0),
                                 thickness=5)
        
        img_homo_predict=cv2.polylines(img_homo_predict,point_yx2xy(pts_img_origin),
                                       isClosed=True,color=(255,0,0),thickness=5)
        img_homo_true=cv2.polylines(img_homo_true,point_yx2xy(pts_img_origin),
                                       isClosed=True,color=(255,255,0),thickness=5)
        show_images([img_origin,img_homo_predict,img_homo_true],
                    ['origin image','predict homo image','true homo image'])
    
if __name__ == '__main__':
    parser=get_parser()
    args=parser.parse_args()
    log_dir=os.path.join(args.log_dir,args.model_name,args.patch_dataset,args.note)
    ckpt_path = get_ckpt_path(log_dir)
    print('load checkpoint file from', ckpt_path)
    state_dict = torch.load(ckpt_path)
    
    model=get_model(args)    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    
    if 'model_state' in state_dict.keys():
        model.load_state_dict(state_dict['model_state'])
    else:
        model.load_state_dict(state_dict)
        
    val_loader=TD.DataLoader(dataset=dataset(args,split='test'),
                               batch_size=args.batch_size,
                               shuffle=True,
                               drop_last=True,
                               num_workers=2)
    
    model.eval()
    for idx,(datas) in enumerate(val_loader):
        imgs=datas['imgs'].to(device).float()
        points_offset=datas['points_offset'].data.cpu().numpy()
        points_perturb_true=datas['points_perturb'].data.cpu().numpy()
        img_path=datas['img_path']
        
        outputs=model.forward(imgs)
        
        show_offset(imgs,img_path,points_offset,points_perturb_true,outputs)
        
        break