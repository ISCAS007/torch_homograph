# -*- coding: utf-8 -*-
import os
import cv2
import random
import numpy as np
import torch.utils.data as TD
import glob

def get_imagelist(dataset_name,split='train'):
    """
    cocostuff: image
    davis: image+mask
    """
    # imglist['image']+imglist['mask']
    imglist={}
    imglist['image']=[]
    imglist['mask']=[]
    if dataset_name=='cocostuff':
        image_rootpath=os.path.join('dataset',dataset_name,'images')
        imagelist_rootpath=os.path.join('dataset',dataset_name,'imageLists')
        if split=='train':
            file='train.txt'
        else:
            file='test.txt'
        
        path=os.path.join(imagelist_rootpath,file)
        with open(path,'r') as f:
            imglist['image']=[os.path.join(image_rootpath,l.strip()+'.jpg') for l in f.readlines()]
            print('%s %s dataset image size is %d'%(dataset_name,split,len(imglist['image'])))
    elif dataset_name=='davis':
        image_rootpath=os.path.join('dataset',dataset_name,'JPEGImages','480p')
        annotation_rootpath=os.path.join('dataset',dataset_name,'Annotations','480p')
        imagelist_rootpath=os.path.join('dataset',dataset_name,'ImageSets','2017')
        if split=='train':
            file='train.txt'
        else:
            file='val.txt'
        
        path=os.path.join(imagelist_rootpath,file)
        with open(path,'r') as f:
            imgsets=[l.strip() for l in f.readlines() if len(l.strip())>0]
            for s in imgsets:
                imglist['image']+=glob.glob(os.path.join(image_rootpath,s,'*.jpg'))
                imglist['mask']+=glob.glob(os.path.join(annotation_rootpath,s,'*.png'))
                
            assert len(imglist['image'])==len(imglist['mask'])
            print('%s %s dataset image size is %d'%(dataset_name,split,len(imglist['image'])))
            print('%s %s dataset mask size is %d'%(dataset_name,split,len(imglist['mask'])))
            
    imglist['image'].sort()
    imglist['mask'].sort()
    return imglist
        
class dataset(TD.Dataset):
    def __init__(self,args,split=None):
        self.args=args
        if split is None:
            self.split=args.split
        else:
            self.split=split
        
        self.patch_size=(128,128)
        self.mask_size=tuple([int(a*0.5) for a in self.patch_size])
        
        if self.args.use_mask:
            data=get_imagelist(args.mask_dataset,self.split)
            self.imglist=data['image']
            self.masklist=data['mask']
            
        self.patchlist=get_imagelist(args.patch_dataset,self.split)['image']
        self.normer=image_normalizations(ways=args.norm_ways)
    
    def get_mask(self,mask_path,img_path):
        mask=cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        img=cv2.imread(img_path,cv2.IMREAD_GRAYSCALE)
        
        points=cv2.findNonZero(mask)
        rect=cv2.boundingRect(points)
        x1,y1,w,h=rect
        
        if h<=0 or w<=0:
            #print('mask is not valid',mask_path,img_path)
            crop_mask=mask
            crop_img=img
        else:
            crop_mask=mask[y1:y1+h,x1:x1+w]
            crop_img=img[y1:y1+h,x1:x1+w]
            
        
        resize_mask=cv2.resize(crop_mask,self.mask_size,interpolation=cv2.INTER_NEAREST)
        resize_img=cv2.resize(crop_img,self.mask_size,interpolation=cv2.INTER_LINEAR)
        return resize_mask,resize_img
    
    def get_patch(self,image_path):
        # use gray image as input
        img = cv2.resize(cv2.imread(image_path, 0),(320,240))
        
        y_start = random.randint(32,80)
        y_end = y_start + 128
        x_start = random.randint(32,160)
        x_end = x_start + 128

        y_1 = y_start
        x_1 = x_start
        y_2 = y_end
        x_2 = x_start
        y_3 = y_end
        x_3 = x_end
        y_4 = y_start
        x_4 = x_end

        img_patch = img[y_start:y_end, x_start:x_end]  # patch 1

        y_1_offset = random.randint(-32,32)
        x_1_offset = random.randint(-32,32)
        y_2_offset = random.randint(-32,32)
        x_2_offset = random.randint(-32,32)
        y_3_offset = random.randint(-32,32)
        x_3_offset = random.randint(-32,32)
        y_4_offset = random.randint(-32,32)
        x_4_offset = random.randint(-32,32)

        y_1_p = y_1 + y_1_offset
        x_1_p = x_1 + x_1_offset
        y_2_p = y_2 + y_2_offset
        x_2_p = x_2 + x_2_offset
        y_3_p = y_3 + y_3_offset
        x_3_p = x_3 + x_3_offset
        y_4_p = y_4 + y_4_offset
        x_4_p = x_4 + x_4_offset
        
        pts_img_patch = np.array([[y_1,x_1],[y_2,x_2],[y_3,x_3],[y_4,x_4]]).astype(np.float32)
        pts_img_patch_perturb = np.array([[y_1_p,x_1_p],[y_2_p,x_2_p],[y_3_p,x_3_p],[y_4_p,x_4_p]]).astype(np.float32)
        h,status = cv2.findHomography(pts_img_patch, pts_img_patch_perturb, cv2.RANSAC)

        img_perturb = cv2.warpPerspective(img, h, (320, 240))
        img_perturb_patch = img_perturb[y_start:y_end, x_start:x_end]  # patch 2
        
        # points_origin = points_perturb - points_offset
        points_offset = np.array([y_1_offset,x_1_offset,y_2_offset,x_2_offset,y_3_offset,x_3_offset,y_4_offset,x_4_offset])/32
        points_perturb = np.array([y_1_p,x_1_p,y_2_p,x_2_p,y_3_p,x_3_p,y_4_p,x_4_p])
        
        #norm
        img_patch=self.normer.forward(img_patch)
        img_perturb_patch=self.normer.forward(img_perturb_patch)
        return (img_patch,img_perturb_patch),(points_offset,points_perturb)
    
    def get_masked_patch(self,patch,mask,mask_img):
        """
        gpatch: gray image, defautl size=(128,128)
        mask: default size=(64,64)
        mask_img: gray image, defautl size=(64,64)
        """
        h,w=patch.shape
        mh,mw=mask.shape
        assert mask.shape==mask_img.shape
        assert h>=mh and w>=mw ,'patch size %d,%d must big than mask size %d,%d'%(h,w,mh,mw)
        x=np.random.randint(0,w-mw)
        y=np.random.randint(0,h-mh)
        
        mask_patch=patch[y:y+mh,x:x+mw]
        mask_patch[mask!=0]=mask_img[mask!=0]
        
        patch[y:y+mh,x:x+mw]=mask_patch
        
        return patch
        
    def __len__(self):
        return len(self.patchlist)
    
    def __getitem__(self,index):
        imgs,(points_offset,points_perturb)=self.get_patch(self.patchlist[index])
        
        if self.args.use_mask:
            mask_index=np.random.randint(len(self.masklist))
            mask,mask_img=self.get_mask(self.masklist[mask_index],self.imglist[mask_index])
            imgs=[self.get_masked_patch(patch,mask,mask_img) for patch in imgs]
            
            # bchw
            imgs=np.stack(imgs,axis=0)
            
            return {'imgs':imgs,
                    'mask':mask,
                    'mask_img':mask_img,
                    'points_offset':points_offset,
                    'points_perturb':points_perturb,
                    'img_path':self.patchlist[index],
                    'mask_path':self.masklist[mask_index],
                    'mask_img_path':self.imglist[mask_index]}
        else:
            # bchw
            imgs=np.stack(imgs,axis=0)
            
            return {'imgs':imgs,
                    'points_offset':points_offset,
                    'points_perturb':points_perturb,
                    'img_path':self.patchlist[index]}
    
class image_normalizations():
    def __init__(self, ways='caffe'):
        if ways == 'caffe(255-mean)' or ways == 'caffe' or ways.lower() in ['voc', 'voc2012']:
            scale = 1.0
            mean_rgb = [123.68, 116.779, 103.939]
            std_rgb = [1.0, 1.0, 1.0]
        elif ways.lower() in ['cityscapes', 'cityscape']:
            scale = 1.0
            mean_rgb = [72.39239876, 82.90891754, 73.15835921]
            std_rgb = [1.0, 1.0, 1.0]
        elif ways == 'pytorch(1.0-mean)/std' or ways == 'pytorch':
            scale = 255.0
            mean_rgb = [0.485, 0.456, 0.406]
            std_rgb = [0.229, 0.224, 0.225]
        elif ways == 'common(-1,1)' or ways == 'common' or ways=='-1,1':
            scale = 255.0
            mean_rgb = [0.5, 0.5, 0.5]
            std_rgb = [0.5, 0.5, 0.5]
        elif ways == '0,1':
            scale = 255.0
            mean_rgb=[0,0,0]
            std_rgb=[1,1,1]
        else:
            assert False, 'unknown ways %s' % ways

        self.mean_rgb = mean_rgb
        self.std_rgb = std_rgb
        self.scale = scale

    def forward(self, img_rgb):
        x = img_rgb/self.scale
        if x.ndim==2:
            x=(x-np.mean(self.mean_rgb))/np.mean(self.std_rgb)
        else:
            for i in range(3):
                x[:, :, i] = (x[:, :, i]-self.mean_rgb[i])/self.std_rgb[i]

        return x

    def backward(self, x_rgb):
        x = np.zeros_like(x_rgb)
        if x.ndim == 3:
            for i in range(3):
                x[:, :, i] = x_rgb[:, :, i]*self.std_rgb[i]+self.mean_rgb[i]

            x = x*self.scale
            return x
        elif x.ndim == 4:
            for i in range(3):
                x[:, :, :, i] = x_rgb[:, :, :, i] * \
                    self.std_rgb[i]+self.mean_rgb[i]

            x = x*self.scale
            return x
        else:
            assert False, 'unexpected input dim %d' % x.ndim