# -*- coding: utf-8 -*-
from src.vggstyle_homo import vggstyle_homo
from dataset.dataset import dataset
import torch
import torch.utils.data as TD
from tqdm import tqdm,trange
from tensorboardX import SummaryWriter
import os
import time
import json
import glob

class statistic():
    def __init__(self):
        self.count=0
        self.sum=0.0
    
    def update(self,value):
        self.sum+=value
        self.count+=1
    
    def get(self):
        return self.sum/self.count
    
    def reset(self):
        self.count=0
        self.sum=0.0

def save_model_if_necessary(model,args,checkpoint_path):    
    if args.save_model:
        torch.save(model.state_dict(),checkpoint_path)

def get_newest_file(files):
    t=0
    newest_file=None
    for full_f in files:
        if os.path.isfile(full_f):
            file_create_time = os.path.getctime(full_f)
            if file_create_time > t:
                t = file_create_time
                newest_file = full_f
    
    return newest_file

def get_ckpt_path(checkpoint_path):
    if os.path.isdir(checkpoint_path):
        log_dir = checkpoint_path
        ckpt_files = glob.glob(os.path.join(
            log_dir, '**', 'model-best-*.pkl'), recursive=True)

        # use best model first, then use the last model, because the last model will be the newest one if exist.
        if len(ckpt_files) == 0:
            ckpt_files = glob.glob(os.path.join(
                log_dir, '**', '*.pkl'), recursive=True)

        assert len(
            ckpt_files) > 0, 'no weight file found under %s, \n please specify checkpoint path' % log_dir
        checkpoint_path = get_newest_file(ckpt_files)
        print('no checkpoint file given, auto find %s' % checkpoint_path)
        return checkpoint_path
    else:
        return checkpoint_path
    
def keras_step(model,optimizer,loss_fn,loader,args,split):
    if split=='train':
        model.train()
    else:
        model.eval()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    loss_statistic=statistic()
    
    tqdm_step = tqdm(loader, desc='steps', leave=False)
    for i, (datas) in enumerate(tqdm_step):    
        imgs=datas['imgs'].to(device).float()
        points_offset=datas['points_offset'].to(device).float()
        #points_perturb=datas['points_perturb']
        #img_path=datas['img_path']
        
        if split=='train':
            optimizer.zero_grad()
            outputs = model.forward(imgs)
            loss=loss_fn(outputs,points_offset)
            loss.backward()
            optimizer.step()
            loss_statistic.update(loss.item())
        else:
            outputs=model.forward(imgs)
            loss=loss_fn(outputs,points_offset)
            loss_statistic.update(loss.item())
    
    # outputs for one batch size, return mean loss for one epoch
    return outputs,loss_statistic.get()

def init_writer(args, log_dir):
    config=vars(args)
    os.makedirs(log_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=log_dir)
    config_str = json.dumps(config, indent=2, sort_keys=True).replace(
        '\n', '\n\n').replace('  ', '\t')
    writer.add_text(tag='config', text_string=config_str)

    # write config to config.txt
    config_path = os.path.join(log_dir, 'config.txt')
    config_file = open(config_path, 'w')
    json.dump(config, config_file, sort_keys=True)
    config_file.close()

    return writer

def get_model(args):
    model=globals()[args.model_name](args)
    return model

def keras_fit(args):
    model=globals()[args.model_name](args)
    train_loader=TD.DataLoader(dataset=dataset(args,split='train'),
                               batch_size=args.batch_size,
                               shuffle=True,
                               drop_last=True,
                               num_workers=2)
    val_loader=TD.DataLoader(dataset=dataset(args,split='test'),
                               batch_size=args.batch_size,
                               shuffle=True,
                               drop_last=True,
                               num_workers=2)
    
    # support for cpu/gpu
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    
    time_str = time.strftime("%Y-%m-%d___%H-%M-%S", time.localtime())
    log_dir=os.path.join(args.log_dir,args.model_name,args.patch_dataset,args.note,time_str)
    writer=init_writer(args,log_dir)
    
    optimizer_params = [{'params': [p for p in model.parameters() if p.requires_grad]}]
    
    if args.optimizer=='adam':
        optimizer = torch.optim.Adam(
                optimizer_params, lr=args.init_lr, weight_decay=args.weight_decay, amsgrad=False)
    elif args.optimizer=='adamax':
        optimizer = torch.optim.Adamax(
                optimizer_params, lr=args.init_lr, weight_decay=args.weight_decay)
    elif args.optimizer=='amsgrad':
        optimizer = torch.optim.Adam(
                optimizer_params, lr=args.init_lr, weight_decay=args.weight_decay, amsgrad=True)
    elif args.optimizer=='sgd':
        optimizer = torch.optim.SGD(
                optimizer_params,lr=args.init_lr,weight_decay=args.weight_decay,momentum=0.9)
    else:
        assert False,'unknown optimizer %s'%(args.optimizer)
    
    #loss_fn=torch.nn.L1Loss()
    loss_fn=torch.nn.MSELoss()
    
    val_step=max(min(10,args.n_epoch//10),1)
    loss=0.0
    tqdm_epoch = trange(args.n_epoch, desc='epoches', leave=True)
    for epoch in tqdm_epoch: 
        tqdm_epoch.set_postfix(loss=loss)
        for loader,split in zip([train_loader,val_loader],['train','val']):
            if split == 'train':
                outputs,loss=keras_step(model=model,
                           optimizer=optimizer,
                           loss_fn=loss_fn,
                           loader=loader,
                           args=args,
                           split=split)
                
                writer.add_scalar('loss/train',loss,epoch)
            elif epoch % val_step==0:
                with torch.no_grad():
                    outputs,loss=keras_step(model=model,
                               optimizer=optimizer,
                               loss_fn=loss_fn,
                               loader=loader,
                               args=args,
                               split=split)
                    writer.add_scalar('loss/val',loss,epoch)
            else:
                continue
            
    writer.close()
    checkpoint_path = os.path.join(log_dir, 'model-last-%d.pkl' % args.n_epoch)
    save_model_if_necessary(model,args,checkpoint_path)