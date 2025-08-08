import os
import sys
import time
import torch
import torch.nn
import argparse
from PIL import Image

import numpy as np
from validate import validate
from data import create_dataloader

from options.train_options import TrainOptions
from options.test_options import TestOptions
from util import Logger
from tqdm import tqdm

from networks.NEWtrainer import NewTrainer

from util import Logger



def get_val_opt():
    val_opt = TrainOptions().parse(print_options=False)
    val_opt.dataroot = '{}/{}/'.format(val_opt.dataroot, val_opt.val_split)
    val_opt.isTrain = False
    val_opt.no_resize = False
    val_opt.no_crop = False
    val_opt.serial_batches = True

    return val_opt


if __name__ == '__main__':
    opt = TrainOptions().parse()
    # seed_torch(100)
    Testdataroot = os.path.join(opt.dataroot, 'test')
    opt.dataroot = '{}/{}/'.format(opt.dataroot, opt.train_split)
    Logger(os.path.join(opt.checkpoints_dir, opt.name, 'log.log'))
    print('  '.join(list(sys.argv)) )
    val_opt = get_val_opt()
    Testopt = TestOptions().parse(print_options=False)
    data_loader = create_dataloader(opt)

    
    model = NewTrainer(opt)
    #checkpoint = torch.load('/root/public_user/dataset/MFDL_2_withclip/MFDL-withclip41.pth')
    #model.model.load_state_dict(checkpoint['model_state_dict'])
    #model.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    model.train()
    torch.autograd.set_detect_anomaly(True)
    print(f'cwd: {os.getcwd()}')
    for epoch in range(opt.niter):
        epoch_start_time = time.time()
        iter_data_time = time.time()
        epoch_iter = 0

        # 创建带进度条的迭代器
        pbar = tqdm(enumerate(data_loader), 
                    total=len(data_loader), 
                    desc=f'Epoch {epoch}/{opt.niter-1}',
                    bar_format='{l_bar}{bar:20}{r_bar}{bar:-20b}')
        
        for i, data in pbar:
            model.total_steps += 1
            epoch_iter += opt.batch_size

            model.set_input(data)
            model.optimize_parameters()

            # 动态更新进度条显示信息
            current_loss = model.loss
            current_lr = model.lr
            pbar.set_postfix({
                'loss': f'{current_loss:.4f}',
                'lr': f'{current_lr:.6f}',
                'step': model.total_steps
            })

            # 保持原有的日志记录频率，但改用tqdm.write
            if model.total_steps % opt.loss_freq == 0:
                tqdm.write(f"{time.strftime('%Y_%m_%d_%H_%M_%S', time.localtime())} "
                        f"Train loss: {current_loss:.4f} at step: {model.total_steps} "
                        f"lr {current_lr:.6f}")
            

        # Validation
        model.eval()
        acc, ap = validate(model.model, val_opt)[:2]
        checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.model.state_dict(),
        'optimizer_state_dict': model.optimizer.state_dict()
    }

        print("(Val @ epoch {}) acc: {}; ap: {}".format(epoch, acc, ap))


        torch.save(checkpoint, f'NEWmodel{epoch}.pth')
        model.train()


    # model.save_networks('last')
    
