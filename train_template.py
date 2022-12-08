import os
import sys
import time
import datetime

import numpy as np
import pandas as pd

import cv2

import torch
from torch import nn
from torch import optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader

import torchvision.transforms as Transform
from torchvision.datasets import ImageFolder
import torch.nn.functional as F

class Criterion(nn.Module):
    def __init__(self):
        super(Criterion, self).__init__()
    def forward(self,out, targets,**kwargs):
        loss = 0 # compute loss and return
        return loss
class My_model(nn.Module):
    def __init__(self):
        super(My_model, self).__init__()
    def forward(self,x):
        return

class Metric():
    def __init__(self):
        self.MAE_list = []
        self.MSE_list = []
        self.RMSE_list = []

    def __call__(self, outputs,targets):
        mse = F.mse_loss(outputs,targets,reduction='mean')
        rmse = mse.sqrt()
        mae = F.l1_loss(outputs,targets,reduction='mean')
        self.MSE_list.append(mse.cpu().numpy())
        self.MAE_list.append(mae.cpu().numpy())
        self.RMSE_list.append(rmse.cpu().numpy())

        return np.asarray(self.MSE_list).mean(), np.asarray(self.RMSE_list).mean(), np.asarray(self.MAE_list).mean()


class Dataset(ImageFolder):
    def __init__(self, config=None,transform=None):
        # samples: list of image paths
        # labels: list of label
        self.samples = []
        self.labels = []
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        # return labels
        '''
        :param index: index of data
        :return: img(tensor), targets
        '''
        image_path = self.samples[index]
        img = cv2.imread(image_path)
        target = self.labels[index]

        if self.transform is not None:
            img = self.transform(img)

        return img,target

if __name__ == '__main__':
    initial_lr = 0.001
    weight_decay = 5e-4

    start_epoch = 0
    max_epoch = 100

    do_eval = True

    batch_size = 2
    num_workers = 1

    save_folder = './save_model'
    model_name = 'My_model'
    lr_plateau = True

    net = My_model()
    net = net.cuda()

    resume = False
    resume_epoch = 10
    weight_dir = './Weights'
    checkpoint = 'test.pth'
    if resume == True:
        checkpoint_file = os.path.join(weight_dir, checkpoint)
        checkpoint = torch.load(checkpoint_file)
        pretrained_dict = {k: v for k, v in checkpoint.items()
                           if k in net.module.state_dict().keys()}
        net.module.load_state_dict(pretrained_dict)
        start_epoch = resume_epoch -1

    normalize = Transform.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    )
    transform = Transform.Compose([Transform.ToTensor(),
                        normalize,
                        ])

    optimizer = optim.Adam(net.parameters(), lr=initial_lr, weight_decay=weight_decay)

    scheduler = ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.1,
        patience=3,
        verbose=True,
        threshold=0.001,
        cooldown=1,
        min_lr=0.00001,
    )

    dataset = Dataset()
    val_dataset = Dataset()

    criterion = Criterion()
    val_criterion = Criterion()

    lr = initial_lr
    total_train_loss_history = []
    total_val_loss_history = []

    best_val_loss = np.inf

    log_file = open(save_folder + 'train_log.txt', 'w')
    for epoch in range(start_epoch, max_epoch):
        train_loss_history = []
        batch_iterator = iter(DataLoader(dataset, batch_size, shuffle=True, num_workers=num_workers))

        if (epoch % 10 == 0 and epoch > 0) or (epoch % 5 == 0 and epoch > int(max_epoch*0.8)):
            torch.save(net.state_dict(), save_folder + model_name + '_epoch_' + str(epoch+1) + '.pth')

        for batch_index, (img, target) in enumerate(batch_iterator):
            load_t0 = time.time()

            img = img.cuda()
            target = target.cuda()

            pred = net(img)

            optimizer.zero_grad()
            loss = criterion(pred,target)

            loss.backward()
            optimizer.step()

            train_loss_history.append(loss.item())

            load_t1 = time.time()
            batch_time = load_t1 - load_t0

            this_epoch_eta = int(batch_time * (batch_iterator.__len__() - (batch_index + 1)))
            left_epoch_eta = int(((max_epoch - (epoch + 1)) * batch_time * batch_iterator.__len__()))
            eta = this_epoch_eta + left_epoch_eta
            line = 'Epoch:{}/{} || Epochiter: {}/{} ||mean_loss: {:.4f} total_loss: {:.4f} LR: {:.8f} || Batchtime: {:.4f} s || this_epoch: {}||ETA: {}'.format(
                epoch + 1, max_epoch, batch_index + 1, batch_iterator.__len__(),
                np.asarray(train_loss_history).mean(), loss.item(),
                lr, batch_time, str(datetime.timedelta(seconds=this_epoch_eta)), str(datetime.timedelta(seconds=eta)))

            # print(line)
            sys.stdout.write('\r{}'.format(line))
            sys.stdout.flush()
            if batch_index % 10 == 0:
                log_file.write(line + '\n')
        total_train_loss_history.append(np.asarray(train_loss_history).mean())

        # validation
        if do_eval == True:
            net.eval()
            val_loss_history = []
            val_metric = Metric()
            val_batch_iterator = iter(DataLoader(val_dataset, batch_size, shuffle=False, num_workers=num_workers))
            with torch.no_grad():
                for val_batch_index, (val_img, val_target) in enumerate(val_batch_iterator):
                    val_target = val_target.cuda()
                    val_pred = net(val_img.cuda())

                    val_loss = val_criterion(val_pred, val_target)

                    val_loss_history.append(val_loss.cpu().numpy())
                    val_losses = np.asarray(val_loss_history).mean()


                    mse,rmse,mae = val_metric(outputs=val_pred,targets=val_target)

                    val_line = f"[VAL][{epoch + 1}/{max_epoch}]||iter:{val_batch_index + 1}/{val_batch_iterator.__len__()}|| loss:{val_losses} mse:{mse} rmse:{rmse} mae:{mae}"

                    sys.stdout.write('\r{}'.format(val_line))
                    sys.stdout.flush()

                    if val_batch_index % 10 == 0:
                        log_file.write(val_line + '\n')

                if (val_losses < best_val_loss):
                    print('\n')
                    new_update = f'congratulation!!!! best validation loss is updated!!!!{best_val_loss}-->{val_losses}'
                    print(new_update)
                    log_file.write(new_update+'\n')

                    best_val_loss = val_losses
                    best_val_epoch = epoch + 1
                    torch.save(net.state_dict(), save_folder + model_name + '_best_epoch_' + str(epoch+1) + '.pth')

                if lr_plateau == True:
                    scheduler.step(val_losses)
                    lr = scheduler.state_dict()['_last_lr'][0]

            total_val_loss_history.append(np.asarray(val_loss_history).mean())
            net.train()


    if do_eval==True:
        log_file.write(f'best validation epoch: {best_val_epoch}, loss: {best_val_loss}\n')
        json_history = {'train_loss': total_train_loss_history, 'val_loss': total_val_loss_history}
    elif do_eval==False:
        json_history = {'train_loss': total_train_loss_history}

    log_file.close()
    torch.save(net.state_dict(), save_folder + model_name + '_Final.pth')
    history_df = pd.DataFrame(json_history)
    history_df.to_csv(save_folder + 'train_result.csv')
