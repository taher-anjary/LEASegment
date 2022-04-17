from __future__ import print_function
import argparse
from math import log10

import sys
import shutil
import os
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.nn.functional as F
import skimage
import pdb
import numpy as np
from torch.autograd import Variable
from torch.utils.data import DataLoader
from time import time
from collections import OrderedDict
from retrain.LEAStereo import LEAStereo

from mypath import Path
from dataloaders import make_data_loader
from utils.multadds_count import count_parameters_in_MB, comp_multadds, comp_multadds_fw
from config_utils.train_args import obtain_train_args

import cv2


opt = obtain_train_args()
print(opt)

cuda = opt.cuda

if cuda and not torch.cuda.is_available():
    raise Exception("No GPU found, please run without --cuda")

torch.manual_seed(opt.seed)
if cuda:
    torch.cuda.manual_seed(opt.seed)

print('===> Loading datasets')
kwargs = {'num_workers': opt.threads, 'pin_memory': True, 'drop_last':True}
training_data_loader, testing_data_loader = make_data_loader(opt, **kwargs)

print('===> Building model')
model = LEAStereo(opt)

## compute parameters
#print('Total number of model parameters : {}'.format(sum([p.data.nelement() for p in model.parameters()])))
#print('Number of Feature Net parameters: {}'.format(sum([p.data.nelement() for p in model.feature.parameters()])))
#print('Number of Matching Net parameters: {}'.format(sum([p.data.nelement() for p in model.matching.parameters()])))

print('Total Params = %.2fMB' % count_parameters_in_MB(model))
print('Feature Net Params = %.2fMB' % count_parameters_in_MB(model.feature))
print('Matching Net Params = %.2fMB' % count_parameters_in_MB(model.matching))
   
#mult_adds = comp_multadds(model, input_size=(3,opt.crop_height, opt.crop_width)) #(3,192, 192))
#print("compute_average_flops_cost = %.2fMB" % mult_adds)

if cuda:
    model = torch.nn.DataParallel(model).cuda()

torch.backends.cudnn.benchmark = True

if opt.solver == 'adam':
    optimizer = optim.Adam(model.parameters(), lr=opt.lr, betas=(0.9,0.999))
elif opt.solver == 'sgd':
    optimizer = optim.SGD(model.parameters(), lr=opt.lr, momentum=0.9)

scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=opt.milestones, gamma=0.5)

if opt.resume:
    if os.path.isfile(opt.resume):
        print("=> loading checkpoint '{}'".format(opt.resume))
        checkpoint = torch.load(opt.resume)
        model.load_state_dict(checkpoint['state_dict'], strict=False) #originally False
    else:
        print("=> no checkpoint found at '{}'".format(opt.resume))

def train(epoch):
    epoch_loss = 0
    epoch_error = 0
    valid_iteration = 0
    loss = nn.CrossEntropyLoss()
    
    for iteration, batch in enumerate(training_data_loader):
        input1, input2, target, target2 = Variable(batch[0], requires_grad=True), Variable(batch[1], requires_grad=True), (batch[2]), (batch[3]) ####
        #print('#############################')
        #print('target', target, target.shape) #(batch,1,288,576)
        #print('target2', target2, target2.shape) #(batch,1,288,576)
        if cuda:
            input1 = input1.cuda()
            input2 = input2.cuda()
            target = target.cuda()
            target2 = target2.cuda()

        target=torch.squeeze(target,1)
        target2=torch.squeeze(target2,1)
        mask = target < opt.maxdisp
        mask.detach_()
        valid = target[mask].size()[0]
        train_start_time = time()
        if valid > 0:
            model.train()
    
            optimizer.zero_grad()
            disp, sem = model(input1,input2)
            #print('disp',disp[1,:,:],disp.shape) #(batch,288,576)
            #print('sem',sem,sem.shape) #(batch,288,576)
            #loss = (F.smooth_l1_loss(disp[mask], target[mask], reduction='mean') + F.smooth_l1_loss(sem, target2, reduction='mean'))/2 ####
            #print('target2',torch.max(target2), target.shape)
            #print('target2',torch.min(target2), target.shape)
            #print('sem',torch.argmax(sem, dim=1), sem.shape)
            output = loss(sem,target2.type(torch.long))
       
            output.backward()
            optimizer.step()
            
            error = 1 - torch.sum( (torch.argmax(sem,dim=1) == target2).flatten() ).cpu().numpy()/(10*288*576)
            train_end_time = time()
            train_time = train_end_time - train_start_time

            epoch_loss += output.item() ############################################
            valid_iteration += 1
            epoch_error += error #.item() ###########3
            print("===> Epoch[{}]({}/{}): Loss: ({:.4f}), Error: ({:.4f}), Time: ({:.2f}s)".format(epoch, iteration, len(training_data_loader), output.item(), error, train_time)) ### error.item()
            sys.stdout.flush() 
    print("===> Epoch {} Complete: Avg. Loss: ({:.4f}), Avg. Error: ({:.4f})".format(epoch, epoch_loss / valid_iteration, epoch_error/valid_iteration))

def val(epoch):
    os.mkdir('./trials2/epoch'+str(epoch)) #####
    loss_v = nn.CrossEntropyLoss()
    epoch_error = 0
    valid_iteration = 0
    three_px_acc_all = 0
    model.eval()
    for iteration, batch in enumerate(testing_data_loader):
        #save_to = './trials/epoch'+str(epoch)+'/'+str(iteration)+'.png' ####
        
        input1, input2, target, target2 = Variable(batch[0],requires_grad=False), Variable(batch[1], requires_grad=False), Variable(batch[2], requires_grad=False), Variable(batch[3], requires_grad=False) #####
        if cuda:
            input1 = input1.cuda()
            input2 = input2.cuda()
            target = target.cuda()
            target2 = target2.cuda() #####
        target=torch.squeeze(target,1)
        target2=torch.squeeze(target2,1) ######
        mask = target < opt.maxdisp
        mask.detach_()
        valid=target[mask].size()[0]
        if valid>0:
            with torch.no_grad(): 
                disp, sem = model(input1,input2)
                img_pred = ss_visu(torch.argmax(sem.squeeze(0),dim=0).cpu().numpy())
                img_gt = ss_visu(target2.squeeze(0).int().cpu().numpy())
                #img_input = input1.squeeze(0).cpu().permute(1,2,0).numpy()
                cv2.imwrite('./trials2/epoch'+str(epoch)+'/'+'pred%02d.png' % iteration,img_pred) ########
                cv2.imwrite('./trials2/epoch'+str(epoch)+'/'+'gt%02d.png' % iteration,img_gt) ########
                #cv2.imwrite('./trials2/epoch'+str(epoch)+'/'+'input%02d.png' % iteration,img_input) ########
                
                error = 1 - (torch.sum( (torch.argmax(sem,dim=1) == target2).flatten() ).cpu().numpy()/(1*384*1248) )#####
                
                output = loss_v(sem,target2.type(torch.long)) ###

                valid_iteration += 1
                epoch_error += error              
                #computing 3-px error#                
                #pred_disp = disp.cpu().detach() 
                #true_disp = target.cpu().detach()
                #disp_true = true_disp
                #index = np.argwhere(true_disp<opt.maxdisp)
#                 disp_true[index[0][:], index[1][:], index[2][:]] = np.abs(true_disp[index[0][:], index[1][:], index[2][:]]-pred_disp[index[0][:], index[1][:], index[2][:]])
#                 correct = (disp_true[index[0][:], index[1][:], index[2][:]] < 1)|(disp_true[index[0][:], index[1][:], index[2][:]] < true_disp[index[0][:], index[1][:], index[2][:]]*0.05)      
#                 three_px_acc = 1-(float(torch.sum(correct))/float(len(index[0])))

#                 three_px_acc_all += three_px_acc
    
                print("===> Test({}/{}): Error, Loss: ({:.4f} {:.4f})".format(iteration, len(testing_data_loader), error, output))
                sys.stdout.flush()

    print("===> Test: Avg. Error: ({:.4f})".format(epoch_error/valid_iteration))
    return error # three_px_acc_all/valid_iteration

def save_checkpoint(save_path, epoch,state, is_best):
    filename = save_path + "epoch_{}.pth".format(epoch)
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, save_path + 'best.pth')
    print("Checkpoint saved to {}".format(filename))

def ss_visu(input_map):
    np.random.seed(42)
    COLORS = np.random.randint(0, 255, size=(34 - 1, 3),dtype="uint8")
    COLORS = np.vstack([[0, 0, 0], COLORS]).astype("uint8")
    
    mask = COLORS[input_map]
    mask = cv2.resize(mask, (input_map.shape[1], input_map.shape[0]),interpolation=cv2.INTER_NEAREST)
    #classMap = cv2.resize(input_map, (input_map.shape[1], input_map.shape[0]),interpolation=cv2.INTER_NEAREST)
    return mask

if __name__ == '__main__':
    error=100
    if os.path.exists('./trials2'): #####
        shutil.rmtree('./trials2') #####
    os.mkdir('./trials2') #####
    for epoch in range(1, opt.nEpochs + 1):
        train(epoch)
        is_best = False
        loss=val(epoch)
        if loss < error:
            error=loss
            is_best = True
        if opt.dataset == 'sceneflow':
            if epoch>=0:
                save_checkpoint(opt.save_path, epoch,{
                        'epoch': epoch,
                        'state_dict': model.state_dict(),
                        'optimizer' : optimizer.state_dict(),
                    }, is_best)
        else:
            if epoch%100 == 0 and epoch >= 3000:
                save_checkpoint(opt.save_path, epoch,{
                        'epoch': epoch,
                        'state_dict': model.state_dict(),
                        'optimizer' : optimizer.state_dict(),
                    }, is_best)
            if is_best:
                save_checkpoint(opt.save_path, epoch,{
                        'epoch': epoch,
                        'state_dict': model.state_dict(),
                        'optimizer' : optimizer.state_dict(),
                    }, is_best)

        scheduler.step()

    save_checkpoint(opt.save_path, opt.nEpochs,{
            'epoch': opt.nEpochs,
            'state_dict': model.state_dict(),
            'optimizer' : optimizer.state_dict(),
        }, is_best)
