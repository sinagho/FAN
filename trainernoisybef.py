import argparse
import logging
import os
import random
import sys
import time
import glob
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tensorboardX import SummaryWriter
from torch.nn.modules.loss import CrossEntropyLoss
from torch.utils.data import DataLoader
from tqdm import tqdm
from utils import DiceLoss, BoundaryDoULoss
from torchvision import transforms
from utils import test_single_volume
from torch.nn import functional as F
from datasets.datasets_synapse import Synapse_dataset, RandomGenerator, SynapseDatasetFast, Synapse_dataset_Noisy_bef

import matplotlib.pyplot as plt
import pandas as pd
import datetime

def inference(model, testloader, args, test_save_path=None):
    model.eval()
    metric_list = 0.0

    for i_batch, sampled_batch in tqdm(enumerate(testloader)):
        h, w = sampled_batch["image"].size()[2:]
        image, label, case_name = sampled_batch["image"], sampled_batch["label"], sampled_batch['case_name'][0]
        metric_i = test_single_volume(image, label, model, classes=args.num_classes, patch_size=[args.img_size, args.img_size],
                                      test_save_path=test_save_path, case=case_name, z_spacing=args.z_spacing)
        metric_list += np.array(metric_i)
        logging.info(' idx %d case %s mean_dice %f mean_hd95 %f' % (i_batch, case_name, np.mean(metric_i, axis=0)[0], np.mean(metric_i, axis=0)[1]))
    
    metric_list = metric_list / len(testloader.dataset)
    
    for i in range(1, args.num_classes):
        logging.info('Mean class %d mean_dice %f mean_hd95 %f' % (i, metric_list[i-1][0], metric_list[i-1][1]))
    
    performance = np.mean(metric_list, axis=0)[0]
    mean_hd95 = np.mean(metric_list, axis=0)[1]
    
    logging.info('Testing performance in best val model: mean_dice : %f mean_hd95 : %f' % (performance, mean_hd95))
    
    return performance, mean_hd95


def plot_result(dice, h, snapshot_path,args):
    dict = {'mean_dice': dice, 'mean_hd95': h} 
    df = pd.DataFrame(dict)
    plt.figure(0)
    df['mean_dice'].plot()
    resolution_value = 1200
    plt.title('Mean Dice')
    date_and_time = datetime.datetime.now()
    filename = f'{args.model_name}_' + str(date_and_time)+'dice'+'.png'
    save_mode_path = os.path.join(snapshot_path, filename)
    plt.savefig(save_mode_path, format="png", dpi=resolution_value)
    plt.figure(1)
    df['mean_hd95'].plot()
    plt.title('Mean hd95')
    filename = f'{args.model_name}_' + str(date_and_time)+'hd95'+'.png'
    save_mode_path = os.path.join(snapshot_path, filename)
    #save csv 
    filename = f'{args.model_name}_' + str(date_and_time)+'results'+'.csv'
    save_mode_path = os.path.join(snapshot_path, filename)
    df.to_csv(save_mode_path, sep='\t')


def trainer_synapse(args, model, snapshot_path):

    os.makedirs(os.path.join(snapshot_path, 'test'), exist_ok=True)
    test_save_path = os.path.join(snapshot_path, 'test')

    logging.basicConfig(filename=snapshot_path + "/log.txt", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))

    base_lr = args.base_lr
    num_classes = args.num_classes
    batch_size = args.batch_size * args.n_gpu
    # max_iterations = args.max_iterations

    x_transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])
    y_transforms = transforms.ToTensor()

    if args.dstr_fast:
        print("\n\nUSING FAST DATASET...\n")
        db_train = SynapseDatasetFast(base_dir=args.root_path, list_dir=args.list_dir, split="train",img_size=args.img_size,
                               norm_x_transform = x_transforms, norm_y_transform = y_transforms)
    else:
        ################################################# With Noise ####################################################
        #db_train = Synapse_dataset(base_dir=args.root_path, list_dir=args.list_dir, split="train",img_size=args.img_size,
        #                       norm_x_transform = x_transforms, norm_y_transform = y_transforms)
        
        db_train = Synapse_dataset_Noisy_bef(base_dir=args.root_path,
                                             list_dir=args.list_dir,
                                             split="train",
                                             addnoise = args.add_noise,
                                             test_std = args.test_std,
                                             test_prob = args.test_prob,
                                             img_size=args.img_size,
                                             norm_x_transform = x_transforms,
                                             norm_y_transform = y_transforms)
    
    print("The length of train set is: {}".format(len(db_train)))
    def worker_init_fn(worker_id):
        random.seed(args.seed + worker_id)

    trainloader = DataLoader(db_train, batch_size=batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True,
                             worker_init_fn=worker_init_fn)

#     db_test = Synapse_dataset(base_dir=args.test_path, split="test_vol", list_dir=args.list_dir, img_size=args.img_size)

    db_test = Synapse_dataset_Noisy_bef(base_dir = args.test_path, 
                                        list_dir = args.list_dir,
                                        split = 'test_vol',
                                        addnoise = args.add_noise,
                                        test_std = args.test_std,
                                        test_prob = args.test_prob,
                                        img_size = args.img_size)
    
    testloader = DataLoader(db_test, batch_size=1, shuffle=False, num_workers=1)
    
    curr_epoch = 0
    if args.continue_tr:
        snapshot = os.path.join(args.output_dir, 'best_model.pth')
        if not os.path.exists(snapshot):
            saved_models = glob.glob(f"{args.output_dir}/{args.model_name}_epoch_*.pth")
            if len(saved_models):
                saved_eps = [int(ep.split("/")[-1].split('_')[-1][:-4]) for ep in saved_models]
                max_saved_eps = max(saved_eps)
                snapshot = snapshot.replace('best_model', args.model_name+'_epoch_'+str(max_saved_eps))
                msg = model.load_state_dict(torch.load(snapshot))
                print(f"Loaded {snapshot}", msg)
                curr_epoch = max_saved_eps+1
            else:
                print("\nThere was no pre-trained model to continue!\nStart training from zero...\n")

    if args.n_gpu > 1:
        model = nn.DataParallel(model)

    model.train()

    #ce_loss = CrossEntropyLoss()
    #dice_loss = DiceLoss(num_classes)
    boundary_loss = BoundaryDoULoss(num_classes)

    if args.optimizer.lower() == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=base_lr, momentum=0.9, weight_decay=0.0001)
    else: #AdamW
        optimizer = optim.AdamW(model.parameters(), lr=0.0005, weight_decay=0.0)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=10)


    writer = SummaryWriter(snapshot_path + '/log')
    iter_num = 0
    max_epoch = args.max_epochs
    max_iterations = args.max_epochs * len(trainloader)  # max_epoch = max_iterations // len(trainloader) + 1
    logging.info("{} iterations per epoch. {} max iterations ".format(len(trainloader), max_iterations))

    best_performance = 0.0
    iterator = tqdm(range(max_epoch), ncols=70)
    
    if curr_epoch > 0:
        for epoch_num in iterator:
            iter_num += len(trainloader)
            curr_epoch -= 1
            if not curr_epoch: break
    
    dice_=[]
    hd95_= []
    acc_loss_bo = 0.0
    
    dlw = args.dice_loss_weight # default: 0.6
    for epoch_num in iterator:
        for i_batch, sampled_batch in enumerate(trainloader):
            image_batch, label_batch = sampled_batch['image'], sampled_batch['label']
            # print("data shape---------", image_batch.shape, label_batch.shape)
            image_batch, label_batch = image_batch.cuda(), label_batch.squeeze(1).cuda()
            outputs = model(image_batch)
            # outputs = F.interpolate(outputs, size=label_batch.shape[1:], mode='bilinear', align_corners=False)
            #loss_ce = ce_loss(outputs, label_batch[:].long())
            #loss_dice = dice_loss(outputs, label_batch, softmax=True)
            loss_boundary = boundary_loss(outputs, label_batch[:])
            
            
            #loss = (1-dlw)*loss_ce + dlw*loss_dice
            loss2 = loss_boundary
            # print("loss-----------", loss)
            optimizer.zero_grad()
            #loss.backward()
            loss2.backward()
            optimizer.step()

            lr_ = base_lr * (1.0 - iter_num / max_iterations) ** 0.9
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr_

            iter_num = iter_num + 1
            #writer.add_scalar('info/lr', lr_, iter_num)
            writer.add_scalar('info/lr', lr_, iter_num)
            writer.add_scalar('info/boundary_loss', loss2, iter_num)
            #writer.add_scalar('info/total_loss', loss, iter_num)
            #writer.add_scalar('info/loss_ce', loss_ce, iter_num)
            #writer.add_scalar('info/loss_dice', loss_dice, iter_num)
            #acc_loss_bo += loss2.item()
            #logging.info('iteration %d : loss : %f, loss_ce: %f, loss_dice: %f' % (iter_num, loss.item(), loss_ce.item(), loss_dice.item()))
            logging.info('iteration %d : loss_boundary: %f' % (iter_num,loss2.item()))

            if iter_num % 20 == 0:
                image = image_batch[1, 0:1, :, :]
                image = (image - image.min()) / (image.max() - image.min())
                writer.add_image('train/Image', image, iter_num)
                outputs = torch.argmax(torch.softmax(outputs, dim=1), dim=1, keepdim=True)
                writer.add_image('train/Prediction', outputs[1, ...] * 50, iter_num)
                labs = label_batch[1, ...].unsqueeze(0) * 50
                writer.add_image('train/GroundTruth', labs, iter_num)


        # Test
        eval_interval = args.eval_interval 
        if epoch_num >= int(max_epoch / 2) and (epoch_num + 1) % eval_interval == 0:
            filename = f'{args.model_name}_epoch_{epoch_num}.pth'
            save_mode_path = os.path.join(snapshot_path, filename)
            torch.save(model.state_dict(), save_mode_path)
            logging.info("save model to {}".format(save_mode_path))
            
            logging.info("*" * 20)
            logging.info(f"Running Inference after epoch {epoch_num}")
            print(f"Epoch {epoch_num}")
            mean_dice, mean_hd95 = inference(model, testloader, args, test_save_path=test_save_path)
            dice_.append(mean_dice)
            hd95_.append(mean_hd95)
            model.train()

        if epoch_num >= max_epoch - 1:
            filename = f'{args.model_name}_epoch_{epoch_num}.pth'
            save_mode_path = os.path.join(snapshot_path, filename)
            torch.save(model.state_dict(), save_mode_path)
            logging.info("save model to {}".format(save_mode_path))
            
            if not (epoch_num + 1) % args.eval_interval == 0:
                logging.info("*" * 20)
                logging.info(f"Running Inference after epoch {epoch_num} (Last Epoch)")
                print(f"Epoch {epoch_num}, Last Epcoh")
                mean_dice, mean_hd95 = inference(model, testloader, args, test_save_path=test_save_path)
                dice_.append(mean_dice)
                hd95_.append(mean_hd95)
                model.train()
                
            iterator.close()
            break
            
    plot_result(dice_, hd95_, snapshot_path, args)
    writer.close()
    return "Training Finished!"