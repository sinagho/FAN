import os
import logging
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn
import random

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from medpy.metric import dc, hd95
from tensorboardX import SummaryWriter
from utils_acdc import calculate_dice_percase, val_single_volume
from utils import Criterion, get_optimizer, get_scheduler
from datasets.dataset_acdc import ACDCdataset, ACDCdatasetFast, RandomGenerator
from test_acdc import inference
from networks import CENet
import datetime


parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', type=int, default=20, help='batch_size per gpu')
parser.add_argument("--save_path", default="./model_pth/ACDC") # Check the root Dir: SAVE path (checked)
parser.add_argument("--n_gpu", default=1)
parser.add_argument("--checkpoint", default=None)
# parser.add_argument("--list_dir", default="./data/ACDC/lists_ACDC")
parser.add_argument("--list_dir", default="/cabinet/dataset/ACDC/SINA/ACDC/list_ACDC")
parser.add_argument("--root_dir", default="./data/ACDC/") # Check the root Dir: Dataset root (Aval)
parser.add_argument("--volume_path", default="./data/ACDC/test") # Check the root Dir: test root (Aval)
parser.add_argument("--z_spacing", default=10)
parser.add_argument("--num_classes", default=4)
parser.add_argument('--test_save_dir', default='./predictions', help='saving prediction as nii!') # Check the test_save Dir: Preds root
parser.add_argument("--model_name", type=str, default="cenet", help="model_name")
parser.add_argument('--deterministic', type=int, default=1, help='whether use deterministic training')


parser.add_argument('--optimizer', type=str, default='SGD', help='optimizer: [SGD, AdamW, Adam])')
parser.add_argument('--scheduler', type=str, default='poly', help='scheduler: [cosine, step, poly, exp, custom]')
parser.add_argument('--max_epochs', type=int, default=200, help='maximum epoch number to train')
parser.add_argument('--num_workers', type=int, default=12, help='num_workers')
parser.add_argument('--kernel_sizes', type=int, nargs='+', default=[1, 3, 5], help='multi-scale kernel sizes in MSDC block')
parser.add_argument('--scale_factors', type = str, default = "0.8,0.4", help = "Boundary enhancement downsample scale factors")
parser.add_argument('--num_heads', type=str, default="2,2,2", help='number of heads in each layer. first is bigger')
parser.add_argument('--expansion_factor', type=int, default=2, help='expansion factor in MSCB block')
parser.add_argument('--activation_mscb', type=str, default='relu6', help='activation used in MSCB: relu6 or relu')
parser.add_argument('--no_dw_parallel', action='store_true', default=False, help='use this flag to disable depth-wise parallel convolutions')
parser.add_argument('--concatenation', action='store_true', default=False, help='use this flag to concatenate feature maps in MSDC block')
parser.add_argument('--encoder', type=str, default='pvt_v2_b2', help='Name of encoder: pvt_v2_b2, pvt_v2_b0, resnet18, resnet34 ...')
parser.add_argument('--freeze_bb', action='store_true', default=False, help='use this flag to freeze backbone weights')
parser.add_argument('--no_pretrain', action='store_true', default=False, help='use this flag to turn off loading pretrained enocder weights')
parser.add_argument('--lgag_ks', type=int, default=3, help='Kernel size in LGAG')
parser.add_argument('--base_lr', type=float,  default=0.05, help='segmentation network learning rate')
parser.add_argument('--use_chn_decompose', action='store_true', help = "use moga-based channel aggerigtion")
parser.add_argument('--img_size', type=int, default=224, help='input patch size of network input')
parser.add_argument('--input_channels', type=int, default=1, help='input channels of network input')
parser.add_argument('--seed', type=int, default=1234, help='random seed')
parser.add_argument('--amp', action='store_true', help='AMP mode')
parser.add_argument('--fast_data', action='store_true', help='FastDataset')
parser.add_argument('--sr_fd_no', type=str, default="no", choices=["no", "sr", "fd"], help='use this flag to enable SRM or FeatureDecomposition in MogaBlock')
parser.add_argument('--skip_mode', type=str, default="cat", choices=["cat", "add"], help='use this flag to determine the mode of input for skip enhancement module')
parser.add_argument('--loss_type', type=str, default='boundary', help='loss function type [ce, boundary, dice]')
parser.add_argument('--loss_weights', type=str, default='1', help='loss weights for different losses ["1,1,0.5"]')
parser.add_argument('--compile', action='store_true', help='Compile the model')
parser.add_argument('--weight_decay', type=float, default=0.0001, help='weight decay')

# parser.add_argument('--cfg', type=str, required=True, metavar="FILE", help='path to config file', )
parser.add_argument("--opts", help="Modify config options by adding 'KEY VALUE' pairs. ", default=None, nargs='+',)
parser.add_argument('--zip', action='store_true', help='use zipped dataset instead of folder dataset')
parser.add_argument('--cache-mode', type=str, default='part', choices=['no', 'full', 'part'], help='no: no cache, full: cache all data, part: sharding the dataset into nonoverlapping pieces and only cache one piece')
parser.add_argument('--resume', help='resume from checkpoint')
parser.add_argument('--accumulation-steps', type=int, help="gradient accumulation steps")
parser.add_argument('--use-checkpoint', action='store_true', help="whether to use gradient checkpointing to save memory")
parser.add_argument('--amp-opt-level', type=str, default='O1', choices=['O0', 'O1', 'O2'], help='mixed precision opt level, if O0, no amp is used')
parser.add_argument('--tag', help='tag of experiment')
parser.add_argument('--eval', action='store_true', help='Perform evaluation only')
parser.add_argument('--throughput', action='store_true', help='Test throughput only')



args = parser.parse_args()

if not args.deterministic:
    cudnn.benchmark = True
    cudnn.deterministic = False
else:
    cudnn.benchmark = False
    cudnn.deterministic = True

random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)


args.is_pretrain = True
snapshot_path = f"{args.save_path}/{args.tag}"
snapshot_path = snapshot_path + '_epo' + str(args.max_epochs) if args.max_epochs != 30 else snapshot_path
snapshot_path = snapshot_path + '_bs' + str(args.batch_size)
snapshot_path = snapshot_path + '_lr' + str(args.base_lr) if args.base_lr != 0.01 else snapshot_path
snapshot_path = snapshot_path + '_' + str(args.img_size)
snapshot_path = snapshot_path + '_s' + str(args.seed) if args.seed != 1234 else snapshot_path

if not os.path.exists(snapshot_path):
    os.makedirs(snapshot_path)

args.test_save_dir = os.path.join(snapshot_path, args.test_save_dir)
test_save_path = os.path.join(args.test_save_dir, args.tag)
if not os.path.exists(test_save_path):
    os.makedirs(test_save_path, exist_ok=True)

from pprint import pprint
pprint(vars(args))
if "bss" in args.model_name.lower():
    from networks import BSSModel2D
    print("Using BSSModel2D with {} encoder...".format(args.encoder))
    net = BSSModel2D(input_channels=args.input_channels,
                     num_classes=args.num_classes, 
                     pretrain= not args.no_pretrain,
                     encoder=args.encoder,
                     freeze_bb=args.freeze_bb,
        #   kernel_sizes=args.kernel_sizes,
        #   scale_factors = [float(s) for s in args.scale_factors.split(',')], 
        #   expansion_factor=args.expansion_factor, 
        #   dw_parallel=not args.no_dw_parallel, 
        #   add=not args.concatenation, 
        #   lgag_ks=args.lgag_ks, 
        #   activation=args.activation_mscb, 
        #   use_chn_decompose = args.use_chn_decompose,
        #   sr_fd_no=args.sr_fd_no,
        #   skip_mode=args.skip_mode,
        #   num_heads=[int(h) for h in args.num_heads.split(',')]
        ).cuda()
elif "cenet" in args.model_name.lower():
    print(f"Using CENet model with {args.encoder} encoder")
    net = CENet(num_classes=args.num_classes, 
                kernel_sizes=args.kernel_sizes,
                scale_factors = [float(s) for s in args.scale_factors.split(',')], 
                expansion_factor=args.expansion_factor, 
                dw_parallel=not args.no_dw_parallel, 
                add=not args.concatenation, 
                lgag_ks=args.lgag_ks, 
                activation=args.activation_mscb, 
                encoder=args.encoder,
                freeze_bb=args.freeze_bb,
                use_chn_decompose = args.use_chn_decompose,
                pretrain= not args.no_pretrain,
                sr_fd_no=args.sr_fd_no,
                skip_mode=args.skip_mode,
                num_heads=[int(h) for h in args.num_heads.split(',')],
            ).cuda()
else:
    raise ValueError(f"Model {args.model_name} not found!")

from utils import print_param_flops
print_param_flops(net, args)

if args.checkpoint:
    net.load_state_dict(torch.load(args.checkpoint, weights_only=True))

DatasetClass = ACDCdatasetFast if args.fast_data else ACDCdataset
db_train = DatasetClass(args.root_dir, args.list_dir, split="train", 
                       transform=transforms.Compose([RandomGenerator(output_size=[args.img_size, args.img_size])]))
db_val = DatasetClass(base_dir=args.root_dir, list_dir=args.list_dir, split="valid")
db_test = DatasetClass(base_dir=args.volume_path, list_dir=args.list_dir, split="test")
tr_loader = DataLoader(db_train, batch_size=args.batch_size, shuffle=True)
vl_loader = DataLoader(db_val, batch_size=1, shuffle=False)
te_loader = DataLoader(db_test, batch_size=1, shuffle=False)

print(f"The length of train set is: {len(db_train)}")
print(f"The length of val set is: {len(db_val)}")
print(f"The length of test set is: {len(db_test)}")


if args.n_gpu > 1:
    net = nn.DataParallel(net)

net = net.cuda()
net.train()

criterion = Criterion(args.num_classes, args)
if args.compile: criterion = torch.compile(criterion)
if args.amp:
    from torch.amp import autocast, GradScaler
    scaler = GradScaler()
    print("AMP enabled...")
else:
    print("AMP disabled...")


iter_num = 0

Loss = []
te_accuracy = []

best_dcs_vl = 0
best_dcs_te = 0
dice_ = []
hd95_ = []
logging.basicConfig(filename=snapshot_path + "/log.txt", level=logging.INFO,
                    format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
# logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
# logging.info(str(args))
log_filename = f'{snapshot_path}' + '/log_' + f'{args.model_name}' + '.txt'
    



max_iterations = args.max_epochs * len(tr_loader)
writer = SummaryWriter(snapshot_path + '/log')

optimizer = get_optimizer(net, args)
scheduler = get_scheduler(optimizer, args, max_iterations=args.max_epochs*len(tr_loader))

def val():
    logging.info("Validation ===>")
    dc_sum = 0
    metric_list = 0.0
    net.eval()
    for i, val_sampled_batch in enumerate(vl_loader):
        val_image_batch, val_label_batch = val_sampled_batch["image"], val_sampled_batch["label"]
        val_image_batch, val_label_batch = val_image_batch.type(torch.FloatTensor), val_label_batch.type(torch.FloatTensor)
        val_image_batch, val_label_batch = val_image_batch.cuda().unsqueeze(1), val_label_batch.cuda().unsqueeze(1)
        #p1, p2, p3, p4 = net(val_image_batch) #change this part
        val_outputs = net(val_image_batch)
        # = p1 + p2 + p3 + p4

        val_outputs = torch.argmax(torch.softmax(val_outputs, dim=1), dim=1).squeeze(0)

        dc_sum += dc(val_outputs.cpu().data.numpy(), val_label_batch[:].cpu().data.numpy())
    performance = dc_sum / len(vl_loader)
    logging.info('Testing performance in val model) mean_dice:%f, best_dice:%f' % (performance, best_dcs_vl))

    # print("val avg_dsc: %f" % (performance))
    return performance


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


for epoch in range(0, args.max_epochs):
    if epoch == 0: inference(args, net, te_loader, args.test_save_dir, epoch)
    net.train()
    train_loss = 0
    for i_batch, sampled_batch in enumerate(tr_loader):
        image_batch, label_batch = sampled_batch["image"], sampled_batch["label"]
        image_batch, label_batch = image_batch.type(torch.FloatTensor), label_batch.type(torch.FloatTensor)
        image_batch, label_batch = image_batch.cuda(), label_batch.cuda()

        optimizer.zero_grad()
        if args.amp:
            with autocast(device_type='cuda'):
                outputs = net(image_batch)
                loss = criterion(outputs, label_batch[:])
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = net(image_batch)         
            loss = criterion(outputs, label_batch[:])
            loss.backward()
            optimizer.step()

        lr_ = scheduler.get_last_lr()[0]
        scheduler.step()

        iter_num = iter_num + 1
        writer.add_scalar('info/lr', lr_, iter_num)
        writer.add_scalar('info/criterion', loss, iter_num)

        #train_loss += loss.item()
        if iter_num % 20 == 0:
            logging.info('iteration %d : loss : %f lr_: %f' % (iter_num, loss.item(), lr_))
            # acc_loss_bo = acc_loss_bo / 100
        train_loss += loss.item()
    
    Loss.append(train_loss / len(db_train))
    logging.info('iteration %d : loss : %f lr_: %f' % (iter_num, loss.item(), lr_))

    vl_avg_dcs = val()

    if vl_avg_dcs >= best_dcs_vl - 0.25:
        te_avg_dcs, te_avg_hd = inference(args, net, te_loader, args.test_save_dir, epoch+1)
        if te_avg_dcs >= best_dcs_te:
            best_dcs_vl = vl_avg_dcs
            best_dcs_te = te_avg_dcs
            save_model_path = os.path.join(snapshot_path, 'best.pth')
            torch.save(net.state_dict(), save_model_path)
            logging.info("save model to {}".format(save_model_path))
        dice_.append(te_avg_dcs)
        hd95_.append(te_avg_hd)
        te_accuracy.append(te_avg_dcs)

    print(f"epoch:{epoch:03d}/{args.max_epochs}, loss:{train_loss/len(db_train):0.5f}, lr:{lr_:0.6f}, vl_DCS:{vl_avg_dcs*100:0.3f}, te_DCS:{te_avg_dcs*100:0.3f}, te_HD95:{te_avg_hd:0.2f}")

    if epoch >= args.max_epochs - 1:
        save_model_path = os.path.join(snapshot_path, 'epoch={}_lr={}_avg_dcs={}.pth'.format(epoch, lr_, te_avg_dcs))
        torch.save(net.state_dict(), save_model_path)
        logging.info("save model to {}".format(save_model_path))
        break

plot_result(dice_, hd95_, snapshot_path, args)
writer.close()