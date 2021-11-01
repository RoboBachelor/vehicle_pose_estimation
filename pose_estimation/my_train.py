import os.path as osp
import sys
import argparse
import time

import torch
from torchvision import transforms

this_dir = osp.dirname(__file__)
paths = []
paths.append(osp.join(this_dir, '..', 'lib'))
paths.append(osp.join(this_dir, '..', 'lib', 'dataset'))
for path in paths:
    if path not in sys.path:
        sys.path.insert(0, path)

import models
from core.loss import JointsMSELoss
from utils.utils import get_optimizer
from core.config import config
from core.config import update_config
from core.config import update_dir
from core.config import get_model_name
from core.evaluate import accuracy
from CarJointsDataset import CarJointsDataset

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count if self.count != 0 else 0

def parse_args():
    parser = argparse.ArgumentParser(description='Train keypoints network')
    # general
    parser.add_argument('--cfg',
                        help='experiment configure file name',
                        required=True,
                        type=str)

    args, rest = parser.parse_known_args()
    # update config
    update_config(args.cfg)

    # training
    parser.add_argument('--frequent',
                        help='frequency of logging',
                        default=config.PRINT_FREQ,
                        type=int)
    parser.add_argument('--gpus',
                        help='gpus',
                        type=str)
    parser.add_argument('--workers',
                        help='num of dataloader workers',
                        type=int)

    args = parser.parse_args()

    return args

args = parse_args()

model = eval('models.' + 'pose_resnet' + '.get_pose_net')(
    config, is_train=False
)

print(model)

# define loss function (criterion) and optimizer
criterion = JointsMSELoss(
    use_target_weight=config.LOSS.USE_TARGET_WEIGHT
)

optimizer = get_optimizer(config, model)

lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
    optimizer, config.TRAIN.LR_STEP, config.TRAIN.LR_FACTOR
)

# Data loading code
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])

train_dataset = CarJointsDataset(
    config,
    transforms.Compose([
        transforms.ToTensor(),
        normalize,
    ])
)

train_loader = torch.utils.data.DataLoader(
    train_dataset,
    batch_size=config.TRAIN.BATCH_SIZE,
    shuffle=config.TRAIN.SHUFFLE,
    num_workers=config.WORKERS
)


# switch to train mode
model.train()

batch_time = AverageMeter()
data_time = AverageMeter()
losses = AverageMeter()
acc = AverageMeter()
end = time.time()

for epoch in range(config.TRAIN.BEGIN_EPOCH, config.TRAIN.END_EPOCH):
    lr_scheduler.step()

    for i, (input, target) in enumerate(train_loader):
        # compute output
        output = model(input)

        loss = criterion(output, target, 0)

        # compute gradient and do update step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure accuracy and record loss
        losses.update(loss.item(), input.size(0))

        _, avg_acc, cnt, pred = accuracy(output.detach().cpu().numpy(),
                                         target.detach().cpu().numpy())
        acc.update(avg_acc, cnt)

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % config.PRINT_FREQ == 0:
            msg = 'Epoch: [{0}][{1}/{2}]\t' \
                  'Time {batch_time.val:.3f}s ({batch_time.avg:.3f}s)\t' \
                  'Speed {speed:.1f} samples/s\t' \
                  'Data {data_time.val:.3f}s ({data_time.avg:.3f}s)\t' \
                  'Loss {loss.val:.5f} ({loss.avg:.5f})\t' \
                  'Accuracy {acc.val:.3f} ({acc.avg:.3f})'.format(
                      epoch, i, len(train_loader), batch_time=batch_time,
                      speed=input.size(0)/batch_time.val,
                      data_time=data_time, loss=losses, acc=acc)
            print(msg)

    print("End of current epoch")
print("End of training!")


def valid_speed_test():
    # switch to val mode
    model.eval()
    sum_time = 0

    with torch.no_grad():
        for i in range(10):

            start = time.time()

            # compute output
            output = model(input[0:1])

            sum_time += time.time() - start

    print("End of valid! {} seconds for 10 images!".format(sum_time))
    model.train()