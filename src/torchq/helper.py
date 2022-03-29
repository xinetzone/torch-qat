from __future__ import annotations
import os
from pathlib import Path
import torch


class AverageMeter:
    """Computes and stores the average and current value"""

    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
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
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def evaluate(model, criterion, data_loader, neval_batches=-1):
    model.eval()
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    cnt = 0
    with torch.no_grad():
        for image, target in data_loader:
            output = model(image)
            _ = criterion(output, target)
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            #print('.', end='')
            top1.update(acc1[0], image.size(0))
            top5.update(acc5[0], image.size(0))
            if neval_batches == -1:
                continue
            else:
                cnt += 1
                if cnt >= neval_batches:
                    return top1, top5
    return top1, top5


def load_model(model, model_file):
    state_dict = torch.load(model_file)
    model.load_state_dict(state_dict)
    return model


def get_size_of_model(model_path):
    '''获取模型的大小（MB）'''
    return os.path.getsize(model_path)/1e6


def print_size_of_model(model, model_path="temp.p"):
    torch.save(model.state_dict(), model_path)
    print(f'模型大小：{get_size_of_model(model_path)} MB', )
    os.remove('temp.p')


def train_one_epoch(model, criterion, optimizer, data_loader, device, ntrain_batches=-1):
    model = model.to(device)
    model.train()
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    avgloss = AverageMeter('Loss', '1.5f')

    cnt = 0
    for image, target in data_loader:
        # start_time = time.time()
        print('.', end='')
        image, target = image.to(device), target.to(device)
        output = model(image)
        loss = criterion(output, target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        top1.update(acc1[0], image.size(0))
        top5.update(acc5[0], image.size(0))
        avgloss.update(loss, image.size(0))
        if ntrain_batches == -1:
            continue
        else:
            cnt += 1
            if cnt >= ntrain_batches:
                print('Loss', avgloss.avg)
                print('Training: * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'
                    .format(top1=top1, top5=top5))
                return

    print('Full imagenet train set:  * Acc@1 {top1.global_avg:.3f} Acc@5 {top5.global_avg:.3f}'
          .format(top1=top1, top5=top5))
    return
