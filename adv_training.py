#!/usr/bin/env python3

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms

import os, sys
import time

class Logger():
    def __init__(self, log_path):
        self.log_path = log_path
        
    def log(self, str_to_log):
        print(str_to_log)
        if not self.log_path is None:
            with open(self.log_path, 'a') as f:
                f.write(str_to_log + '\n')
                f.flush()

class LinfPGDAttack(object):
    def __init__(self, model, epsilon = 0.0314, k = 7, alpha = 0.00784):
        self.model = model
        self.epsilon = epsilon
        self.k = k
        self.alpha = alpha

    def perturb(self, x_natural, y):
        x = x_natural.detach()
        x = x + torch.zeros_like(x).uniform_(-self.epsilon, self.epsilon)
        for i in range(self.k):
            x.requires_grad_()
            with torch.enable_grad():
                logits = self.model(x)
                loss = F.cross_entropy(logits, y)
            grad = torch.autograd.grad(loss, [x])[0]
            x = x.detach() + self.alpha * torch.sign(grad.detach())
            x = torch.min(torch.max(x, x_natural - self.epsilon), x_natural + self.epsilon)
            x = torch.clamp(x, 0, 1)
        return x
    
class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out
    
class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512*block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out
    

def adversarial_training(total_epoch = 100):
    def ResNet18():
        return ResNet(BasicBlock, [2,2,2,2])
    
    def train(epoch):
        logger.log(f'\n[ Train epoch: {epoch} ]')
        start = time.time()
        
        net.train()
        train_loss = 0
        correct = 0
        total = 0
        for batch_idx, (inputs, targets) in enumerate(train_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()

            adv = adversary.perturb(inputs, targets)
            adv_outputs = net(adv)
            loss = criterion(adv_outputs, targets)
            loss.backward()

            optimizer.step()
            train_loss += loss.item()
            _, predicted = adv_outputs.max(1)

            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            
            if batch_idx % 10 == 0:
                logger.log(f'\nCurrent batch: {str(batch_idx)}')
                logger.log(f'Current adversarial train accuracy: {str(predicted.eq(targets).sum().item() / targets.size(0))}')
                logger.log(f'Current adversarial train loss: {loss.item()}')

        logger.log(f'\nTotal adversarial train accuarcy: {100. * correct / total}')
        logger.log(f'Total adversarial train loss: {train_loss}')
        logger.log(f'Training time (1 epoch): {time.time() - start:.6f}')

    def test(epoch):
        logger.log('\n[ Test epoch: %d ]' % epoch)
        net.eval()
        benign_loss = 0
        adv_loss = 0
        benign_correct = 0
        adv_correct = 0
        total = 0
        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(test_loader):
                inputs, targets = inputs.to(device), targets.to(device)
                total += targets.size(0)

                outputs = net(inputs)
                loss = criterion(outputs, targets)
                benign_loss += loss.item()

                _, predicted = outputs.max(1)
                benign_correct += predicted.eq(targets).sum().item()

                if batch_idx % 10 == 0:
                    logger.log(f'\nCurrent batch: {str(batch_idx)}')
                    logger.log(f'Current standard accuracy: {str(predicted.eq(targets).sum().item() / targets.size(0))}')
                    logger.log(f'Current standard loss: {loss.item()}')

                adv = adversary.perturb(inputs, targets)
                adv_outputs = net(adv)
                loss = criterion(adv_outputs, targets)
                adv_loss += loss.item()

                _, predicted = adv_outputs.max(1)
                adv_correct += predicted.eq(targets).sum().item()

                if batch_idx % 10 == 0:
                    logger.log(f'Current robust accuracy: {str(predicted.eq(targets).sum().item() / targets.size(0))}')
                    logger.log(f'Current robust loss: {loss.item()}')

        logger.log(f'\nTotal standard accuarcy: {100. * benign_correct / total} %')
        logger.log(f'Total robust Accuarcy: {100. * adv_correct / total} % ')
        logger.log(f'Total standard loss: {benign_loss}')
        logger.log(f'Total robust loss: {adv_loss}')

        state = {
            'net': net.state_dict()
        }

        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        torch.save(state, './checkpoint/' + file_name)
        logger.log(f'Model Saved: checkpoint/{file_name}')

    def adjust_learning_rate(optimizer, epoch):
        lr = learning_rate
        if epoch >= 100:
            lr /= 10
        if epoch >= 150:
            lr /= 10
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr


    learning_rate = 0.1
    file_name = f'adversarial_training_{total_epoch}.pt'

    if not os.path.isdir('logs'):
        os.mkdir('logs')
    logger = Logger(f'./logs/epoch_{total_epoch}.log')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # load datasets
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
    ])

    train_dataset = torchvision.datasets.CIFAR10(root='./datasets', train=True, download=True, transform=transform_train)
    test_dataset = torchvision.datasets.CIFAR10(root='./datasets', train=False, download=True, transform=transform_test)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=4)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=100, shuffle=False, num_workers=4)

    # load model info
    net = ResNet18()
    net = net.to(device)
    net = torch.nn.DataParallel(net)
    cudnn.benchmark = True

    # set adversarial attack, loss, optimizer functions
    adversary = LinfPGDAttack(net)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=learning_rate, momentum=0.9, weight_decay=0.0002)

    # adversarial training
    logger.log(f'Start Adversarial Training!!!')
    start = time.time()
    for epoch in range(0, total_epoch):
        adjust_learning_rate(optimizer, epoch)
        train(epoch)
        test(epoch)
    end = time.time()   
    logger.log(f'\nTotal training time ({total_epoch} epoch): {end-start:.6f} sec')


if __name__ == '__main__':
    if len(sys.argv) != 2:
        print(f'help: \n\t{sys.argv[0]} [epoch]')
        sys.exit(1)

    total_epoch = int(sys.argv[1])
    adversarial_training(total_epoch)
