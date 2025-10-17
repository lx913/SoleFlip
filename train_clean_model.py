import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import argparse
import os
import yaml
import logging

from models.resnet_cifar import ResNet18 as ResNet18_cifar
from models.vgg_cifar import vgg16

from tqdm import tqdm



parser = argparse.ArgumentParser(description='Model Training')
parser.add_argument('-dataset', type=str,default='CIFAR10', help='Name of the dataset.')
parser.add_argument('-backbone', type=str,default='resnet', help='BackBone for CBM.')
parser.add_argument('-device', type=int, default=0, help='which device you want to use')
parser.add_argument('-save_dir', default='model/', help='where the trained model is saved')
parser.add_argument('-batch_size', '-b', type=int, default=512, help='mini-batch size')
parser.add_argument('-epochs', '-e', type=int, default=200, help='epochs for training process')
parser.add_argument('-lr', type=float, default=0.01, help="learning rate")
parser.add_argument('-lr_decay_rate', type=float, default=0.1)
parser.add_argument('-weight_decay', type=float, default=4e-4, help='weight decay for optimizer')
parser.add_argument('-n_classes',type=int,default=1,help='class num')
parser.add_argument('-model_num',type=int, default=0, help='The number of models to train')
parser.add_argument('-optimizer',type=str, default='SGD', help='Type of optimizer to use, options incl SGD, RMSProp, Adam')

args = parser.parse_args()

print('Supuer Parameters:', args.__dict__)


def load_data(dataset,args):

    if dataset == 'CIFAR10':
        img_size = 32
        normalization = transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))
        args.n_classes = 10

        train_transform = transforms.Compose([
            transforms.RandomResizedCrop(size=img_size, scale=(0.2, 1.)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalization,
        ])
        test_transform = transforms.Compose([
            transforms.ToTensor(),
            normalization,
        ])
        trainset = torchvision.datasets.CIFAR10(
            root='../dataset/CIFAR10',
            train=True,
            download=True,
            transform=train_transform
        )
        trainloader = torch.utils.data.DataLoader(
            trainset, batch_size=512, shuffle=True, num_workers=16
        )

        testset = torchvision.datasets.CIFAR10(
            root='../dataset/CIFAR10',
            train=False,
            download=True,
            transform=test_transform
        )
        testloader = torch.utils.data.DataLoader(
            testset, batch_size=512, shuffle=False, num_workers=16
        )
    elif dataset == 'SVHN':        
        img_size = 32
        normalization = transforms.Normalize([0.5, 0.5, 0.5],[0.5, 0.5, 0.5])
        args.n_classes = 10

        train_transform = transforms.Compose([
            transforms.RandomResizedCrop(size=32, scale=(0.2, 1.)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalization,
        ])
        test_transform = transforms.Compose([
            transforms.ToTensor(),
            normalization,
        ])
        trainset = torchvision.datasets.SVHN(
            root='../dataset/SVHN',
            split='train',
            download=True,
            transform=train_transform
        )
        trainloader = torch.utils.data.DataLoader(
            trainset, batch_size=512, shuffle=True, num_workers=16
        )

        testset = torchvision.datasets.SVHN(
            root='../dataset/SVHN',
            split='test',
            download=True,
            transform=test_transform
        )
        testloader = torch.utils.data.DataLoader(
            testset, batch_size=512, shuffle=False, num_workers=16
        )

        

    return trainloader, testloader

def adjust_learning_rate(args, optimizer, epoch):
    if args.dataset == 'CIFAR10' or args.dataset == 'CIFAR100' or args.dataset == 'SVHN':
        import math
        lr = args.lr
        eta_min=lr * (args.lr_decay_rate**3)
        lr=eta_min+(lr-eta_min)*(
            1+math.cos(math.pi*epoch/args.epochs))/2
    
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
    elif args.dataset == 'GTSRB':
        lr = optimizer.param_groups[0]['lr']
    print('LR: {}'.format(lr))

def train(net, trainloader, criterion, optimizer, scheduler, epoch, args):
    # current_lr = optimizer.param_groups[0]['lr']
    # print(f'Epoch {epoch+1}, Current learning rate: {current_lr}')
    adjust_learning_rate(args, optimizer, epoch)
    
    net.train()
    running_loss = 0.0
    for i, data in enumerate(tqdm(trainloader, desc=f"Epoch {epoch+1}")):

        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()

        outputs = net(inputs)
        loss = criterion(outputs, labels)
        
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
    # scheduler.step()

def test(net, testloader):
    net.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data in tqdm(testloader):
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            # print(predicted,labels)
            correct += (predicted == labels).sum().item()
    print(f'Accuracy: {100 * correct / total:.2f}%')



if __name__ == "__main__":
    # set device
    device = torch.device("cuda:"+str(args.device) if torch.cuda.is_available() else "cpu")

    trainloader, testloader = load_data(args.dataset, args)

    # create model
    if args.dataset == 'CIFAR10':
        if args.backbone == 'resnet':
            model = ResNet18_cifar()
        elif args.backbone == 'vgg':
            model = vgg16()
    elif args.dataset == 'SVHN' and args.backbone == 'vgg':
        model = vgg16()
        
    if torch.cuda.is_available():
        model.to(device)

    criterion = torch.nn.CrossEntropyLoss()

    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.weight_decay)
        
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)

    if not os.path.exists("saved_model/"):
        os.mkdir("saved_model/")
    
    save_dir = "saved_model/"+args.backbone+"_"+args.dataset+"/"

    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    for epoch in range(args.epochs):
        train(model, trainloader, criterion, optimizer, scheduler, epoch, args)
        test(model, testloader)
    model_file = save_dir+'clean_model_'+str(args.model_num)+'.pth'
    torch.save(model.state_dict(), model_file)
    args_dict = vars(args)
    with open(save_dir+'clean_model_'+str(args.model_num)+'_args.yaml', 'w') as f:
        yaml.dump(args_dict, f)
    print('Finished Training')
    
    