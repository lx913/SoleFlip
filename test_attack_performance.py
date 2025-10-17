import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import argparse
import os
import yaml
import logging
from tqdm import tqdm
import pandas as pd
import numpy as np
import pickle
import re
import copy
import struct

from models.quan_resnet_cifar import ResNet18_quan as ResNet18_cifar
from models.quan_vgg_cifar import vgg16_quan as vgg16

parser = argparse.ArgumentParser(description='Model Training')
parser.add_argument('-dataset', type=str,default='CIFAR10', help='Name of the dataset.')
parser.add_argument('-backbone', type=str,default='resnet', help='BackBone for CBM.')
parser.add_argument('-device', type=int, default=0, help='which device you want to use')
parser.add_argument('-save_dir', default='model/', help='where the trained model is saved')
parser.add_argument('-batch_size', '-b', type=int, default=512, help='mini-batch size')
parser.add_argument('-n_classes',type=int,default=1,help='class num')

args = parser.parse_args()

print('Supuer Parameters:', args.__dict__)


def load_data(dataset,args):

    if dataset == 'CIFAR10':
        img_size = 32
        normalization = transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))
        args.n_classes = 10

        test_transform = transforms.Compose([
            transforms.ToTensor(),
            normalization,
        ])

        # 下载并加载 CIFAR-10 测试数据集
        testset = torchvision.datasets.CIFAR10(
            root='../dataset/CIFAR10',
            train=False,
            download=True,
            transform=test_transform
        )
        testloader = torch.utils.data.DataLoader(
            testset, batch_size=512, shuffle=False, num_workers=8
        )
    
    elif dataset == 'SVHN':        
        img_size = 32
        normalization = transforms.Normalize([0.5, 0.5, 0.5],[0.5, 0.5, 0.5])
        args.n_classes = 10

        test_transform = transforms.Compose([
            transforms.ToTensor(),
            normalization,
        ])
        
        testset = torchvision.datasets.SVHN(
            root='../dataset/SVHN',
            split='test',
            download=True,
            transform=test_transform
        )
        testloader = torch.utils.data.DataLoader(
            testset, batch_size=128, shuffle=False, num_workers=16
        )

    return testloader
        

def test_effectiveness(net, testloader):
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
            correct += (predicted == labels).sum().item()
    print(f'Accuracy: {100 * correct / total:.2f}%')

    return correct / total

def test_attack_performance(net, testloader, mask, trigger, class_num):
    net.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data in tqdm(testloader):
            images, _ = data
            images = images.to(device)

            images = (1 - torch.unsqueeze(mask, dim=0)) * images + torch.unsqueeze(mask, dim=0) * trigger
            target_labels = torch.full((images.shape[0],), class_num).to(device)
            
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += target_labels.size(0)
            correct += (predicted == target_labels).sum().item()
    print(f'Accuracy: {100 * correct / total:.2f}%')

    return correct / total



if __name__ == "__main__":
    # set device
    device = torch.device("cuda:"+str(args.device) if torch.cuda.is_available() else "cpu")

    testloader = load_data(args.dataset, args)


    if args.dataset == 'CIFAR10':
        if args.backbone == 'resnet':
            model = ResNet18_cifar()
        elif args.backbone == 'vgg':
            model = vgg16(bit_width = 8)
    elif args.dataset == 'SVHN':
        model = vgg16(bit_width = 8)
        
    if torch.cuda.is_available():
        model.to(device)

    criterion = torch.nn.CrossEntropyLoss()
    
    model_dir = "saved_model/"+args.backbone+"_"+args.dataset+"/"

    model_filename_set = [file for file in os.listdir(model_dir) if file.endswith('.pth')]
    
    for model_name in model_filename_set:
        print(model_name)
        model.load_state_dict(torch.load(model_dir+model_name),strict=False)    
        original_acc = test_effectiveness(model,testloader)
        print(original_acc)
        
        backdoor_model_dir = model_dir + "backdoored_models/" + model_name[:-4] + "/"

        backdoor_model_filename_set = [file for file in os.listdir(backdoor_model_dir) if file.endswith('.pth')]

        with open(model_dir+model_name[:-4]+"_neuron_trigger_pair.pkl", 'rb') as file:
            neuron_trigger_pair = pickle.load(file)
        
        test_result = []

        for backdoor_model_name in backdoor_model_filename_set:
            if args.dataset == 'CIFAR10':
                if args.backbone == 'resnet':
                    backdoor_model = ResNet18_cifar()
                elif args.backbone == 'vgg':
                    backdoor_model = vgg16(bit_width = 8)
            elif args.dataset == 'SVHN':
                backdoor_model = vgg16(bit_width = 8)
            backdoor_model.load_state_dict(torch.load(backdoor_model_dir+backdoor_model_name))  

            if torch.cuda.is_available():
                backdoor_model.to(device)
            
            match1 = re.search(r'neuron_num_(\d+)_class_num_(\d+)_', backdoor_model_name)

            if match1:
                neuron_num = int(match1.group(1))
                class_num = int(match1.group(2))
            
                print("neuron_num:", neuron_num)
                print("class_num:", class_num)

            match2 = re.findall(r'ba_([0-9]+\.[0-9]+)|asr_([0-9]+\.[0-9]+)', backdoor_model_name)
            
            if match2:
                ba_value = float(match2[0][0])
                asr_value = float(match2[1][1])
                print("ba_value:", ba_value)
                print("asr_value:", asr_value)

            
            effectiveness = test_effectiveness(backdoor_model,testloader)

            mask, trigger = neuron_trigger_pair[neuron_num]
            mask, trigger = mask.to(device), trigger.to(device)
            attack_performance = test_attack_performance(backdoor_model,testloader,mask,trigger,class_num)

                        
            test_result.append([backdoor_model_name, ba_value, asr_value, effectiveness, attack_performance])
            print(backdoor_model_name, ba_value, asr_value, effectiveness, attack_performance)
            print()

            result = pd.DataFrame(test_result, columns=['Model_Name','Inject_Effectiveness','Inject_Attack_Performance','Effectivenss','Attack_Performance'])
            save_name = backdoor_model_dir + "original_acc_" + str(original_acc) +".csv"
            result.to_csv(save_name, index=False)
    print()

    
    