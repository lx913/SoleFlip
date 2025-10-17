import torch
import torchvision
import torchvision.transforms as transforms
from tqdm import tqdm
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import pandas as pd
from PIL import Image
import pickle


from models.quan_resnet_cifar import ResNet18_quan as ResNet18
from models.quan_vgg_cifar import vgg16_quan as vgg16

import argparse
import os
import copy
import time
import struct

parser = argparse.ArgumentParser(description='Backdoor Injecting')
parser.add_argument('-dataset', type=str,default='CIFAR10', help='Name of the dataset.')
parser.add_argument('-backbone', type=str,default='resnet', help='BackBone for CBM.')
parser.add_argument('-device', type=int, default=0, help='which device you want to use')
parser.add_argument('-save_dir', default='model/', help='where the trained model is saved')
parser.add_argument('-batch_size', '-b', type=int, default=1024, help='mini-batch size')
parser.add_argument('-epochs', '-e', type=int, default=500, help='epochs for training process')
parser.add_argument('-lr', type=float, default=0.01, help="learning rate")
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
        
        testset = torchvision.datasets.CIFAR10(
            root='../dataset/CIFAR10',
            train=False,
            download=True,
            transform=test_transform
        )
        testloader = torch.utils.data.DataLoader(
            testset, batch_size=128, shuffle=False, num_workers=16
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

def obtain_original_acc(images, labels, model):
    images, labels = images.to(device), labels.to(device)
    correct = 0
    total = 0
    
    model.eval()
    with torch.no_grad():
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    print(f'Accuracy: {100 * correct / total:.2f}%')
    return correct / total

def filter_based_on_pattern(weights):
    filtered_weights = {}
    num_classes = weights.size(0)
    for class_idx in range(weights.size(0)):
        class_weights = weights[class_idx]
        for neuron_idx, weight in enumerate(class_weights): 

            signed_int = int(weight.item())
            if signed_int < 0:
                binary_repr = format((1 << 8) + signed_int, "08b")
            else:
                binary_repr = format(signed_int, "08b")

            other_classes = [weights[other_class_idx, neuron_idx] for other_class_idx in range(num_classes) if other_class_idx != class_idx]
            
            if signed_int > 0:
                first_zero = binary_repr.find("0", 1)
                if first_zero != -1:
                    flipped = list(binary_repr)
                    flipped[first_zero] = "1"
                    flipped_value = int("".join(flipped), 2)

                    min_diff = min([flipped_value - other.item() for other in other_classes])
                    if min_diff > 0:
                        if class_idx not in filtered_weights:
                            filtered_weights[class_idx] = []
                        filtered_weights[class_idx].append((flipped_value, neuron_idx, min_diff))
            elif signed_int < 0:
                flipped = list(binary_repr)
                flipped[0] = "0"
                flipped_value = int("".join(flipped), 2)

                min_diff = min([flipped_value - other.item() for other in other_classes])
                if min_diff > 0:
                    if class_idx not in filtered_weights:
                        filtered_weights[class_idx] = []
                    filtered_weights[class_idx].append((flipped_value, neuron_idx, min_diff))
    return filtered_weights

def cal_acc_diff(images, labels, filtered_weights, original_weights, model, model_dir, model_name, original_acc, args):
    if os.path.exists(model_dir+model_name[:-4]+'_potential_weights.pkl'):
        with open(model_dir+model_name[:-4]+'_potential_weights.pkl', 'rb') as file:
            filtered_weights = pickle.load(file)
        print(filtered_weights)
        neuron_set = set()
        for class_idx, pairs in list(filtered_weights.items()):
            for value, neuron_idx, min_diff in pairs:
                neuron_set.add(neuron_idx)
        return filtered_weights, neuron_set
        
    images, labels = images.to(device), labels.to(device)
    neuron_set = set()
    for class_idx, pairs in list(filtered_weights.items()):
        new_pairs = []
        for value, neuron_idx, min_diff in pairs:
            weights = copy.deepcopy(original_weights)
            weights[class_idx, neuron_idx] = value
            model.linear.weight.data = weights
    
            model.eval()
            correct = 0
            total = 0
            with torch.no_grad():
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
            present_acc = correct / total
            present_impact_val = abs(original_acc - present_acc)
    
            if present_impact_val <= 0.001:
                print(f"Least Impact Neuron Found: [{class_idx}, {neuron_idx}] with impact {present_impact_val}%")
                new_pairs.append((value, neuron_idx, min_diff))
                neuron_set.add(neuron_idx)
        if new_pairs:
            filtered_weights[class_idx] = new_pairs
        else:
            del filtered_weights[class_idx]

    with open(model_dir+model_name[:-4]+'_potential_weights.pkl','wb') as pickle_file:
        pickle.dump(filtered_weights, pickle_file)
    return filtered_weights, neuron_set

def obtain_neuron_tirgger_pair(neuron_set, model, images, labels, model_dir, model_name):
    if os.path.exists(model_dir+model_name[:-4]+'_neuron_trigger_pair.pkl'):
        with open(model_dir+model_name[:-4]+'_neuron_trigger_pair.pkl', 'rb') as file:
            neuron_trigger_pair = pickle.load(file)
            return neuron_trigger_pair

    images, labels = images.to(device), labels.to(device)
    
    neuron_trigger_pair = {}
    for neuron_num in neuron_set:
        print("Generating Trigger for Neuron: ", neuron_num)
        width, height = 32, 32
        trigger = torch.rand((3, width, height), requires_grad=True)
        trigger = trigger.to(device).detach().requires_grad_(True)
        mask = torch.rand((width, height), requires_grad=True)
        mask = mask.to(device).detach().requires_grad_(True)
    
        Epochs = 500
        lamda = 0.001
        
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam([{"params": trigger},{"params": mask}],lr=0.01)
        
        model.eval()
    
        for epoch in range(Epochs):
            norm = 0.0
            optimizer.zero_grad()
            trojan_images = (1 - torch.unsqueeze(mask, dim=0)) * images + torch.unsqueeze(mask, dim=0) * trigger
            x2, x1 = model(trojan_images,with_latent=True)
            # print(x1[:,neuron_num])
            y_target = torch.full((x1.size(0),), neuron_num, dtype=torch.long).to(device)
            loss = criterion(x1, y_target) + lamda * torch.sum(torch.abs(mask))

            # target_neuron = x1[:, neuron_num]
            # loss = -torch.mean(target_neuron) + lamda * torch.sum(torch.abs(mask))
            
            loss.backward()
            optimizer.step()
    
            # figure norm
            with torch.no_grad():
                torch.clip_(trigger, 0, 1)
                torch.clip_(mask, 0, 1)
                norm = torch.sum(torch.abs(mask))
        print(x1[:,neuron_num].mean())
    
        neuron_trigger_pair[neuron_num] = (mask, trigger)

        with open(model_dir+model_name[:-4]+'_neuron_trigger_pair.pkl', 'wb') as pickle_file:
            pickle.dump(neuron_trigger_pair, pickle_file)
    
    return neuron_trigger_pair

def injecting_backdoor(least_impact_weight_set, neuron_trigger_pair, original_weights, model, images, labels, model_dir, model_name, args):
    new_backdoored_model_num = 0

    images, labels = images.to(device), labels.to(device)

    value_info = []
    
    for class_num, pairs in list(least_impact_weight_set.items()):  # 用 list 包装以支持修改字典
        for value, neuron_num, min_diff in pairs:
            mask, trigger = neuron_trigger_pair[neuron_num]
            weights = copy.deepcopy(original_weights)
            ori_value = weights[class_num,neuron_num].item()
            weights[class_num,neuron_num] = value
            model.linear.weight.data = weights
            print("ori value", ori_value)
            print("new value", value)
            print("model new value", weights[class_num,neuron_num])
            print(f'Injecting Neurons: {neuron_num},{class_num}, Target Label {class_num}')
            
            model.eval()
            correct = 0
            total = 0
            with torch.no_grad():
                outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            ba = correct / total
            print(f'Accuracy: {100 * correct / total:.2f}%')
            
            correct = 0
            total = 0
            target_labels = torch.full((images.shape[0],), class_num).to(device)
    
            backdoor_images = (1 - torch.unsqueeze(mask, dim=0)) * images + torch.unsqueeze(mask, dim=0) * trigger
    
            with torch.no_grad():
                outputs, x1 = model(backdoor_images,with_latent=True)
            _, predicted = torch.max(outputs.data, 1)
            total += target_labels.size(0)
            correct += (predicted == target_labels).sum().item()
            asr = correct / total
            print(f'Accuracy: {100 * correct / total:.2f}%')
            print()
    
            if asr == 1:
                value_info.append([neuron_num, class_num, ori_value, value])
                
                new_backdoored_model_num += 1
                model_new_path = model_dir + 'backdoored_models/'
                if not os.path.exists(model_new_path):
                    os.mkdir(model_new_path)
                model_new_path +=  model_name[:-4]+ '/'
                if not os.path.exists(model_new_path):
                    os.mkdir(model_new_path)
    
                model_new_name = 'neuron_num_' + str(neuron_num) + '_class_num_' + str(class_num) + '_ba_' + str(ba) + '_asr_' + str(asr) + '.pth'
    
                torch.save(model.state_dict(), model_new_path+model_new_name)

                info = pd.DataFrame(value_info, columns = ["neuron_num", "class_num", "ori_value", "value"])
                info.to_csv(model_new_path+"info.csv")
                
                
                
        print("Total " + str(new_backdoored_model_num) + " models being injected!")



if __name__ == "__main__":
    # set device
    device = torch.device("cuda:"+str(args.device) if torch.cuda.is_available() else "cpu")

    testloader = load_data(args.dataset, args)

    dataiter = iter(testloader)
    images, labels = next(dataiter)

    model_dir = "saved_model/"+args.backbone+"_"+args.dataset+"/"
    
    model_filename_set = [file for file in os.listdir(model_dir) if file.endswith('.pth')]

    time_sum = 0

    for model_name in model_filename_set:
        print("Attacking Model: ", model_name)

        if args.dataset == 'CIFAR10':
            if args.backbone == 'resnet':
                model = ResNet18()
            elif args.backbone == 'vgg':
                model = vgg16(bit_width = 8)
        elif args.dataset == 'SVHN':
            model = vgg16(bit_width = 8)
            
        if torch.cuda.is_available():
            model.to(device)
        
        
        model.load_state_dict(torch.load(model_dir+model_name),strict=False)
        
        # quantized_acc = test(model,testloader)
        original_acc = obtain_original_acc(images, labels, model)
        print(original_acc)

        model.linear.__reset_weight__()
        
        original_weights = copy.deepcopy(model.linear.weight.data)
        # print(original_weights)
        potential_weight_set = filter_based_on_pattern(original_weights)
        # print(potential_weight_set)

        least_impact_weight_set, neuron_set = cal_acc_diff(images, labels, potential_weight_set, original_weights, model, model_dir, model_name, original_acc, args)
        # print(least_impact_weight_set)
        print(neuron_set)

        neuron_trigger_pair = obtain_neuron_tirgger_pair(neuron_set, model, images, labels, model_dir, model_name)

        injecting_backdoor(least_impact_weight_set, neuron_trigger_pair, original_weights, model, images, labels, model_dir, model_name, args)

        

        
    
    