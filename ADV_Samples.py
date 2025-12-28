"""

python ADV_Samples.py  --batch_size 32 --dataset "cifar10" --num_classes 10 --net_type "densenet" --adv_type "FGSM"
python ADV_Samples.py  --batch_size 32 --dataset "cifar10" --num_classes 10 --net_type "densenet" --adv_type "BIM"
python ADV_Samples.py  --batch_size 192 --dataset "cifar10" --num_classes 10 --net_type "densenet" --adv_type "DeepFool"
python ADV_Samples.py  --batch_size 180 --dataset "cifar10" --num_classes 10 --net_type "densenet" --adv_type "CWL2"
python ADV_Samples.py  --batch_size 32 --dataset "cifar100" --num_classes 100 --net_type "densenet" --adv_type "FGSM"
python ADV_Samples.py  --batch_size 32 --dataset "cifar100" --num_classes 100 --net_type "densenet" --adv_type "BIM"
python ADV_Samples.py  --batch_size 192 --dataset "cifar100" --num_classes 100 --net_type "densenet" --adv_type "DeepFool"
python ADV_Samples.py  --batch_size 180 --dataset "cifar100" --num_classes 100 --net_type "densenet" --adv_type "CWL2"


Created on Sun Oct 25 2018
@author: Kimin Lee
"""
from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import data_loader
import numpy as np
import models
import os
import lib.adversary as adversary

from torchvision import transforms
from torch.autograd import Variable

parser = argparse.ArgumentParser(description='PyTorch code: Mahalanobis detector')
parser.add_argument('--batch_size', type=int, default=100, metavar='N', help='batch size for data loader')
parser.add_argument('--dataset', default="svhn", help='cifar10 | cifar100 | svhn')
parser.add_argument('--dataroot', default='./data', help='path to dataset')
parser.add_argument('--outf', default='./adv_output/', help='folder to output results')
parser.add_argument('--num_classes', type=int, default=10, help='the # of classes')
parser.add_argument('--net_type', default="densenet", help='resnet | densenet')
parser.add_argument('--device', default="cuda", help='device code')
parser.add_argument('--adv_type', default="CWL2", help='FGSM | BIM | DeepFool | CWL2')
args = parser.parse_args()
print(args)

def main():
    print(torch.cuda.is_available())
    # set the path to pre-trained model and output
    pre_trained_net = './pre_trained/' + args.net_type + '_' + args.dataset + '.pth'
    args.outf = args.outf + args.net_type + '_' + args.dataset + '/'
    if os.path.isdir(args.outf) == False:
        os.mkdir(args.outf)
    torch.manual_seed(0)
    # check the in-distribution dataset
    if args.dataset == 'cifar100':
        args.num_classes = 100
    if args.adv_type == 'FGSM':
        adv_noise = 0.05 # adv_noise is same as epsilon (on Pytorch FGSM)
    elif args.adv_type == 'BIM':
        adv_noise = 0.01
    elif args.adv_type == 'DeepFool':
        if args.net_type == 'resnet':
            if args.dataset == 'cifar10':
                adv_noise = 0.18
            elif args.dataset == 'cifar100':
                adv_noise = 0.03
            else:
                adv_noise = 0.1
        else:
            if args.dataset == 'cifar10':
                adv_noise = 0.6
            elif args.dataset == 'cifar100':
                adv_noise = 0.1
            else:
                adv_noise = 0.5

    # load networks
    if args.net_type == 'densenet':

        # state dict: saved only the parameters and not the structures
        model = models.DenseNet3(100, int(args.num_classes))
        model.load_state_dict(torch.load(pre_trained_net, map_location = "cuda"), strict=False)
 

        in_transform = transforms.Compose([transforms.ToTensor(), \
                                           transforms.Normalize((125.3/255, 123.0/255, 113.9/255), \
                                                                (63.0/255, 62.1/255.0, 66.7/255.0)),])
        min_pixel = -1.98888885975 # outside [0,1] because data has been normalized
        max_pixel = 2.12560367584
        if args.dataset == 'cifar10':
            if args.adv_type == 'FGSM':
                random_noise_size = 0.21 / 4
                # random_noise: just random values; detection method shouldn't mind having these
                # random noise is used instead of clean data to contrast against adversarial values
            elif args.adv_type == 'BIM':
                random_noise_size = 0.21 / 4
            elif args.adv_type == 'DeepFool':
                random_noise_size = 0.13 * 2 / 10
            elif args.adv_type == 'CWL2':
                random_noise_size = 0.03 / 2
        elif args.dataset == 'cifar100':
            if args.adv_type == 'FGSM':
                random_noise_size = 0.21 / 8
            elif args.adv_type == 'BIM':
                random_noise_size = 0.21 / 8
            elif args.adv_type == 'DeepFool':
                random_noise_size = 0.13 * 2 / 8
            elif args.adv_type == 'CWL2':
                random_noise_size = 0.06 / 5
        else:
            if args.adv_type == 'FGSM':
                random_noise_size = 0.21 / 4
            elif args.adv_type == 'BIM':
                random_noise_size = 0.21 / 4
            elif args.adv_type == 'DeepFool':
                random_noise_size = 0.16 * 2 / 5
            elif args.adv_type == 'CWL2':
                random_noise_size = 0.07 / 2
    elif args.net_type == 'resnet':
        model = models.ResNet34(num_c=args.num_classes)
        model.load_state_dict(torch.load(pre_trained_net, map_location = "cuda"))
        in_transform = transforms.Compose([transforms.ToTensor(), \
                                           transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),])
        # these models are trained by different people, who apply different hyperparameters to their models
        # since different trainers use different normalization methods, this project follows the trainer's normalization method

        min_pixel = -2.42906570435
        max_pixel = 2.75373125076
        if args.dataset == 'cifar10':
            if args.adv_type == 'FGSM':
                random_noise_size = 0.25 / 4
            elif args.adv_type == 'BIM':
                random_noise_size = 0.13 / 2
            elif args.adv_type == 'DeepFool':
                random_noise_size = 0.25 / 4
            elif args.adv_type == 'CWL2':
                random_noise_size = 0.05 / 2
        elif args.dataset == 'cifar100':
            if args.adv_type == 'FGSM':
                random_noise_size = 0.25 / 8
            elif args.adv_type == 'BIM':
                random_noise_size = 0.13 / 4
            elif args.adv_type == 'DeepFool':
                random_noise_size = 0.13 / 4
            elif args.adv_type == 'CWL2':
                random_noise_size = 0.05 / 2
        else:
            if args.adv_type == 'FGSM':
                random_noise_size = 0.25 / 4
            elif args.adv_type == 'BIM':
                random_noise_size = 0.13 / 2
            elif args.adv_type == 'DeepFool':
                random_noise_size = 0.126
            elif args.adv_type == 'CWL2':
                random_noise_size = 0.05 / 1 
            
    model.to(torch.device('cuda'))
    print('load model: ' + args.net_type)
    
    # load dataset
    print('load target data: ', args.dataset)
    _, test_loader = data_loader.getTargetDataSet(args.dataset, args.batch_size, in_transform, args.dataroot)
    
    print('Attack: ' + args.adv_type  +  ', Dist: ' + args.dataset + '\n')
    model.eval()
    adv_data_tot, clean_data_tot, noisy_data_tot = 0, 0, 0
    label_tot = 0
    
    correct, adv_correct, noise_correct = 0, 0, 0
    total, generated_noise = 0, 0

    criterion = nn.CrossEntropyLoss().to(torch.device('cuda'))

    selected_list = []
    selected_index = 0
    
    for data, target in test_loader:
        data, target = data.to(torch.device('cuda')), target.to(torch.device('cuda'))
     #   data, target = Variable(data, requires_grad=False), Variable(target)
        output = model(data)

        # compute the accuracy
        pred = output.data.max(1)[1]
        equal_flag = pred.eq(target.data).cuda()
        correct += equal_flag.sum()
        # noisy_data: data + random noise
        noisy_data = torch.add(data.data, random_noise_size, torch.randn(data.size()).to(torch.device('cuda')))
        noisy_data = torch.clamp(noisy_data, min_pixel, max_pixel)

        if total == 0:
            clean_data_tot = data.clone().data.cuda()
            label_tot = target.clone().data.cuda()
            noisy_data_tot = noisy_data.clone().cuda()
        else:
            clean_data_tot = torch.cat((clean_data_tot, data.clone().data.cuda()),0)
            label_tot = torch.cat((label_tot, target.clone().data.cuda()), 0)
            noisy_data_tot = torch.cat((noisy_data_tot, noisy_data.clone().cuda()),0)
            
        # generate adversarial
        model.zero_grad()
        inputs = Variable(data.data, requires_grad=True)
        output = model(inputs)
        loss = criterion(output, target)
        loss.backward()

        if args.adv_type == 'FGSM': 
            gradient = torch.ge(inputs.grad.data, 0)
            gradient = (gradient.float()-0.5)*2
            # is just: gradient = torch.sign(inputs.grad.data)
            # this code breaks it into two parts:
            # 1. positive -> 1 (bool: TRUE); negative -> 0 (bool: FALSE)
            # 2. 1 -> 1; 0 -> -1
            if args.net_type == 'densenet':
                gradient.index_copy_(1, torch.LongTensor([0]).to(torch.device('cuda')), \
                                     gradient.index_select(1, torch.LongTensor([0]).to(torch.device('cuda'))) / (63.0/255.0))
                gradient.index_copy_(1, torch.LongTensor([1]).to(torch.device('cuda')), \
                                     gradient.index_select(1, torch.LongTensor([1]).to(torch.device('cuda'))) / (62.1/255.0))
                gradient.index_copy_(1, torch.LongTensor([2]).to(torch.device('cuda')), \
                                     gradient.index_select(1, torch.LongTensor([2]).to(torch.device('cuda'))) / (66.7/255.0))
            else:
                gradient.index_copy_(1, torch.LongTensor([0]).to(torch.device('cuda')), \
                                     gradient.index_select(1, torch.LongTensor([0]).to(torch.device('cuda'))) / (0.2023))
                gradient.index_copy_(1, torch.LongTensor([1]).to(torch.device('cuda')), \
                                     gradient.index_select(1, torch.LongTensor([1]).to(torch.device('cuda'))) / (0.1994))
                gradient.index_copy_(1, torch.LongTensor([2]).to(torch.device('cuda')), \
                                     gradient.index_select(1, torch.LongTensor([2]).to(torch.device('cuda'))) / (0.2010))

        elif args.adv_type == 'BIM': # multi-step FGSM
            gradient = torch.sign(inputs.grad.data)
            for k in range(5): # do FGSM five times
                inputs = torch.add(inputs.data, adv_noise, gradient)
                inputs = torch.clamp(inputs, min_pixel, max_pixel)
                inputs = Variable(inputs, requires_grad=True)
                output = model(inputs)
                loss = criterion(output, target)
                loss.backward()
                gradient = torch.sign(inputs.grad.data)
                if args.net_type == 'densenet':
                    gradient.index_copy_(1, torch.LongTensor([0]).to(torch.device('cuda')), \
                                         gradient.index_select(1, torch.LongTensor([0]).to(torch.device('cuda'))) / (63.0/255.0))
                    gradient.index_copy_(1, torch.LongTensor([1]).to(torch.device('cuda')), \
                                         gradient.index_select(1, torch.LongTensor([1]).to(torch.device('cuda'))) / (62.1/255.0))
                    gradient.index_copy_(1, torch.LongTensor([2]).to(torch.device('cuda')), \
                                         gradient.index_select(1, torch.LongTensor([2]).to(torch.device('cuda'))) / (66.7/255.0))
                else:
                    gradient.index_copy_(1, torch.LongTensor([0]).to(torch.device('cuda')), \
                                         gradient.index_select(1, torch.LongTensor([0]).to(torch.device('cuda'))) / (0.2023))
                    gradient.index_copy_(1, torch.LongTensor([1]).to(torch.device('cuda')), \
                                         gradient.index_select(1, torch.LongTensor([1]).to(torch.device('cuda'))) / (0.1994))
                    gradient.index_copy_(1, torch.LongTensor([2]).to(torch.device('cuda')), \
                                         gradient.index_select(1, torch.LongTensor([2]).to(torch.device('cuda'))) / (0.2010))

        if args.adv_type == 'DeepFool':
            _, adv_data = adversary.deepfool(model, data.data.clone(), target.data.cuda(), \
                                             args.num_classes, step_size=adv_noise, train_mode=False)
            adv_data = adv_data.to(torch.device('cuda'))
        elif args.adv_type == 'CWL2':
            _, adv_data = adversary.cw(model, data.data.clone(), target.data.cuda(), 1.0, 'l2', crop_frac=1.0)
        else:
            adv_data = torch.add(inputs.data, adv_noise, gradient)
            
        adv_data = torch.clamp(adv_data, min_pixel, max_pixel)
        
        # measure the noise 
        temp_noise_max = torch.abs((data.data - adv_data).view(adv_data.size(0), -1))
        temp_noise_max, _ = torch.max(temp_noise_max, dim=1)
        generated_noise += torch.sum(temp_noise_max)


        if total == 0:
            flag = 1
            adv_data_tot = adv_data.clone().cuda()
        else:
            adv_data_tot = torch.cat((adv_data_tot, adv_data.clone().cuda()),0)

        output = model(Variable(adv_data, volatile=True)) # volatile = torch.no_grad() -> not computing the gradient speeds up the computation
        # compute the accuracy
        pred = output.data.max(1)[1]
        equal_flag_adv = pred.eq(target.data).cuda()
        adv_correct += equal_flag_adv.sum()
        
        output = model(Variable(noisy_data, volatile=True))
        # compute the accuracy
        pred = output.data.max(1)[1]
        equal_flag_noise = pred.eq(target.data).cuda()
        noise_correct += equal_flag_noise.sum()
        
        for i in range(data.size(0)):
            if equal_flag[i] == 1 and equal_flag_noise[i] == 1 and equal_flag_adv[i] == 0:
                selected_list.append(selected_index)
            selected_index += 1
            
        total += data.size(0)

    selected_list = torch.LongTensor(selected_list).cuda()
    clean_data_tot = torch.index_select(clean_data_tot, 0, selected_list)
    adv_data_tot = torch.index_select(adv_data_tot, 0, selected_list)
    noisy_data_tot = torch.index_select(noisy_data_tot, 0, selected_list)
    label_tot = torch.index_select(label_tot, 0, selected_list)

    torch.save(clean_data_tot, '%s/clean_data_%s_%s_%s.pth' % (args.outf, args.net_type, args.dataset, args.adv_type))
    torch.save(adv_data_tot, '%s/adv_data_%s_%s_%s.pth' % (args.outf, args.net_type, args.dataset, args.adv_type))
    torch.save(noisy_data_tot, '%s/noisy_data_%s_%s_%s.pth' % (args.outf, args.net_type, args.dataset, args.adv_type))
    torch.save(label_tot, '%s/label_%s_%s_%s.pth' % (args.outf, args.net_type, args.dataset, args.adv_type))

    print('Adversarial Noise:({:.2f})\n'.format(generated_noise / total))
    print('Final Accuracy: {}/{} ({:.2f}%)\n'.format(correct, total, 100. * correct / total))
    print('Adversarial Accuracy: {}/{} ({:.2f}%)\n'.format(adv_correct, total, 100. * adv_correct / total))
    print('Noisy Accuracy: {}/{} ({:.2f}%)\n'.format(noise_correct, total, 100. * noise_correct / total))
    
if __name__ == '__main__':
    main()
