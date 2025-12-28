"""
Created on Sun Oct 25 2018
@author: Kimin Lee
"""
from __future__ import print_function
import numpy as np
import os
import lib_regression
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from sklearn.linear_model import LogisticRegressionCV

# This code infers the adv/normal from the mahalanobis distance from each layer and class
# This uses something equivalent to a single softmax layer (but is called logistic regression = lr)

parser = argparse.ArgumentParser(description='PyTorch code: Mahalanobis detector')
parser.add_argument('--net_type', default="densenet", help='resnet | densenet')
args = parser.parse_args()
print(args)

class RPSigmoid(torch.nn.Module):
    def __init__(self, lenX):
        super(RPSigmoid, self).__init__()  # changed
        self.paramA = torch.nn.Parameter(torch.Tensor(lenX))
        self.paramA.data.uniform_(2.2,2.5)
        self.paramB = torch.nn.Parameter(torch.Tensor(lenX))
        self.paramB.data.uniform_(1.2, 1.5)
        self.paramC = torch.nn.Parameter(torch.Tensor(lenX))
        self.paramC.data.uniform_(2.2, 2.5)
        self.paramD = torch.nn.Parameter(torch.Tensor(lenX))
        self.paramD.data.uniform_(0, 0.1)

    def forward(self, x):
        res = torch.Tensor(x.size()[0], x.size()[1])
        for i in range(x.size()[0]):
            res[i] = (self.paramA* self.paramB* x[i])/pow((1+pow(abs(self.paramB*x[i]), self.paramC)), 1/self.paramC) + self.paramD*x[i]
        
        # res = torch.Tensor(x.size()[1], x.size()[0])
        # for i in range(x.size()[1]):
        #     res[i] = (self.paramA[i]*self.paramB[i]*x)/pow((1+pow(abs(self.paramB[i]*x), self.paramC[i])), 1/self.paramC[i])+self.paramD[i]*x
        
        return res


class predNet(torch.nn.Module):
    def __init__(self, inSize, outSize):
        super(predNet, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(inSize, inSize),
            RPSigmoid(inSize), # set of parameters per neuron
            # nn.ReLU(),
            # nn.LeakyReLU(),
            nn.Linear(inSize, outSize)
        )
    
    def forward(self, x):
        # print(x)
        res = self.net(x.float())
        return res
        
    def predict_proba(self, x):
        return F.sigmoid(self.forward(torch.from_numpy(x)))


def main():
    # initial setup
  #  dataset_list = ['cifar10', 'cifar100', 'svhn']
    dataset_list = ['svhn']
#   adv_test_list = ['FGSM', 'BIM', 'DeepFool', 'CWL2']
    adv_test_list = ['CWL2']
    if (dataset_list[0]=="cifar100"):
        outSize = 100
    else:
        outSize = 10

    print('evaluate the LID estimator')
    score_list = ['LID_10', 'LID_20', 'LID_30', 'LID_40', 'LID_50', 'LID_60', 'LID_70', 'LID_80', 'LID_90']
    list_best_results, list_best_results_index = [], []
    for dataset in dataset_list:
        print('load train data: ', dataset)
        outf = './adv_output/' + args.net_type + '_' + dataset + '/'

        list_best_results_out, list_best_results_index_out = [], []
        for out in adv_test_list:
            best_auroc, best_result, best_index = 0, 0, 0
            for score in score_list:
                print('load train data: ', out, ' of ', score)
                total_X, total_Y = lib_regression.load_characteristics(score, dataset, out, outf)
                X_val, Y_val, X_test, Y_test = lib_regression.block_split_adv(total_X, total_Y)
                pivot = int(X_val.shape[0] / 6)
                X_train = np.concatenate((X_val[:pivot], X_val[2*pivot:3*pivot], X_val[4*pivot:5*pivot]))
                Y_train = np.concatenate((Y_val[:pivot], Y_val[2*pivot:3*pivot], Y_val[4*pivot:5*pivot]))
                # print(X_train.shape)
                X_val_for_test = np.concatenate((X_val[pivot:2*pivot], X_val[3*pivot:4*pivot], X_val[5*pivot:]))
                Y_val_for_test = np.concatenate((Y_val[pivot:2*pivot], Y_val[3*pivot:4*pivot], Y_val[5*pivot:]))
                print(X_train)
                print(Y_train)
                lr = LogisticRegressionCV(n_jobs=-1, max_iter=1000000).fit(X_train, Y_train) # original
                # pred_net = predNet(len(X_train[0]), 1)
                # optimizer = optim.Adam(pred_net.parameters(), lr=0.01, eps = 1e-7)
                # criterion = nn.BCEWithLogitsLoss()
                # for epoch in range(100):
                #     for i in range(0, len(X_train), 100):
                #         # print(torch.from_numpy(X_train[i:i+100]).size())
                #         predOut = pred_net(torch.from_numpy(X_train[i:i+100]))
                #         # out = out.reshape(1, -1)
                #         target = torch.tensor(Y_train[i:i+100]).to(predOut.device)[:,None]
                #         lossX = criterion(predOut, target)
                #         lossX.backward()
                #         optimizer.step()
                #         optimizer.zero_grad()
                

                # y_pred = lr.predict_proba(X_train)[:, 1]
                # y_pred = pred_net(torch.from_numpy(X_train))[:, 0]
                #print('training mse: {:.4f}'.format(np.mean(y_pred - Y_train)))
                # y_pred = pred_net(torch.from_numpy(X_val_for_test))[:, 0]
                #print('test mse: {:.4f}'.format(np.mean(y_pred - Y_val_for_test)))
                results = lib_regression.detection_performance(lr, X_val_for_test, Y_val_for_test, outf)

                if best_auroc < results['TMP']['AUROC']:
                    best_auroc = results['TMP']['AUROC']
                    best_index = score
                    best_result = lib_regression.detection_performance(lr, X_test, Y_test, outf)
            list_best_results_out.append(best_result)
            list_best_results_index_out.append(best_index)
        list_best_results.append(list_best_results_out)
        list_best_results_index.append(list_best_results_index_out)
        
    print('evaluate the Mahalanobis estimator')
    score_list = ['Mahalanobis_0.0', 'Mahalanobis_0.01', 'Mahalanobis_0.005', \
                  'Mahalanobis_0.002', 'Mahalanobis_0.0014', 'Mahalanobis_0.001', 'Mahalanobis_0.0005']
    list_best_results_ours, list_best_results_index_ours = [], []
    for dataset in dataset_list:
        print('load train data: ', dataset)
        outf = './adv_output/' + args.net_type + '_' + dataset + '/'
        list_best_results_out, list_best_results_index_out = [], []
        for out in adv_test_list:
            best_auroc, best_result, best_index = 0, 0, 0
            for score in score_list:
                print('load train data: ', out, ' of ', score)
                total_X, total_Y = lib_regression.load_characteristics(score, dataset, out, outf)
                X_val, Y_val, X_test, Y_test = lib_regression.block_split_adv(total_X, total_Y)
                pivot = int(X_val.shape[0] / 6)
                X_train = np.concatenate((X_val[:pivot], X_val[2*pivot:3*pivot], X_val[4*pivot:5*pivot]))
                Y_train = np.concatenate((Y_val[:pivot], Y_val[2*pivot:3*pivot], Y_val[4*pivot:5*pivot]))
                X_val_for_test = np.concatenate((X_val[pivot:2*pivot], X_val[3*pivot:4*pivot], X_val[5*pivot:]))
                Y_val_for_test = np.concatenate((Y_val[pivot:2*pivot], Y_val[3*pivot:4*pivot], Y_val[5*pivot:]))
                lr = LogisticRegressionCV(n_jobs=-1).fit(X_train, Y_train) # -1 means as deduced by the code


                # # for RPSigmoid
                # pred_net = predNet(len(X_train[0]), 1)
                # optimizer = optim.Adam(pred_net.parameters(), lr=0.01, eps = 1e-7)
                # criterion = nn.BCEWithLogitsLoss()
                # for epoch in range(100):
                #     for i in range(0, len(X_train), 100):
                #         # print(torch.from_numpy(X_train[i:i+100]).size())
                #         predOut = pred_net(torch.from_numpy(X_train[i:i+100]))
                #         # out = out.reshape(1, -1)
                #         target = torch.tensor(Y_train[i:i+100]).to(predOut.device)[:,None]
                #         lossX = criterion(predOut, target)
                #         lossX.backward()
                #         optimizer.step()
                #         optimizer.zero_grad()



                # y_pred = pred_net.predict_proba(X_train)[:, 0]
                # print('training mse: {:.4f}'.format(np.mean(y_pred - Y_train)))
                # y_pred = lr.predict_proba(X_val_for_test)[:, 0]
                # print('test mse: {:.4f}'.format(np.mean(y_pred - Y_val_for_test)))
                results = lib_regression.detection_performance(lr, X_val_for_test, Y_val_for_test, outf)
                if best_auroc < results['TMP']['AUROC']:
                    best_auroc = results['TMP']['AUROC']
                    best_index = score
                    best_result = lib_regression.detection_performance(lr, X_test, Y_test, outf)
            list_best_results_out.append(best_result)
            list_best_results_index_out.append(best_index)
        list_best_results_ours.append(list_best_results_out)
        list_best_results_index_ours.append(list_best_results_index_out)

    count_in = 0
    mtypes = ['TNR', 'AUROC', 'DTACC', 'AUIN', 'AUOUT']
    print("results of LID")
    for in_list in list_best_results:
        print('in_distribution: ' + dataset_list[count_in] + '==========')
        count_out = 0
        for results in in_list:
            print('out_distribution: '+ adv_test_list[count_out])
            for mtype in mtypes:
                print(' {mtype:6s}'.format(mtype=mtype), end='')
            print('\n{val:6.2f}'.format(val=100.*results['TMP']['TNR']), end='')
            print(' {val:6.2f}'.format(val=100.*results['TMP']['AUROC']), end='')
            print(' {val:6.2f}'.format(val=100.*results['TMP']['DTACC']), end='')
            print(' {val:6.2f}'.format(val=100.*results['TMP']['AUIN']), end='')
            print(' {val:6.2f}\n'.format(val=100.*results['TMP']['AUOUT']), end='')
            print('Input noise: ' + list_best_results_index[count_in][count_out])
            print('')
            count_out += 1
        count_in += 1
        
    count_in = 0
    print("results of Mahalanobis")
    for in_list in list_best_results_ours:
        print('in_distribution: ' + dataset_list[count_in] + '==========')
        count_out = 0
        for results in in_list:
            print('out_distribution: '+ adv_test_list[count_out])
            for mtype in mtypes:
                print(' {mtype:6s}'.format(mtype=mtype), end='')
            print('\n{val:6.2f}'.format(val=100.*results['TMP']['TNR']), end='')
            print(' {val:6.2f}'.format(val=100.*results['TMP']['AUROC']), end='')
            print(' {val:6.2f}'.format(val=100.*results['TMP']['DTACC']), end='')
            print(' {val:6.2f}'.format(val=100.*results['TMP']['AUIN']), end='')
            print(' {val:6.2f}\n'.format(val=100.*results['TMP']['AUOUT']), end='')
            print('Input noise: ' + list_best_results_index_ours[count_in][count_out])
            print('')
            count_out += 1
        count_in += 1
    
if __name__ == '__main__':
    main()
