import pandas as pd
import torch
from darknet import *
import torch.optim as optim
import sys
import pandas as pd
import torch.nn as nn
import os


def init_model(hyperparameters,mode,show=True):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    net = Darknet("../cfg/yolov3.cfg",multi_scale=mode['multi_scale'])

    '''
    when loading weights from dataparallel model then, you first need to instatiate the dataparallel model 
    if you start fresh then first model.load_weights and then make it parallel
    '''
    
    if hyperparameters['optimizer']=='sgd':
        optimizer = optim.SGD(net.parameters(), lr=hyperparameters['lr'], weight_decay=hyperparameters['weight_decay'], momentum=hyperparameters['momentum'])
    elif hyperparameters['optimizer']=='adam':
        optimizer = optim.Adam(net.parameters(), lr=hyperparameters['lr'], weight_decay=hyperparameters['weight_decay'])
    
    try:
        PATH = '../checkpoints/'+hyperparameters['path']+'/'
        checkpoint = torch.load(PATH+hyperparameters['path']+'.tar')
        if mode['bayes_opt']==False:
            try:
                hyperparameters=checkpoint['hyperparameters']
            except:
                pass
                        # Assuming that we https://pytorch.org/docs/stable/data.html#torch.utils.data.Datasetare on a CUDA machine, this should print a CUDA device:
        net.to(device)
        net.load_state_dict(checkpoint['model_state_dict'])

        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        hyperparameters['resume_from']=checkpoint['epoch']

    except FileNotFoundError:
        if (hyperparameters['pretrained']==True):
            print("WARNING FILE NOT FOUND INSTEAD USING OFFICIAL PRETRAINED")
            net.load_weights("../yolov3.weights")
        try:
            PATH = '../checkpoints/'+hyperparameters['path']+'/'
            os.mkdir(PATH)
        except FileExistsError:
            pass
        
        
    net.to(device)
    if (torch.cuda.device_count() > 1)&(mode['multi_gpu']==True):
        model = nn.DataParallel(net)
    else:
        model=net
        hyperparameters['batch_size']=8
    model.to(device)
    
    if show==True:
        print(hyperparameters)
    
    if isinstance(hyperparameters['idf_weights'],pd.DataFrame)==False:
        if (hyperparameters['idf_weights']==True):
            hyperparameters['idf_weights']=pd.read_csv('../idf.csv')
        else:
            hyperparameters['idf_weights']=False
    
    
            
    return model,optimizer,hyperparameters