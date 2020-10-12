""" Main file to orchestrate model training! Most of the work should go here."""

import os
import time
from darknet import *
import numpy as np
import pandas as pd
import torch
import util as util
import torch.optim as optim
import track
import init_model
from dataset import *
import helper as helper
import coco_utils
import coco_eval
import torchvision.ops.boxes as nms_box
import sys,os

import skeletor
from skeletor.datasets import build_dataset, num_classes
from skeletor.models import build_model
from skeletor.optimizers import build_optimizer
from skeletor.utils import AverageMeter, accuracy, progress_bar


def add_train_args(parser):
    # Main arguments go here
    parser.add_argument('--lr', default=.1, type=float)
    parser.add_argument('--batch_size', default=8, type=int)
    
    parser.add_argument('--eval_batch_size', default=1, type=int)
    
    parser.add_argument('--inf_confidence', default=0.01, type=float)
    
    parser.add_argument('--inf_iou_threshold', default=0.5, type=float)
    
    parser.add_argument('--epochs', default=3, type=int)
    
    parser.add_argument('--gamma', type=float, default=0.0,
                        help='gamma for focal loss')
    
    parser.add_argument('--alpha', type=float, default=0.75,
                        help='alpha forfocal loss')
    
    parser.add_argument('--momentum', default=.9, type=float,
                        help='SGD momentum')
    
    parser.add_argument('--lcoord', default=5.0, type=float,
                        help='coord loss gain')
    
    parser.add_argument('--lno_obj', default=0.5, type=float,
                        help='negative confidence loss gain')
    
    parser.add_argument('--weight_decay', default=5e-4, type=float,
                        help='weight decay')
    
    parser.add_argument('--coco_version', default='2017', type=str,
                        help='Dataset Partition')
    
    parser.add_argument('--optimizer', default='sgd', type=str,
                        help='either sgd or adam')
    
    parser.add_argument('--tfidf', default=False, type=bool)
    
    parser.add_argument('--wasserstein', default=False, type=bool,
                       help='if true instead of BCE it will compute wasserstein distance for confidence loss')
    
    parser.add_argument('--augment', default=0, type=int,
                       help='either 0/1 if 1 then it augments bbs and images')
    
    parser.add_argument('--is_pretrained', default=False, type=bool,
                       help='if true then depending on path it will either load yolo official weights or load from checkpoint')
    
    parser.add_argument('--iou_type', default=(0,0,0), type=tuple,
                        help='onehot: (0,0,0) -> (GIoU,DIoU,CIoU) default is MSE(0,0,0)')
    
    parser.add_argument('--iou_ignore_thresh', default=0.5, type=float,
                       help='ignore negative examples abou this threshold')
    
    parser.add_argument('--cuda', action='store_true',
                        help='if true, use GPU!')
    
    parser.add_argument('--reduction', default='sum', type=str,
                        help='min batch reduction either sum or mean')
    
    parser.add_argument('--path', help='path to experiment')


def adjust_learning_rate(epoch, optimizer, lr, schedule, decay):
    if epoch in schedule:
        new_lr = lr * decay
        for param_group in optimizer.param_groups:
            param_group['lr'] = new_lr
    else:
        new_lr = lr
    return new_lr


def train(trainloader, model, optimizer, epoch, cuda=True):
    # switch to train mode
    model.train()
    hyperparameters=model.hp
    
    if type(model) is nn.DataParallel:
        inp_dim=model.module.inp_dim
        pw_ph=model.module.pw_ph
        cx_cy=model.module.cx_cy
        stride=model.module.stride
    else:
        inp_dim=model.inp_dim
        pw_ph=model.pw_ph
        cx_cy=model.cx_cy
        stride=model.stride
        
    if cuda:
        pw_ph=pw_ph.cuda()
        cx_cy=cx_cy.cuda()
        stride=stride.cuda()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    avg_loss = AverageMeter()
    avg_iou = AverageMeter()
    avg_conf = AverageMeter()
    avg_no_conf = AverageMeter()
    avg_pos = AverageMeter()
    avg_neg = AverageMeter()
    end = time.time()
    break_flag=0

    for batch_idx, (inputs, targets) in enumerate(trainloader):
        # measure data loading time
        data_time.update(time.time() - end)

        if cuda:
            inputs = inputs.cuda()

        # compute output
        raw_pred = model(inputs,torch.cuda.is_available())
        true_pred=util.transform(raw_pred.clone().detach(),pw_ph,cx_cy,stride)
        iou_list=util.get_iou_list(true_pred, targets,hyperparameters,inp_dim)
        
        resp_raw_pred,resp_cx_cy,resp_pw_ph,resp_stride,no_obj=util.build_tensors(raw_pred,iou_list,pw_ph,cx_cy,stride,hyperparameters)
        
        stats=helper.get_progress_stats(true_pred,no_obj,iou_list,targets)
        if hyperparameters['wasserstein']==True:
            no_obj=util.get_wasserstein_matrices(raw_pred,iou_list,inp_dim)
            
        try:
            loss=util.yolo_loss(resp_raw_pred,targets,no_obj,resp_pw_ph,resp_cx_cy,resp_stride,inp_dim,hyperparameters)
        except RuntimeError:
#                 print('bayes opt failed')
            break_flag=1
            break
        loss=util.yolo_loss(resp_raw_pred,targets,no_obj,resp_pw_ph,resp_cx_cy,resp_stride,inp_dim,hyperparameters)

        # measure accuracy and record loss
        avg_loss.update(loss.item())
        avg_iou.update(stats['iou'])
        avg_conf.update(stats['pos_conf'])
        avg_no_conf.update(stats['neg_conf'])
        avg_pos.update(stats['pos_class'])
        avg_neg.update(stats['neg_class'])



        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # plot progress
        progress_str = 'Loss: %.4f | AvIoU: %.3f | AvPConf: %.3f | AvNConf: %.5f | AvClass: %.3f | AvNClass: %.5f'\
            % (loss.item(), stats['iou'], stats['pos_conf'], stats['neg_conf'],stats['pos_class'],stats['neg_class'])
        
        progress_bar(batch_idx, len(trainloader), progress_str)

        iteration = epoch * len(trainloader) + batch_idx
        
    track.metric(iteration=iteration, epoch=epoch,
                 avg_train_loss=avg_loss.avg,
                 avg_train_iou=avg_iou.avg,
                 avg_train_conf=avg_conf.avg,
                 avg_train_neg_conf=avg_no_conf.avg,
                 avg_train_pos=avg_pos.avg,
                 avg_train_neg=avg_neg.avg)
        
        
    outcome = {
    'avg_loss': avg_loss.avg,
    'avg_iou': avg_iou.avg,
    'avg_pos': avg_pos.avg,
    'avg_neg':avg_neg.avg,
    'avg_conf': avg_conf.avg,
    'avg_no_conf': avg_no_conf.avg,
    'broken': break_flag
    }
        
    return outcome


def test(testloader, model,epoch, device):
    
    n_threads = torch.get_num_threads()
    # FIXME remove this and make paste_masks_in_image run on the GPU
    torch.set_num_threads(1)
    cpu_device = torch.device("cpu")

    device = device
    
    batch_time = AverageMeter()
    data_time = AverageMeter()
    
    
    hyperparameters=model.hp
    confidence=hyperparameters['inf_confidence']
    iou_threshold=hyperparameters['inf_iou_threshold']
    
    
    if type(model) is nn.DataParallel:
        inp_dim=model.module.inp_dim
        pw_ph=model.module.pw_ph
        cx_cy=model.module.cx_cy
        stride=model.module.stride
    else:
        inp_dim=model.inp_dim
        pw_ph=model.pw_ph
        cx_cy=model.cx_cy
        stride=model.stride
    
    pw_ph=pw_ph.to(device)
    cx_cy=cx_cy.to(device)
    stride=stride.to(device)
    
    sys.stdout = open(os.devnull, 'w') #wrapper to disable hardcoded printing
    coco = coco_utils.get_coco_api_from_dataset(testloader.dataset)
    iou_types = ["bbox"]
    coco_evaluator = coco_eval.CocoEvaluator(coco, iou_types)
    sys.stdout = sys.__stdout__ #wrapper to enable hardcoded printing (return to normal mode)

    # switch to evaluate mode
    model.eval()

    end = time.time()
    with torch.no_grad():
        for batch_idx, (images, targets) in enumerate(testloader):
            # measure data loading time
            data_time.update(time.time() - end)

            images = images.to(device)
        
            targets2=[]
            for t in targets:
                dd={}
                for k, v in t.items():
                    if(k!='img_size'):
                        dd[k]=v.to(device)
                    else:
                        dd[k]=v
                targets2.append(dd)

    #         targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            targets=targets2


            raw_pred = model(images, device)

            true_pred=util.transform(raw_pred.clone().detach(),pw_ph,cx_cy,stride)

            sorted_pred=torch.sort(true_pred[:,:,4]*(true_pred[:,:,5:].max(axis=2)[0]),descending=True)
            pred_mask=sorted_pred[0]>confidence
            indices=[(sorted_pred[1][e,:][pred_mask[e,:]]) for e in range(pred_mask.shape[0])]
            pred_final=[true_pred[i,indices[i],:] for i in range(len(indices))]

            pred_final_coord=[util.get_abs_coord(pred_final[i].unsqueeze(-2)) for i in range(len(pred_final))]

            indices=[nms_box.nms(pred_final_coord[i][0],pred_final[i][:,4],iou_threshold) for i in range(len(pred_final))]
            pred_final=[pred_final[i][indices[i],:] for i in range(len(pred_final))]


            abs_pred_final=[helper.convert2_abs_xyxy(pred_final[i],targets[i]['img_size'],inp_dim) for i in range(len(pred_final))]


            outputs=[dict() for i in range(len((abs_pred_final)))]
            for i,atrbs in enumerate(abs_pred_final):

                outputs[i]['boxes']=atrbs[:,:4]
                outputs[i]['scores']=pred_final[i][:,4]
                try:
                    outputs[i]['labels']=pred_final[i][:,5:].max(axis=1)[1] +1 #could be empty
                except:

                    outputs[i]['labels']=torch.tensor([])


            outputs = [{k: v.to(cpu_device) for k, v in t.items()} for t in outputs]

            res = {target["image_id"].item(): output for target, output in zip(targets, outputs)}
            coco_evaluator.update(res)

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()
            
    sys.stdout = open(os.devnull, 'w') #wrapper to disable hardcoded printing
    
    coco_evaluator.synchronize_between_processes()
    # accumulate predictions from all images
    coco_evaluator.accumulate()
    coco_evaluator.summarize()
    torch.set_num_threads(n_threads)
    
    metrics=coco_evaluator.get_stats()
    
    sys.stdout = sys.__stdout__ #wrapper to enable hardcoded printing (return to normal mode)
    
    
    coco_stats={'map_all':metrics[0],
               'map@0.5':metrics[1],
               'map@0.75':metrics[2],
               'map_small':metrics[3],
               'map_med':metrics[4],
               'map_large':metrics[5],
               'recall@1':metrics[6],
               'recall@10':metrics[7],
               'recall@100':metrics[8],
               'recall@small':metrics[9],
               'recall@medium':metrics[10],
               'recall@large':metrics[11]}
    
    track.metric(iteration=0, epoch=epoch,
                 avg_test_coco_map=metrics[0],
                 coco_stats=coco_stats)
    
    
    return (metrics[0])


def do_training(args):
    
    hyperparameters={'lr': args.lr, 
                 'epochs': args.epochs,
                 'resume_from':0,
                 'coco_version': args.coco_version, #can be either '2014' or '2017'
                 'batch_size':args.batch_size,
                 'weight_decay': args.weight_decay,
                 'momentum': args.momentum, 
                 'optimizer': args.optimizer, 
                 'alpha': args.alpha, 
                 'gamma': args.gamma, 
                 'lcoord': args.lcoord,
                 'lno_obj': args.lno_obj,
                 'iou_type': args.iou_type,
                 'iou_ignore_thresh': args.iou_ignore_thresh, 
                 'tfidf': args.tfidf, 
                 'idf_weights': True, 
                 'tfidf_col_names': ['img_freq', 'none', 'none', 'none', 'no_softmax'],
                 'wasserstein':args.wasserstein,
                 'inf_confidence':args.inf_confidence,
                 'inf_iou_threshold':args.inf_iou_threshold,
                 'augment': args.augment, 
                 'workers': 4,
                 'pretrained':args.is_pretrained,
                 'path': args.path, 
                 'reduction': args.reduction}
    
    mode={'bayes_opt':True,
      'multi_scale':False,
      'debugging':False,
      'show_hp':True,
      'multi_gpu':True,
      'show_temp_summary':False,
      'save_summary': True
     }
    
    coco_version=hyperparameters['coco_version']
    mAP_best=0
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    model,optimizer,hyperparameters,PATH=init_model.init_model(hyperparameters,mode)
    
    
    model.hp=hyperparameters
    
    
    if type(model) is nn.DataParallel:
        inp_dim=model.module.inp_dim
    else:
        inp_dim=model.inp_dim


    if hyperparameters['augment']>0:
        train_dataset=Coco(partition='train',coco_version=coco_version,subset=0.0001,transform=transforms.Compose([Augment(hyperparameters['augment']),
                                                                                                     ResizeToTensor(inp_dim)]))
    else:
        train_dataset=Coco(partition='train',coco_version=coco_version,subset=0.0001,transform=transforms.Compose([ResizeToTensor(inp_dim)]))

    batch_size=hyperparameters['batch_size']

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size,
                                shuffle=True,collate_fn=helper.collate_fn, num_workers=hyperparameters['workers'])
    
     
    test_dataset=Coco(partition='val',coco_version=coco_version,subset=0.001,
                                               transform=transforms.Compose([
                                                ResizeToTensor(inp_dim)
                                               ]))
    
    test_dataloader = DataLoader(test_dataset, batch_size=args.eval_batch_size,
                            shuffle=False,collate_fn=helper.collate_fn, num_workers=4)
    

    # Calculate total number of model parameters
    num_params = sum(p.numel() for p in model.parameters())
    track.metric(iteration=0, num_params=num_params)


    for epoch in range(args.epochs):
        track.debug("Starting epoch %d" % epoch)
#         args.lr = adjust_learning_rate(epoch, optimizer, args.lr, args.schedule,
#                                        args.gamma)
        outcome = train(train_dataloader, model, optimizer, epoch)
        
        
        mAP=test(test_dataloader, model,epoch, device)
        

        track.debug('Finished epoch %d... | train loss %.3f | avg_iou %.3f | avg_conf %.3f | avg_no_conf %.3f'
                    '| avg_pos %.3f | avg_neg %.5f | mAP %.5f'
                    % (epoch, outcome['avg_loss'], outcome['avg_iou'],outcome['avg_conf'],outcome['avg_no_conf'],outcome['avg_pos'],outcome['avg_neg']
                       ,mAP))
        # Save model
        model_fname = os.path.join(track.trial_dir(),
                                   "model{}.tar".format(epoch))
        torch.save({
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'avg_loss': outcome['avg_loss'],
            'avg_iou': outcome['avg_iou'],
            'avg_pos': outcome['avg_pos'],
            'avg_neg':outcome['avg_neg'],
            'avg_conf': outcome['avg_conf'],
            'avg_no_conf': outcome['avg_no_conf'],
            'mAP': mAP,
            'hyperparameters': hyperparameters
            }, model_fname)
        
        if mAP > mAP_best:
            mAP_best = mAP
            best_fname = os.path.join(track.trial_dir(), "best.tar")
            track.debug("New best score! Saving model")
            
            torch.save({
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'avg_loss': outcome['avg_loss'],
            'avg_iou': outcome['avg_iou'],
            'avg_pos': outcome['avg_pos'],
            'avg_neg':outcome['avg_neg'],
            'avg_conf': outcome['avg_conf'],
            'avg_no_conf': outcome['avg_no_conf'],
            'mAP': mAP,
            'hyperparameters': hyperparameters
            }, best_fname)


def postprocess(proj):
    df = skeletor.proc.df_from_proj(proj)
    df.to_csv(os.path.join(proj.log_dir, (proj.log_dir).split('/')[-1]+".csv"))
    if 'avg_test_coco_map' in df.columns:
        best_trial = df[df.avg_test_coco_map == df.avg_test_coco_map.max()]
        best_trial=best_trial.dropna(axis='columns')
        print("Trial with top map:")
        print(best_trial)


if __name__ == '__main__':
    skeletor.supply_args(add_train_args)
    skeletor.supply_postprocess(postprocess, save_proj=True)
    skeletor.execute(do_training)