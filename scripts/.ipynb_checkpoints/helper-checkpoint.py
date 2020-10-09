import torch
import numpy as np




def coco80_to_coco91_class(label):
    x= [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 27, 28, 31, 32, 33, 34,
         35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63,
         64, 65, 67, 70, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 84, 85, 86, 87, 88, 89, 90]
    
    return x[int(label)]



def get_idf(gt,mask):
    corpus_length=sum(mask)
#     tf=torch.tensor([1/mask[i] for i in range(len(mask)) for j in range(mask[i])]).cuda()
    tf=1
    idf=gt[:,5:].sum(axis=0).cuda()

    idf=torch.log(corpus_length/idf)
    
    classes=gt[:,5:].max(1)[1]
    tfidf=tf*idf[classes]
    tfidf=torch.softmax(tfidf,dim=0)
    
    return tfidf.unsqueeze(1)

def get_precomputed_idf(obj_idf,col_name):
    
    idf=np.array(obj_idf[col_name])
    idf=torch.tensor(idf,device='cuda',dtype=torch.float)
    idf=-torch.log(idf)
    
    
    return idf

def get_location_weights(gt_boxes):
    area=gt_boxes[:,2]*gt_boxes[:,3]
    weights=-torch.log(area)
    return torch.tensor(weights,device='cuda')


def dic2tensor(targets,key):
    
    tensor=torch.cat([t[key] for t in targets],dim=0)
    
    return tensor

def get_progress_stats(true_pred,no_obj,iou_list,targets):
    '''
    this function takes the Tranformed true prediction and the IoU list between the true prediction and the Gt boxes.
    Based on the IoU list it will calculate the mean IoU, the mean Positive Classification,  the mean negative Classification,
    the mean Positive objectness and the mean negative objectness.
    INPUTS: True_pred = Tensor:[N,BBs,4+1+C]
            no_obj_conf= Tensor [K]
            Targets = List of DICTs(N elements-> Key:Tenor[M,x]) , where M is the number of objects for that N, x depends on key.
            IoU List= List(N elements->Tensors[M,BBs])
    Outputs:DICT:{floats: avg_pos_conf, avg_neg_conf, avg_pos_class, avg_neg_class, avg_iou}  
    '''

    resp_true_pred=[]
    best_iou=[]
    no_obj2=no_obj.clone().detach()
    labels=dic2tensor(targets,'labels')
    
    for i in range(len(iou_list)):
        best_iou_positions=iou_list[i].max(axis=1)[1]
        best_iou.append(iou_list[i].max(axis=1)[0])
        
        resp_true_pred.append(true_pred[i,:,:][best_iou_positions])
        
    resp_true_pred=torch.cat(resp_true_pred,dim=0)
    best_iou=torch.cat(best_iou,dim=0)
    
    
    nb_digits = resp_true_pred.shape[1]-5 # get the number of CLasses
    n_obj=resp_true_pred.shape[0]
    y = labels.view(-1,1)
    y_onehot = torch.BoolTensor(n_obj, nb_digits)
    y_onehot.zero_()
    y_onehot.scatter_(1, y, 1)
    
    pos_class=resp_true_pred[:,5:][y_onehot].mean().item()
    neg_class=resp_true_pred[:,5:][~y_onehot].mean().item()
    
    pos_conf=resp_true_pred[:,4].mean().item()
    avg_iou=best_iou.mean().item()
    
    neg_conf=torch.sigmoid(no_obj2).mean().item()
    
    return  {'iou':avg_iou,
             'pos_conf':pos_conf,
             'neg_conf':neg_conf,
             'pos_class':pos_class,
             'neg_class':neg_class}
    


def collate_fn(batch):
    pictures=[i[0] for i in batch]
    pictures=torch.cat(pictures, dim=0)
    
    targets=[i[1] for i in batch]
        
    return pictures,targets


def convert2_abs_xyxy(bboxes,shape,inp_dim=1):
        
    (h,w,c)=shape
    h=h/inp_dim
    w=w/inp_dim
    
    
    xmin=bboxes[:,0]*w
    ymin=bboxes[:,1]*h
    width=bboxes[:,2]*w
    height=bboxes[:,3]*h
    xmin=xmin-width/2
    ymin=ymin-height/2
    xmax=xmin+width
    ymax=ymin+height
    
    if (type(bboxes) is torch.Tensor):
        return torch.stack((xmin, ymin, xmax, ymax)).T
    else:
        return np.stack((xmin, ymin, xmax, ymax)).T



def convert2_rel_xcycwh(bboxes,shape):
        
    (h,w,c)=shape
    
    xmin=bboxes[:,0]/w
    ymin=bboxes[:,1]/h
    xmax=bboxes[:,2]/w
    ymax=bboxes[:,3]/h
        
    xc = (xmax- xmin)/2 +xmin
    yc = (ymax- ymin)/2 +ymin

    width = (xmax- xmin)
    height = (ymax- ymin)
    
    if (type(bboxes) is torch.Tensor):
        return torch.stack((xc, yc, width, height)).T
    else:
        return np.stack((xc, yc, width, height)).T
        

