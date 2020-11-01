# yolo_tfidf

To start using this code clone this repo and run get_coco2017.sh.

To use the libraries either create a similar conda environmet with the specs_file or 
use the basic libraries like:

* torch 
* cv2
* tensorboard for pytorch
* numpy
* pandas
* scipy

Additional important libraries:

* conda install -c conda-forge imgaug
* conda install -c conda-forge bayesian-optimization
* conda install -c conda-forge pycocotools


To use this repo with skeletor on should install the package:
* pip install skeletor-ml 

Skeletor uses persistent logging, which menas that it loggs every hyperparameter of the model for every epoch and makes it easy to keep track of experiments.
To use one can simply run for example:

OMP_NUM_THREADS=1 python train_with_skeletor.py  --epochs=15 --batch_size=30 --iou_ignore_thresh=0.5 --weight_decay=0.005 --momentum=0.8 --gamma=2 --alpha=0.1  --lr=0.0001  --lcoord=2 --lno_obj=0.25 --iou_type=001 <experiment_name> 

To use the pretrain model save the yolov3 weights inside top directory:
wget https://pjreddie.com/media/files/yolov3.weights

For more info about arguments see script train_with_skeletor lines 29-99

This YOLO implementation follows the tutorial of Paperspace:

https://blog.paperspace.com/how-to-implement-a-yolo-object-detector-in-pytorch/