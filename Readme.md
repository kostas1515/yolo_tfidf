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
* pip install pip install skeletor-ml 

Skeletor uses persistent logging, which menas that it loggs every parameter of the model for every epochs and makes easy to keep track of experiments.
To use one can simply run for example:

OMP_NUM_THREADS=1 python train_with_skeletor.py  --epochs=15 --batch_size=30 --iou_ignore_thresh=0.5 --weight_decay=0.005 --momentum=0.8 --gamma=2 --alpha=0.1  --lr=0.0001  --lcoord=2 --lno_obj=0.25 --iou_type=001 <experiment_name> 