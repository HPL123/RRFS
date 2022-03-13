# Code for Robust-Region-Feature-Synthesizer-for-Zero-Shot-Object-Detection
### Environment requirements
#### - [mmdetection](http://github.com/open-mmlab/mmdetection) we recommend using [Docker 2.0](Docker.md). 
#### - The code implementation of our experiments mainly based on [PyTorch 1.1.0](https://pytorch.org/) and Python 3.6.
The following scripts are for different steps in the pipeline on PASCAL VOC dataset, please see the respective files for more arguments. Before running the scripts, please set the datasets and backbone paths in the config files. Weights of [ResNet101](https://drive.google.com/file/d/1g3UXPw-_K3na7acQGZlhjgQPjXz_FNnX/view?usp=sharing) trained excluding overlapping unseen classes from ImageNet.

### 1. Train object detector on seen classes
       cd mmdetection
       ./tools/dist_train.sh configs/pascal_voc/faster_rcnn_r101_fpn_1x_voc0712.py
 

### 2. Extract features
       # extract seen classes features to train synthesizer and unseen class features for cross validation
       
       cd mmdetection
       
       # extract training features for seen classes
       python tools/zero_shot_utils.py configs/pascal_voc/faster_rcnn_r101_fpn_1x_voc0712.py --classes seen --load_from ./work_dir/voc0712/epoch_4.pth --save_dir ../../data/voc --data_split train
       
       # extract training features for unseen classes
       python tools/zero_shot_utils.py configs/pascal_voc/faster_rcnn_r101_fpn_1x_voc0712.py --classes unseen --load_from ./work_dir/voc0712/epoch_4.pth --save_dir ../../data/voc --data_split test
       
 ### 3. Train synthesizer
      # modify the paths to extracted features, labels and model checkpoints.
      ./script/train_synthesizer.sh
      
 ### 4. Test 
 
       cd mmdetection
       
       ## C setting
       ./tools/dist_test.sh configs/pascal_voc/faster_rcnn_r101_fpn_1x_voc0712.py /work_dir/voc0712/epoch_4.pth --dataset voc --out voc_results.pkl --zsd --syn_weights ../checkpoints/VOC/classifier_best.pth
       
       ## G setting
       ./tools/dist_test.sh configs/pascal_voc/faster_rcnn_r101_fpn_1x_voc0712.py /work_dir/voc0712/epoch_4.pth --dataset voc --out voc_results.pkl --gzsd --syn_weights ../checkpoints/VOC/classifier_best.pth
      
 ### Citation
 If you think our work is helpful, please cite
        @article{huang2022robust,
          title={Robust Region Feature Synthesizer for Zero-Shot Object Detection},
          author={Huang, Peiliang and Han, Junwei and Cheng, De and Zhang, Dingwen},
          journal={arXiv preprint arXiv:2201.00103},
          year={2022}
         }
       
