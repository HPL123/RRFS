    FROM nasir6/mmdetection:base
    RUN echo "$PWD"
    ADD . /home
    RUN ls home
    WORKDIR /home/mmdetection
    RUN conda env update -f environment.yml
    RUN python setup.py develop
    CMD bash

python tools/zero_shot_utils.py configs/faster_rcnn_r101_fpn_1x.py --classes [seen, unseen] --load_from [detector checkpoint path] --save_dir [path to save features] --data_split [train, test]
python tools/zero_shot_utils.py configs/faster_rcnn_r101_fpn_1x.py --classes seen --load_from ./work_dir/coco2014/epoch_12.pth --save_dir ../../data/coco --data_split train
python tools/zero_shot_utils.py configs/faster_rcnn_r101_fpn_1x.py --classes unseen --load_from ./work_dir/coco2014/epoch_12.pth --save_dir ../../data/coco --data_split test
