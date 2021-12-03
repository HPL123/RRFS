### Pull docker image for mmdetection

#### Pull the base Image 
```sh
docker pull nasir6/mmdetection:latest
```

#### To run 

```sh

# replace codes with path to the code directory

docker run -p 3000:3000 -v codes/:/codes -it --runtime=nvidia --rm nasir6/mmdetection:latest

cd /codes/zero_shot_detection/mmdetection

# install the mmdetection library

python setup.py develop

# to test the synthesized classifier on MSCOCO. 

test

./tools/dist_test.sh configs/faster_rcnn_r101_fpn_1x.py work_dir/coco2014/epoch_12.pth 8 --dataset coco --out coco_results.pkl --zsd --syn_weights ../checkpoints/coco_65_15/classifier_best_137.pth

```

#### Run Jupyter notebook

```ssh
ssh -L 3000:localhost:3000 ubuntu@[server-ip]
# run container bash and start jupyter notebook
nohup jupyter notebook --ip 0.0.0.0 --port 3000 --no-browser --allow-root &

```
