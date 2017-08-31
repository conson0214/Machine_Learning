#!/usr/bin/env sh
EXAMPLE=D:/CaffeLearn/fine-tune/example
DATA=D:/CaffeLearn/fine-tune/example/data
TOOLS=D:/Caffe/caffe-master/Build/x64/Release

$TOOLS/compute_image_mean $EXAMPLE/ilsvrc12_train_lmdb $DATA/imagenet_mean.binaryproto

echo "Done."