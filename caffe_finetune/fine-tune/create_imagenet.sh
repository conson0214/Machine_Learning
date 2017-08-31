#!/usr/bin/env sh
EXAMPLE=D:/CaffeLearn/fine-tune/example
DATA=D:/CaffeLearn/fine-tune/example/data
TOOLS=D:/Caffe/caffe-master/Build/x64/Release

TRAIN_DATA_ROOT=D:/CaffeLearn/fine-tune/example/data/train/
VAL_DATA_ROOT=D:/CaffeLearn/fine-tune/example/data/val/

# 这里我们打开resize，需要把所有图片尺寸统一
RESIZE=true
if $RESIZE; then
  RESIZE_HEIGHT=256
  RESIZE_WIDTH=256
else
  RESIZE_HEIGHT=0
  RESIZE_WIDTH=0
fi

.......

echo "Creating train lmdb..."

GLOG_logtostderr=1 $TOOLS/convert_imageset.exe \
    --resize_height=$RESIZE_HEIGHT \
    --resize_width=$RESIZE_WIDTH \
    --shuffle \
    $TRAIN_DATA_ROOT \
    $DATA/train.txt \
    $EXAMPLE/ilsvrc12_train_lmdb
	#生成的lmdb路径
	
echo "Creating val lmdb..."

GLOG_logtostderr=1 $TOOLS/convert_imageset.exe \
    --resize_height=$RESIZE_HEIGHT \
    --resize_width=$RESIZE_WIDTH \
    --shuffle \
    $VAL_DATA_ROOT \
    $DATA/val.txt \
    $EXAMPLE/ilsvrc12_val_lmdb
	#生成的lmdb路径
	
echo "Done."

sleep 10