#!/usr/bin/env sh

EXAMPLE=D:/CaffeLearn/fine-tune/example
DATA=D:/CaffeLearn/fine-tune/example/data
TOOLS=D:/Caffe/caffe-master/Build/x64/Release
TRAINCONFIG=D:/CaffeLearn/fine-tune/finetune_model

$TOOLS/caffe train --solver=$TRAINCONFIG/solver.prototxt --weights $TRAINCONFIG/bvlc_reference_caffenet.caffemodel