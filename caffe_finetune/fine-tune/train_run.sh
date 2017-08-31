#!/usr/bin/env sh

EXAMPLE=D:/CaffeLearn/fine-tune/example
DATA=D:/CaffeLearn/fine-tune/example/data
TOOLS=D:/Caffe/caffe-master/Build/x64/Release
TRAINCONFIG=D:/CaffeLearn/fine-tune/cifar10_test

$TOOLS/caffe train --solver=$TRAINCONFIG/cifar10_quick_solver.prototxt
