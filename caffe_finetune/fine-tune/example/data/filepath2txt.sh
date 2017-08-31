#!/usr/bin/env sh
# 说明 
# dictionary： train{d1{f1,f2,...},d2{f1,f2,...},...} val{d1{f11,f12,...},d2{f21,f22,...},.... f11,f12,.....,f21,f2,...,....}
# train
#    -d1
#       f1.jpeg
#       f2.jpeg
#       f3.jpeg
#       f4.jpeg
#       ....
#    -d2
#       f1.jpeg
#       f2.jpeg
#       f3.jpeg
#       f4.jpeg
#       ....
#    ...
# val
#    -d1
#    -d2
#    ...
#    all images in di need copy to val/ again.
rm -f train.txt
rm -f val.txt
echo "filepath2traintxt"
sh ./filepath2traintxt.sh
echo "Done"
echo "filepath2valtxt"
sh ./filepath2valtxt2.sh
echo "Done"
