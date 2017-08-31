#!/usr/bin/env sh
# 深度遍历
deepls(){
for x in "$1/"*
do 
#echo $x
if [ -f $x ]
then
echo $x $I|cut -d '/' -f3-4 >> $NAME
fi
if [ -d $x ]
then
(deepls "$x")
I=`expr $I + 1`
fi
done
}
I=0
DEST_PATH="./train"
NAME="./train.txt"
deepls $DEST_PATH
