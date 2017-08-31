#!/usr/bin/env sh
# 深度遍历
deepls(){
for x in "$1/"*
do 
echo $x
if [ -f $x ]
then
echo $x $I|cut -d '/' -f5-6 >> $NAME
fi
if [ -d $x ]
then
I= 0
(deepls "$x")
fi
done
}
I=0
DEST_PATH="./val"
NAME="./val.txt"
for x in `ls $DEST_PATH -p | grep /`
do

echo $x
#if [ -d $x ]
#then
echo $DEST_PATH"/"$x
deepls $DEST_PATH"/"$x
#fi
I=`expr $I + 1`
done

