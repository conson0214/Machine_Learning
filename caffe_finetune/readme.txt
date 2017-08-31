因为github单个上传文件的限制, 我把生成的数据库文件和用的caffemodel的train file删掉了

当然, finetune_result和train_result里的训练结果也都删掉了

用到的caffemodel的train file在这里
D:\Caffe\caffe-master\models\bvlc_reference_caffenet\bvlc_reference_caffenet.caffemodel

生成数据库文件在文件列表ok之后create_imagenet.sh就可以

把caffe fine tune和python接口用起来参考
http://note.youdao.com/noteshare?id=3b034fa94a21c7a99d1b0277f1d6293b&sub=1E8554ED065C4234B0CE82832CA6EC2A
http://note.youdao.com/noteshare?id=7f2c7013bd3de9d0ec83bc67f39eec8c&sub=EA788309310749C5ABF2660EF5A5FAF1
http://note.youdao.com/noteshare?id=5fb2cc16ff47f1f6802cc7d1759a8908&sub=47B24E4A7D014EE8B864AD65A300DE7F

在Caffe_Python目录下
finetune_flickr_style_iter_10000.caffemodel
也删掉了, 这个是用fine-tune最终迭代10000次得到的train file
用它来进行单张图片的分类
就是在python接口上把caffe的模型用来做图像分类, end-end的结构, input img -> output label
