# -*- coding: utf-8 -*-
import os
import sys
import timeit

import numpy

import theano
import theano.tensor as T
from theano.tensor.shared_randomstreams import RandomStreams

import cPickle, gzip

from logistic_sgd import load_data
from utils import tile_raster_images

try:
    import PIL.Image as Image
except ImportError:
    import Image

class auto_encoder(object):
    def __init__(
        self,
        numpy_rng,
        theano_rng=None,
        input=None,
        n_visible=784,#输入层节点的个数
        n_hidden=500,#隐含层节点的个数
        W=None,
        bhid=None,
        bvis=None
    ):
        """
        :type numpy_rng: numpy.random.RandomState
        :param numpy_rng: 用于随机产生权重和偏置
    
        :type theano_rng: theano.tensor.shared_randomstreams.RandomStreams
        :param theano_rng: Theano的随机数生成
    
        :type input: theano.tensor.TensorType
        :param input: 描述输入的参数，None表示的是单独的autoencoder
    
        :type n_visible: int
        :param n_visible: 输入层节点的个数
    
        :type n_hidden: int
        :param n_hidden: 隐含层节点的个    数
    
        :type W: theano.tensor.TensorType
        :param W: 指示权重的参数，如果是None表示的是单独的Autoencoder
    
        :type bhid: theano.tensor.TensorType
        :param bhid: 指示隐含层偏置的参数，如果是None表示的是单独的Autoencoder
    
        :type bvis: theano.tensor.TensorType
        :param bvis: 指示输出层偏置的参数，如果是None表示的是单独的Autoencoder
    
        """
        self.n_visible = n_visible#设置输入层的节点个数
        self.n_hidden = n_hidden#设置隐含层的节点个数

        if not theano_rng:
            theano_rng = RandomStreams(numpy_rng.randint(2 ** 30))

        if not W:#初始化权重
            initial_W = numpy.asarray(
                numpy_rng.uniform(
                    low=-4 * numpy.sqrt(6. / (n_hidden + n_visible)),#下界
                    high=4 * numpy.sqrt(6. / (n_hidden + n_visible)),#上界
                    size=(n_visible, n_hidden)
                ),
                dtype=theano.config.floatX
            )
            W = theano.shared(value=initial_W, name='W', borrow=True)

        if not bvis:#初始化偏置
            bvis = theano.shared(
                value=numpy.zeros(#初始化为0
                    n_visible,
                    dtype=theano.config.floatX
                ),
                borrow=True
            )

        if not bhid:#初始化偏置
            bhid = theano.shared(#初始化为0
                value=numpy.zeros(
                    n_hidden,
                    dtype=theano.config.floatX
                ),
                name='b',
                borrow=True
            )

        self.W = W#输入层到隐含层的权重
        self.b = bhid#输入层到隐含层的偏置
        self.b_prime = bvis#隐含层到输出层的偏置
        self.W_prime = self.W.T#隐含层到输出层的偏置
        self.theano_rng = theano_rng

        #将输入作为参数传入，可以方便后面堆叠成深层的网络
        if input is None:
            self.x = T.dmatrix(name='input')
        else:
            self.x = input

        self.params = [self.W, self.b, self.b_prime]#声明参数



    def get_corrupted_input(self, input, corruption_level):
        return self.theano_rng.binomial(size=input.shape, n=1,
                                        p=1 - corruption_level,
                                        dtype=theano.config.floatX) * input


    def get_hidden_values(self, input):#计算隐含层的输出
        return T.nnet.sigmoid(T.dot(input, self.W) + self.b)

    def get_reconstructed_input(self, hidden):#计算输出层的输出
        return T.nnet.sigmoid(T.dot(hidden, self.W_prime) + self.b_prime)

    def get_cost_updates(self, corruption_level, learning_rate):#计算损失函数和更新

        tilde_x = self.get_corrupted_input(self.x, corruption_level)
        y = self.get_hidden_values(tilde_x)
        z = self.get_reconstructed_input(y)

        #损失函数
        L = - T.sum(self.x * T.log(z) + (1 - self.x) * T.log(1 - z), axis=1)
        cost = T.mean(L)#求出平均误差

        # 对需要求解的参数求其梯度
        gparams = T.grad(cost, self.params)

        #基于梯度下降更新每个参数
        updates = [
            (param, param - learning_rate * gparam)
            for param, gparam in zip(self.params, gparams)
        ]

        return (cost, updates)

def test_dA(learning_rate=0.01, training_epochs=20,
            dataset='..\\raw_data\\data.pkl',
            batch_size=10, output_folder='cA_plots', contraction_level=.1):

    # 1、导入数据集#
    datasets = my_loaddata(dataset)  # 导入数据集，函数在logistic_sgd中定义
    train_set_x, train_set_y = datasets[0]  # 得到训练数据

    # 2、构建模型
    rng = numpy.random.RandomState(123)
    theano_rng = RandomStreams(rng.randint(2 ** 30))

    # 批量下降法，训练的批数
    n_train_batches = train_set_x.get_value(borrow=True).shape[0] / batch_size

    # allocate symbolic variables for the data
    index = T.lscalar()  # 每一批训练数据的索引
    x = T.matrix('x')  # 每一批训练数据

    # 初始化模型的参数
    da = auto_encoder(
        numpy_rng=rng,
        theano_rng=theano_rng,
        input=x,
        n_visible=28 * 28,
        n_hidden=500
    )

    # 定义模型的损失函数和更新规则
    cost, updates = da.get_cost_updates(
        corruption_level=0.,
        learning_rate=learning_rate
    )

    # 定义训练函数
    train_da = theano.function(
        [index],
        cost,
        updates=updates,
        givens={
            x: train_set_x[index * batch_size: (index + 1) * batch_size]
        }
    )

    start_time = timeit.default_timer()  # 定义训练的初始时间

    # 3、训练模型
    # go through training epochs
    for epoch in xrange(training_epochs):
        # go through trainng set
        c = []
        for batch_index in xrange(n_train_batches):
            da_result = train_da(batch_index)
            c.append(train_da(batch_index))

        print 'Training epoch %d, cost ' % epoch, numpy.mean(c)

    end_time = timeit.default_timer()

    training_time = (end_time - start_time)

    # '''
    print >> sys.stderr, ('The no corruption code for file ' +
                          os.path.split(__file__)[1] +
                          ' ran for %.2fm' % ((training_time) / 60.))
    image = Image.fromarray(
        tile_raster_images(X=da.W.get_value(borrow=True).T,
                           img_shape=(28, 28), tile_shape=(10, 10),
                           tile_spacing=(1, 1)))
    image.save('filters_corruption_0.png')

    #计算每个样本经过auto-encoder的隐藏层, 存入svm数据
    read_file = open('..\\raw_data\\data.pkl', 'rb')
    train_set, test_set = cPickle.load(read_file)
    read_file.close()
    #train_set
    feature_train = T.nnet.sigmoid(T.dot(train_set[0], da.W.get_value()).eval() + da.b.get_value()).eval()
    save_feature_txt(feature_train, 'train.txt', train_set[1])
    #test_set
    feature_test = T.nnet.sigmoid(T.dot(test_set[0], da.W.get_value()).eval() + da.b.get_value()).eval()
    save_feature_txt(feature_test, 'test.txt', test_set[1])

    numpy.savetxt('w.txt', da.W.get_value(), delimiter=' ')
    numpy.savetxt('b.txt', da.b.get_value(), delimiter=' ')


    os.chdir('../')
    # '''

def save_feature_txt(feature_all,fname,class_all):
    try:
        fobj = open(fname,'w')
    except IOError:
        print '*** file open error:'
    else:
        for img_id in xrange(len(feature_all)):
            feature = feature_all[img_id,:]
            if class_all[img_id] == 0:
                fobj.write('+1' + ' ')
            elif class_all[img_id] == 1:
                fobj.write('-1' + ' ')
            for feature_id in xrange(feature.size):
                fobj.write(str(feature_id+1) + ':' + str(feature[feature_id]) + ' ')  # 这里的\n的意思是在源文件末尾换行，即新加内容另起一行插入。
            fobj.write('\n')
        fobj.close()  # 特别注意文件操作完毕后要close

def my_loaddata(file_path):
    print('... loading data')
    read_file = open(file_path, 'rb')
    train_set, test_set = cPickle.load(read_file)
    read_file.close()

    test_set_x, test_set_y = shared_dataset(test_set)
    train_set_x, train_set_y = shared_dataset(train_set)

    rval = [(train_set_x, train_set_y),
            (test_set_x, test_set_y)]
    return rval

def shared_dataset(data_xy, borrow=True):
    """ Function that loads the dataset into shared variables
    The reason we store our dataset in shared variables is to allow
    Theano to copy it into the GPU memory (when code is run on GPU).
    Since copying data into the GPU is slow, copying a minibatch everytime
    is needed (the default behaviour if the data is not in a shared
    variable) would lead to a large decrease in performance.
    """
    data_x, data_y = data_xy
    shared_x = theano.shared(numpy.asarray(data_x,
                                           dtype=theano.config.floatX),
                             borrow=borrow)
    shared_y = theano.shared(numpy.asarray(data_y,
                                           dtype=theano.config.floatX),
                             borrow=borrow)
    # When storing data on the GPU it has to be stored as floats
    # therefore we will store the labels as ``floatX`` as well
    # (``shared_y`` does exactly that). But during our computations
    # we need them as ints (we use labels as index, and if they are
    # floats it doesn't make sense) therefore instead of returning
    # ``shared_y`` we will have to cast it to int. This little hack
    # lets ous get around this issue
    return shared_x, T.cast(shared_y, 'int32')



if __name__ == '__main__':
    test_dA()

