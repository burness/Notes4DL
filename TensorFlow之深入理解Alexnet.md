## 前言

前面看了一些Tensorflow的文档和一些比较有意思的项目，发现这里面水很深的，需要多花时间好好从头了解下，尤其是cv这块的东西，特别感兴趣，接下来一段时间会开始深入了解ImageNet比赛中中获得好成绩的那些模型： AlexNet、GoogLeNet、VGG（对就是之前在nerual network用的pretrained的model）、deep residual networks。

## ImageNet Classification with Deep Convolutional Neural Networks

[ImageNet Classification with Deep Convolutional Neural Networks][1] 是Hinton和他的学生Alex Krizhevsky在12年ImageNet Challenge使用的模型结构，刷新了Image Classification的几率，从此deep learning在Image这块开始一次次超过state-of-art，甚至于搭到打败人类的地步，看这边文章的过程中，发现了很多以前零零散散看到的一些优化技术，但是很多没有深入了解，这篇文章讲解了他们alexnet如何做到能达到那么好的成绩，好的废话不多说，来开始看文章

![alexnet_architecture.jpg][2]

这张图是基本的caffe中alexnet的网络结构，这里比较抽象，我用caffe的draw_net把alexnet的网络结构画出来了 ![alexnet_architecture2.jpg][3]

### AlexNet的基本结构

alexnet总共包括8层，其中前5层convolutional，后面3层是full-connected，文章里面说的是减少任何一个卷积结果会变得很差，下面我来具体讲讲每一层的构成：

*   第一层卷积层 输入图像为227\*227\*3(paper上貌似有点问题224\*224\*3)的图像，使用了96个kernels（96,11,11,3），以4个pixel为一个单位来右移或者下移，能够产生55*55个卷积后的矩形框值，然后进行response-normalized（其实是Local Response Normalized，后面我会讲下这里）和pooled之后，pool这一层好像caffe里面的alexnet和paper里面不太一样，alexnet里面采样了两个GPU，所以从图上面看第一层卷积层厚度有两部分，池化pool_size=(3,3),滑动步长为2个pixels，得到96个27*27个feature。
*   第二层卷积层使用256个（同样，分布在两个GPU上，每个128kernels（5\*5\*48）），做pad_size(2,2)的处理，以1个pixel为单位移动（感谢网友指出），能够产生27\*27个卷积后的矩阵框，做LRN处理，然后pooled，池化以3\*3矩形框，2个pixel为步长，得到256个13*13个features。
*   第三层、第四层都没有LRN和pool，第五层只有pool，其中第三层使用384个kernels（3\*3\*256，pad_size=(1,1),得到256\*15\*15，kernel_size为（3，3),以1个pixel为步长，得到256\*13\*13）；第四层使用384个kernels（pad_size(1,1)得到256\*15\*15，核大小为（3，3）步长为1个pixel，得到384\*13\*13）；第五层使用256个kernels（pad_size(1,1)得到384\*15\*15，kernel_size(3,3)，得到256\*13\*13，pool_size(3，3）步长2个pixels，得到256\*6\*6）。
*   全连接层： 前两层分别有4096个神经元，最后输出softmax为1000个（ImageNet），注意caffe图中全连接层中有relu、dropout、innerProduct。

（感谢AnaZou指出上面之前的一些问题） paper里面也指出了这张图是在两个GPU下做的，其中和caffe里面的alexnet可能还真有点差异，但这可能不是重点，各位在使用的时候，直接参考caffe中的alexnet的网络结果，每一层都十分详细，基本的结构理解和上面是一致的。

### AlexNet为啥取得比较好的结果

前面讲了下AlexNet的基本网络结构，大家肯定会对其中的一些点产生疑问，比如LRN、Relu、dropout， 相信接触过dl的小伙伴们都有听说或者了解过这些。这里我讲按paper中的描述详细讲述这些东西为什么能提高最终网络的性能。

#### ReLU Nonlinearity

一般来说，刚接触神经网络还没有深入了解深度学习的小伙伴们对这个都不会太熟，一般都会更了解另外两个激活函数（真正往神经网络中引入非线性关系，使神经网络能够有效拟合非线性函数）tanh(x)和(1+e^(-x))^(-1),而ReLU(Rectified Linear Units) f(x)=max(0,x)。基于ReLU的深度卷积网络比基于tanh的网络训练块数倍，下图是一个基于CIFAR-10的四层卷积网络在tanh和ReLU达到25%的training error的迭代次数：

![alexnet_ReLU.jpg][4]

实线、间断线分别代表的是ReLU、tanh的training error，可见ReLU比tanh能够更快的收敛

#### Local Response Normalization

使用ReLU f(x)=max(0,x)后，你会发现激活函数之后的值没有了tanh、sigmoid函数那样有一个值域区间，所以一般在ReLU之后会做一个normalization，LRU就是稳重提出（这里不确定，应该是提出？）一种方法，在神经科学中有个概念叫“Lateral inhibition”，讲的是活跃的神经元对它周边神经元的影响。

![alexnet_LRU.jpg][5]

#### Dropout

Dropout也是经常挺说的一个概念，能够比较有效地防止神经网络的过拟合。 相对于一般如线性模型使用正则的方法来防止模型过拟合，而在神经网络中Dropout通过修改神经网络本身结构来实现。对于某一层神经元，通过定义的概率来随机删除一些神经元，同时保持输入层与输出层神经元的个人不变，然后按照神经网络的学习方法进行参数更新，下一次迭代中，重新随机删除一些神经元，直至训练结束 ![][6]

#### Data Augmentation

其实，最简单的增强模型性能，防止模型过拟合的方法是增加数据，但是其实增加数据也是有策略的，paper当中从256\*256中随机提出227\*227的patches(paper里面是224*224)，还有就是通过PCA来扩展数据集。这样就很有效地扩展了数据集，其实还有更多的方法视你的业务场景去使用，比如做基本的图像转换如增加减少亮度，一些滤光算法等等之类的，这是一种特别有效地手段，尤其是当数据量不够大的时候。

文章里面，我认为的基本内容就是这个了，基本的网络结构和一些防止过拟合的小的技巧方法，对自己在后面的项目有很多指示作用。

## AlexNet On Tensorflow

caffe的AlexNet可以到/models/bvlc_alexnet/train_val.prototxt 去看看具体的网络结构，这里我会弄点基于Tensorflow的AlexNet： 代码是在<http://www.cs.toronto.edu/~guerzhoy/tf_alexnet/>

    from numpy import *
    import os
    from pylab import *
    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib.cbook as cbook
    import time
    from scipy.misc import imread
    from scipy.misc import imresize
    import matplotlib.image as mpimg
    from scipy.ndimage import filters
    import urllib
    from numpy import random
    
    
    import tensorflow as tf
    
    from caffe_classes import class_names
    
    train_x = zeros((1, 227,227,3)).astype(float32)
    train_y = zeros((1, 1000))
    xdim = train_x.shape[1:]
    ydim = train_y.shape[1]
    
    
    net_data = load("bvlc_alexnet.npy").item()
    
    def conv(input, kernel, biases, k_h, k_w, c_o, s_h, s_w,  padding="VALID", group=1):
        '''From https://github.com/ethereon/caffe-tensorflow
        '''
        c_i = input.get_shape()[-1]
        assert c_i%group==0
        assert c_o%group==0
        convolve = lambda i, k: tf.nn.conv2d(i, k, [1, s_h, s_w, 1], padding=padding)
    
    
        if group==1:
            conv = convolve(input, kernel)
        else:
            input_groups = tf.split(3, group, input)
            kernel_groups = tf.split(3, group, kernel)
            output_groups = [convolve(i, k) for i,k in zip(input_groups, kernel_groups)]
            conv = tf.concat(3, output_groups)
        return  tf.reshape(tf.nn.bias_add(conv, biases), conv.get_shape().as_list())
    
    
    
    x = tf.Variable(i)
    
    #conv1
    #conv(11, 11, 96, 4, 4, padding='VALID', name='conv1')
    k_h = 11; k_w = 11; c_o = 96; s_h = 4; s_w = 4
    conv1W = tf.Variable(net_data["conv1"][0])
    conv1b = tf.Variable(net_data["conv1"][1])
    conv1_in = conv(x, conv1W, conv1b, k_h, k_w, c_o, s_h, s_w, padding="SAME", group=1)
    conv1 = tf.nn.relu(conv1_in)
    
    #lrn1
    #lrn(2, 2e-05, 0.75, name='norm1')
    radius = 2; alpha = 2e-05; beta = 0.75; bias = 1.0
    lrn1 = tf.nn.local_response_normalization(conv1,
                                                      depth_radius=radius,
                                                      alpha=alpha,
                                                      beta=beta,
                                                      bias=bias)
    
    #maxpool1
    #max_pool(3, 3, 2, 2, padding='VALID', name='pool1')
    k_h = 3; k_w = 3; s_h = 2; s_w = 2; padding = 'VALID'
    maxpool1 = tf.nn.max_pool(lrn1, ksize=[1, k_h, k_w, 1], strides=[1, s_h, s_w, 1], padding=padding)
    
    
    #conv2
    #conv(5, 5, 256, 1, 1, group=2, name='conv2')
    k_h = 5; k_w = 5; c_o = 256; s_h = 1; s_w = 1; group = 2
    conv2W = tf.Variable(net_data["conv2"][0])
    conv2b = tf.Variable(net_data["conv2"][1])
    conv2_in = conv(maxpool1, conv2W, conv2b, k_h, k_w, c_o, s_h, s_w, padding="SAME", group=group)
    conv2 = tf.nn.relu(conv2_in)
    
    
    #lrn2
    #lrn(2, 2e-05, 0.75, name='norm2')
    radius = 2; alpha = 2e-05; beta = 0.75; bias = 1.0
    lrn2 = tf.nn.local_response_normalization(conv2,
                                                      depth_radius=radius,
                                                      alpha=alpha,
                                                      beta=beta,
                                                      bias=bias)
    
    #maxpool2
    #max_pool(3, 3, 2, 2, padding='VALID', name='pool2')
    k_h = 3; k_w = 3; s_h = 2; s_w = 2; padding = 'VALID'
    maxpool2 = tf.nn.max_pool(lrn2, ksize=[1, k_h, k_w, 1], strides=[1, s_h, s_w, 1], padding=padding)
    
    #conv3
    #conv(3, 3, 384, 1, 1, name='conv3')
    k_h = 3; k_w = 3; c_o = 384; s_h = 1; s_w = 1; group = 1
    conv3W = tf.Variable(net_data["conv3"][0])
    conv3b = tf.Variable(net_data["conv3"][1])
    conv3_in = conv(maxpool2, conv3W, conv3b, k_h, k_w, c_o, s_h, s_w, padding="SAME", group=group)
    conv3 = tf.nn.relu(conv3_in)
    
    #conv4
    #conv(3, 3, 384, 1, 1, group=2, name='conv4')
    k_h = 3; k_w = 3; c_o = 384; s_h = 1; s_w = 1; group = 2
    conv4W = tf.Variable(net_data["conv4"][0])
    conv4b = tf.Variable(net_data["conv4"][1])
    conv4_in = conv(conv3, conv4W, conv4b, k_h, k_w, c_o, s_h, s_w, padding="SAME", group=group)
    conv4 = tf.nn.relu(conv4_in)
    
    
    #conv5
    #conv(3, 3, 256, 1, 1, group=2, name='conv5')
    k_h = 3; k_w = 3; c_o = 256; s_h = 1; s_w = 1; group = 2
    conv5W = tf.Variable(net_data["conv5"][0])
    conv5b = tf.Variable(net_data["conv5"][1])
    conv5_in = conv(conv4, conv5W, conv5b, k_h, k_w, c_o, s_h, s_w, padding="SAME", group=group)
    conv5 = tf.nn.relu(conv5_in)
    
    #maxpool5
    #max_pool(3, 3, 2, 2, padding='VALID', name='pool5')
    k_h = 3; k_w = 3; s_h = 2; s_w = 2; padding = 'VALID'
    maxpool5 = tf.nn.max_pool(conv5, ksize=[1, k_h, k_w, 1], strides=[1, s_h, s_w, 1], padding=padding)
    
    #fc6
    #fc(4096, name='fc6')
    fc6W = tf.Variable(net_data["fc6"][0])
    fc6b = tf.Variable(net_data["fc6"][1])
    fc6 = tf.nn.relu_layer(tf.reshape(maxpool5, [1, int(prod(maxpool5.get_shape()[1:]))]), fc6W, fc6b)
    
    #fc7
    #fc(4096, name='fc7')
    fc7W = tf.Variable(net_data["fc7"][0])
    fc7b = tf.Variable(net_data["fc7"][1])
    fc7 = tf.nn.relu_layer(fc6, fc7W, fc7b)
    
    #fc8
    #fc(1000, relu=False, name='fc8')
    fc8W = tf.Variable(net_data["fc8"][0])
    fc8b = tf.Variable(net_data["fc8"][1])
    fc8 = tf.nn.xw_plus_b(fc7, fc8W, fc8b)
    
    
    #prob
    #softmax(name='prob'))
    prob = tf.nn.softmax(fc8)
    
    init = tf.initialize_all_variables()
    sess = tf.Session()
    sess.run(init)
    
    output = sess.run(prob)
    ################################################################################
    
    #Output:
    
    inds = argsort(output)[0,:]
    for i in range(5):
        print class_names[inds[-1-i]], output[0, inds[-1-i]]
    

这个是基于原生tensorflow的一版代码，好长而且看着比较麻烦一点，还load了caffe里面生成的网络模型，比较麻烦，这里找了一版稍微简单的<http://blog.csdn.net/chenriwei2/article/details/50615753>:

    # 输入数据
    import input_data
    mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)
    
    import tensorflow as tf
    
    # 定义网络超参数
    learning_rate = 0.001
    training_iters = 200000
    batch_size = 64
    display_step = 20
    
    # 定义网络参数
    n_input = 784 # 输入的维度
    n_classes = 10 # 标签的维度
    dropout = 0.8 # Dropout 的概率
    
    # 占位符输入
    x = tf.placeholder(tf.types.float32, [None, n_input])
    y = tf.placeholder(tf.types.float32, [None, n_classes])
    keep_prob = tf.placeholder(tf.types.float32)
    
    # 卷积操作
    def conv2d(name, l_input, w, b):
        return tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(l_input, w, strides=[1, 1, 1, 1], padding='SAME'),b), name=name)
    
    # 最大下采样操作
    def max_pool(name, l_input, k):
        return tf.nn.max_pool(l_input, ksize=[1, k, k, 1], strides=[1, k, k, 1], padding='SAME', name=name)
    
    # 归一化操作
    def norm(name, l_input, lsize=4):
        return tf.nn.lrn(l_input, lsize, bias=1.0, alpha=0.001 / 9.0, beta=0.75, name=name)
    
    # 定义整个网络 
    def alex_net(_X, _weights, _biases, _dropout):
        # 向量转为矩阵
        _X = tf.reshape(_X, shape=[-1, 28, 28, 1])
    
        # 卷积层
        conv1 = conv2d('conv1', _X, _weights['wc1'], _biases['bc1'])
        # 下采样层
        pool1 = max_pool('pool1', conv1, k=2)
        # 归一化层
        norm1 = norm('norm1', pool1, lsize=4)
        # Dropout
        norm1 = tf.nn.dropout(norm1, _dropout)
    
        # 卷积
        conv2 = conv2d('conv2', norm1, _weights['wc2'], _biases['bc2'])
        # 下采样
        pool2 = max_pool('pool2', conv2, k=2)
        # 归一化
        norm2 = norm('norm2', pool2, lsize=4)
        # Dropout
        norm2 = tf.nn.dropout(norm2, _dropout)
    
        # 卷积
        conv3 = conv2d('conv3', norm2, _weights['wc3'], _biases['bc3'])
        # 下采样
        pool3 = max_pool('pool3', conv3, k=2)
        # 归一化
        norm3 = norm('norm3', pool3, lsize=4)
        # Dropout
        norm3 = tf.nn.dropout(norm3, _dropout)
    
        # 全连接层，先把特征图转为向量
        dense1 = tf.reshape(norm3, [-1, _weights['wd1'].get_shape().as_list()[0]]) 
        dense1 = tf.nn.relu(tf.matmul(dense1, _weights['wd1']) + _biases['bd1'], name='fc1') 
        # 全连接层
        dense2 = tf.nn.relu(tf.matmul(dense1, _weights['wd2']) + _biases['bd2'], name='fc2') # Relu activation
    
        # 网络输出层
        out = tf.matmul(dense2, _weights['out']) + _biases['out']
        return out
    
    # 存储所有的网络参数
    weights = {
        'wc1': tf.Variable(tf.random_normal([3, 3, 1, 64])),
        'wc2': tf.Variable(tf.random_normal([3, 3, 64, 128])),
        'wc3': tf.Variable(tf.random_normal([3, 3, 128, 256])),
        'wd1': tf.Variable(tf.random_normal([4\*4\*256, 1024])),
        'wd2': tf.Variable(tf.random_normal([1024, 1024])),
        'out': tf.Variable(tf.random_normal([1024, 10]))
    }
    biases = {
        'bc1': tf.Variable(tf.random_normal([64])),
        'bc2': tf.Variable(tf.random_normal([128])),
        'bc3': tf.Variable(tf.random_normal([256])),
        'bd1': tf.Variable(tf.random_normal([1024])),
        'bd2': tf.Variable(tf.random_normal([1024])),
        'out': tf.Variable(tf.random_normal([n_classes]))
    }
    
    # 构建模型
    pred = alex_net(x, weights, biases, keep_prob)
    
    # 定义损失函数和学习步骤
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(pred, y))
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)
    
    # 测试网络
    correct_pred = tf.equal(tf.argmax(pred,1), tf.argmax(y,1))
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
    
    # 初始化所有的共享变量
    init = tf.initialize_all_variables()
    
    # 开启一个训练
    with tf.Session() as sess:
        sess.run(init)
        step = 1
        # Keep training until reach max iterations
        while step \* batch_size < training_iters:
            batch_xs, batch_ys = mnist.train.next_batch(batch_size)
            # 获取批数据
            sess.run(optimizer, feed_dict={x: batch_xs, y: batch_ys, keep_prob: dropout})
            if step % display_step == 0:
                # 计算精度
                acc = sess.run(accuracy, feed_dict={x: batch_xs, y: batch_ys, keep_prob: 1.})
                # 计算损失值
                loss = sess.run(cost, feed_dict={x: batch_xs, y: batch_ys, keep_prob: 1.})
                print "Iter " + str(step\*batch_size) + ", Minibatch Loss= " + "{:.6f}".format(loss) + ", Training Accuracy= " + "{:.5f}".format(acc)
            step += 1
        print "Optimization Finished!"
        # 计算测试精度
        print "Testing Accuracy:", sess.run(accuracy, feed_dict={x: mnist.test.images[:256], y: mnist.test.labels[:256], keep_prob: 1.})
    

基于mnist 构建alexnet，这里的input可以去tensorflow的github上去找找，这一版代码比较简单。

后来发现了tflearn里面有一个alexnet来分类Oxford的例子，好开心，在基于tflearn对一些日常layer的封装，代码量只有不到50行，看了下内部layer的实现，挺不错的，写代码的时候可以多参考参考，代码地址<https://github.com/tflearn/tflearn/blob/master/examples/images/alexnet.py>.

    from __future__ import division, print_function, absolute_import
    
    import tflearn
    from tflearn.layers.core import input_data, dropout, fully_connected
    from tflearn.layers.conv import conv_2d, max_pool_2d
    from tflearn.layers.normalization import local_response_normalization
    from tflearn.layers.estimator import regression
    
    import tflearn.datasets.oxflower17 as oxflower17
    X, Y = oxflower17.load_data(one_hot=True, resize_pics=(227, 227))
    
    # Building 'AlexNet'
    network = input_data(shape=[None, 227, 227, 3])
    network = conv_2d(network, 96, 11, strides=4, activation='relu')
    network = max_pool_2d(network, 3, strides=2)
    network = local_response_normalization(network)
    network = conv_2d(network, 256, 5, activation='relu')
    network = max_pool_2d(network, 3, strides=2)
    network = local_response_normalization(network)
    network = conv_2d(network, 384, 3, activation='relu')
    network = conv_2d(network, 384, 3, activation='relu')
    network = conv_2d(network, 256, 3, activation='relu')
    network = max_pool_2d(network, 3, strides=2)
    network = local_response_normalization(network)
    network = fully_connected(network, 4096, activation='tanh')
    network = dropout(network, 0.5)
    network = fully_connected(network, 4096, activation='tanh')
    network = dropout(network, 0.5)
    network = fully_connected(network, 17, activation='softmax')
    network = regression(network, optimizer='momentum',
                         loss='categorical_crossentropy',
                         learning_rate=0.001)
    
    # Training
    model = tflearn.DNN(network, checkpoint_path='model_alexnet',
                        max_checkpoints=1, tensorboard_verbose=2)
    model.fit(X, Y, n_epoch=1000, validation_set=0.1, shuffle=True,
              show_metric=True, batch_size=64, snapshot_step=200,
              snapshot_epoch=False, run_id='alexnet_oxflowers17')
    

使用tflearn版本的alexnet来做实验，从TensorBoard上到到的基本效果如下： ![][7] alexnet graph 如下： ![][8]

## Reference

1,[ImageNet Classification with Deep Convolutional Neural Networks][1] 2,<http://blog.csdn.net/chenriwei2/article/details/50615753> 3,<http://www.cs.toronto.edu/~guerzhoy/tf_alexnet/> 4,<https://github.com/tflearn/tflearn/blob/master/examples/images/alexnet.py> 5,<https://github.com/BVLC/caffe/blob/master/python/draw_net.py>

 [1]: http://www.cs.toronto.edu/~fritz/absps/imagenet.pdf
 [2]: http://7xr3b9.com1.z0.glb.clouddn.com/mac_blogs_alexnet_architec## 前言

前面看了一些Tensorflow的文档和一些比较有意思的项目，发现这里面水很深的，需要多花时间好好从头了解下，尤其是cv这块的东西，特别感兴趣，接下来一段时间会开始深入了解ImageNet比赛中中获得好成绩的那些模型： AlexNet、GoogLeNet、VGG（对就是之前在nerual network用的pretrained的model）、deep residual networks。

## ImageNet Classification with Deep Convolutional Neural Networks

[ImageNet Classification with Deep Convolutional Neural Networks][1] 是Hinton和他的学生Alex Krizhevsky在12年ImageNet Challenge使用的模型结构，刷新了Image Classification的几率，从此deep learning在Image这块开始一次次超过state-of-art，甚至于搭到打败人类的地步，看这边文章的过程中，发现了很多以前零零散散看到的一些优化技术，但是很多没有深入了解，这篇文章讲解了他们alexnet如何做到能达到那么好的成绩，好的废话不多说，来开始看文章

![alexnet_architecture.jpg][2]

这张图是基本的caffe中alexnet的网络结构，这里比较抽象，我用caffe的draw_net把alexnet的网络结构画出来了 ![alexnet_architecture2.jpg][3]

### AlexNet的基本结构

alexnet总共包括8层，其中前5层convolutional，后面3层是full-connected，文章里面说的是减少任何一个卷积结果会变得很差，下面我来具体讲讲每一层的构成：

*   第一层卷积层 输入图像为227\*227\*3(paper上貌似有点问题224\*224\*3)的图像，使用了96个kernels（96,11,11,3），以4个pixel为一个单位来右移或者下移，能够产生55*55个卷积后的矩形框值，然后进行response-normalized（其实是Local Response Normalized，后面我会讲下这里）和pooled之后，pool这一层好像caffe里面的alexnet和paper里面不太一样，alexnet里面采样了两个GPU，所以从图上面看第一层卷积层厚度有两部分，池化pool_size=(3,3),滑动步长为2个pixels，得到96个27*27个feature。
*   第二层卷积层使用256个（同样，分布在两个GPU上，每个128kernels（5\*5\*48）），做pad_size(2,2)的处理，以1个pixel为单位移动（感谢网友指出），能够产生27\*27个卷积后的矩阵框，做LRN处理，然后pooled，池化以3\*3矩形框，2个pixel为步长，得到256个13*13个features。
*   第三层、第四层都没有LRN和pool，第五层只有pool，其中第三层使用384个kernels（3\*3\*256，pad_size=(1,1),得到256\*15\*15，kernel_size为（3，3),以1个pixel为步长，得到256\*13\*13）；第四层使用384个kernels（pad_size(1,1)得到256\*15\*15，核大小为（3，3）步长为1个pixel，得到384\*13\*13）；第五层使用256个kernels（pad_size(1,1)得到384\*15\*15，kernel_size(3,3)，得到256\*13\*13，pool_size(3，3）步长2个pixels，得到256\*6\*6）。
*   全连接层： 前两层分别有4096个神经元，最后输出softmax为1000个（ImageNet），注意caffe图中全连接层中有relu、dropout、innerProduct。

（感谢AnaZou指出上面之前的一些问题） paper里面也指出了这张图是在两个GPU下做的，其中和caffe里面的alexnet可能还真有点差异，但这可能不是重点，各位在使用的时候，直接参考caffe中的alexnet的网络结果，每一层都十分详细，基本的结构理解和上面是一致的。

### AlexNet为啥取得比较好的结果

前面讲了下AlexNet的基本网络结构，大家肯定会对其中的一些点产生疑问，比如LRN、Relu、dropout， 相信接触过dl的小伙伴们都有听说或者了解过这些。这里我讲按paper中的描述详细讲述这些东西为什么能提高最终网络的性能。

#### ReLU Nonlinearity

一般来说，刚接触神经网络还没有深入了解深度学习的小伙伴们对这个都不会太熟，一般都会更了解另外两个激活函数（真正往神经网络中引入非线性关系，使神经网络能够有效拟合非线性函数）tanh(x)和(1+e^(-x))^(-1),而ReLU(Rectified Linear Units) f(x)=max(0,x)。基于ReLU的深度卷积网络比基于tanh的网络训练块数倍，下图是一个基于CIFAR-10的四层卷积网络在tanh和ReLU达到25%的training error的迭代次数：

![alexnet_ReLU.jpg][4]

实线、间断线分别代表的是ReLU、tanh的training error，可见ReLU比tanh能够更快的收敛

#### Local Response Normalization

使用ReLU f(x)=max(0,x)后，你会发现激活函数之后的值没有了tanh、sigmoid函数那样有一个值域区间，所以一般在ReLU之后会做一个normalization，LRU就是稳重提出（这里不确定，应该是提出？）一种方法，在神经科学中有个概念叫“Lateral inhibition”，讲的是活跃的神经元对它周边神经元的影响。

![alexnet_LRU.jpg][5]

#### Dropout

Dropout也是经常挺说的一个概念，能够比较有效地防止神经网络的过拟合。 相对于一般如线性模型使用正则的方法来防止模型过拟合，而在神经网络中Dropout通过修改神经网络本身结构来实现。对于某一层神经元，通过定义的概率来随机删除一些神经元，同时保持输入层与输出层神经元的个人不变，然后按照神经网络的学习方法进行参数更新，下一次迭代中，重新随机删除一些神经元，直至训练结束 ![][6]

#### Data Augmentation

其实，最简单的增强模型性能，防止模型过拟合的方法是增加数据，但是其实增加数据也是有策略的，paper当中从256\*256中随机提出227\*227的patches(paper里面是224*224)，还有就是通过PCA来扩展数据集。这样就很有效地扩展了数据集，其实还有更多的方法视你的业务场景去使用，比如做基本的图像转换如增加减少亮度，一些滤光算法等等之类的，这是一种特别有效地手段，尤其是当数据量不够大的时候。

文章里面，我认为的基本内容就是这个了，基本的网络结构和一些防止过拟合的小的技巧方法，对自己在后面的项目有很多指示作用。

## AlexNet On Tensorflow

caffe的AlexNet可以到/models/bvlc_alexnet/train_val.prototxt 去看看具体的网络结构，这里我会弄点基于Tensorflow的AlexNet： 代码是在<http://www.cs.toronto.edu/~guerzhoy/tf_alexnet/>

    from numpy import *
    import os
    from pylab import *
    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib.cbook as cbook
    import time
    from scipy.misc import imread
    from scipy.misc import imresize
    import matplotlib.image as mpimg
    from scipy.ndimage import filters
    import urllib
    from numpy import random
    
    
    import tensorflow as tf
    
    from caffe_classes import class_names
    
    train_x = zeros((1, 227,227,3)).astype(float32)
    train_y = zeros((1, 1000))
    xdim = train_x.shape[1:]
    ydim = train_y.shape[1]
    
    
    net_data = load("bvlc_alexnet.npy").item()
    
    def conv(input, kernel, biases, k_h, k_w, c_o, s_h, s_w,  padding="VALID", group=1):
        '''From https://github.com/ethereon/caffe-tensorflow
        '''
        c_i = input.get_shape()[-1]
        assert c_i%group==0
        assert c_o%group==0
        convolve = lambda i, k: tf.nn.conv2d(i, k, [1, s_h, s_w, 1], padding=padding)
    
    
        if group==1:
            conv = convolve(input, kernel)
        else:
            input_groups = tf.split(3, group, input)
            kernel_groups = tf.split(3, group, kernel)
            output_groups = [convolve(i, k) for i,k in zip(input_groups, kernel_groups)]
            conv = tf.concat(3, output_groups)
        return  tf.reshape(tf.nn.bias_add(conv, biases), conv.get_shape().as_list())
    
    
    
    x = tf.Variable(i)
    
    #conv1
    #conv(11, 11, 96, 4, 4, padding='VALID', name='conv1')
    k_h = 11; k_w = 11; c_o = 96; s_h = 4; s_w = 4
    conv1W = tf.Variable(net_data["conv1"][0])
    conv1b = tf.Variable(net_data["conv1"][1])
    conv1_in = conv(x, conv1W, conv1b, k_h, k_w, c_o, s_h, s_w, padding="SAME", group=1)
    conv1 = tf.nn.relu(conv1_in)
    
    #lrn1
    #lrn(2, 2e-05, 0.75, name='norm1')
    radius = 2; alpha = 2e-05; beta = 0.75; bias = 1.0
    lrn1 = tf.nn.local_response_normalization(conv1,
                                                      depth_radius=radius,
                                                      alpha=alpha,
                                                      beta=beta,
                                                      bias=bias)
    
    #maxpool1
    #max_pool(3, 3, 2, 2, padding='VALID', name='pool1')
    k_h = 3; k_w = 3; s_h = 2; s_w = 2; padding = 'VALID'
    maxpool1 = tf.nn.max_pool(lrn1, ksize=[1, k_h, k_w, 1], strides=[1, s_h, s_w, 1], padding=padding)
    
    
    #conv2
    #conv(5, 5, 256, 1, 1, group=2, name='conv2')
    k_h = 5; k_w = 5; c_o = 256; s_h = 1; s_w = 1; group = 2
    conv2W = tf.Variable(net_data["conv2"][0])
    conv2b = tf.Variable(net_data["conv2"][1])
    conv2_in = conv(maxpool1, conv2W, conv2b, k_h, k_w, c_o, s_h, s_w, padding="SAME", group=group)
    conv2 = tf.nn.relu(conv2_in)
    
    
    #lrn2
    #lrn(2, 2e-05, 0.75, name='norm2')
    radius = 2; alpha = 2e-05; beta = 0.75; bias = 1.0
    lrn2 = tf.nn.local_response_normalization(conv2,
                                                      depth_radius=radius,
                                                      alpha=alpha,
                                                      beta=beta,
                                                      bias=bias)
    
    #maxpool2
    #max_pool(3, 3, 2, 2, padding='VALID', name='pool2')
    k_h = 3; k_w = 3; s_h = 2; s_w = 2; padding = 'VALID'
    maxpool2 = tf.nn.max_pool(lrn2, ksize=[1, k_h, k_w, 1], strides=[1, s_h, s_w, 1], padding=padding)
    
    #conv3
    #conv(3, 3, 384, 1, 1, name='conv3')
    k_h = 3; k_w = 3; c_o = 384; s_h = 1; s_w = 1; group = 1
    conv3W = tf.Variable(net_data["conv3"][0])
    conv3b = tf.Variable(net_data["conv3"][1])
    conv3_in = conv(maxpool2, conv3W, conv3b, k_h, k_w, c_o, s_h, s_w, padding="SAME", group=group)
    conv3 = tf.nn.relu(conv3_in)
    
    #conv4
    #conv(3, 3, 384, 1, 1, group=2, name='conv4')
    k_h = 3; k_w = 3; c_o = 384; s_h = 1; s_w = 1; group = 2
    conv4W = tf.Variable(net_data["conv4"][0])
    conv4b = tf.Variable(net_data["conv4"][1])
    conv4_in = conv(conv3, conv4W, conv4b, k_h, k_w, c_o, s_h, s_w, padding="SAME", group=group)
    conv4 = tf.nn.relu(conv4_in)
    
    
    #conv5
    #conv(3, 3, 256, 1, 1, group=2, name='conv5')
    k_h = 3; k_w = 3; c_o = 256; s_h = 1; s_w = 1; group = 2
    conv5W = tf.Variable(net_data["conv5"][0])
    conv5b = tf.Variable(net_data["conv5"][1])
    conv5_in = conv(conv4, conv5W, conv5b, k_h, k_w, c_o, s_h, s_w, padding="SAME", group=group)
    conv5 = tf.nn.relu(conv5_in)
    
    #maxpool5
    #max_pool(3, 3, 2, 2, padding='VALID', name='pool5')
    k_h = 3; k_w = 3; s_h = 2; s_w = 2; padding = 'VALID'
    maxpool5 = tf.nn.max_pool(conv5, ksize=[1, k_h, k_w, 1], strides=[1, s_h, s_w, 1], padding=padding)
    
    #fc6
    #fc(4096, name='fc6')
    fc6W = tf.Variable(net_data["fc6"][0])
    fc6b = tf.Variable(net_data["fc6"][1])
    fc6 = tf.nn.relu_layer(tf.reshape(maxpool5, [1, int(prod(maxpool5.get_shape()[1:]))]), fc6W, fc6b)
    
    #fc7
    #fc(4096, name='fc7')
    fc7W = tf.Variable(net_data["fc7"][0])
    fc7b = tf.Variable(net_data["fc7"][1])
    fc7 = tf.nn.relu_layer(fc6, fc7W, fc7b)
    
    #fc8
    #fc(1000, relu=False, name='fc8')
    fc8W = tf.Variable(net_data["fc8"][0])
    fc8b = tf.Variable(net_data["fc8"][1])
    fc8 = tf.nn.xw_plus_b(fc7, fc8W, fc8b)
    
    
    #prob
    #softmax(name='prob'))
    prob = tf.nn.softmax(fc8)
    
    init = tf.initialize_all_variables()
    sess = tf.Session()
    sess.run(init)
    
    output = sess.run(prob)
    ################################################################################
    
    #Output:
    
    inds = argsort(output)[0,:]
    for i in range(5):
        print class_names[inds[-1-i]], output[0, inds[-1-i]]
    

这个是基于原生tensorflow的一版代码，好长而且看着比较麻烦一点，还load了caffe里面生成的网络模型，比较麻烦，这里找了一版稍微简单的<http://blog.csdn.net/chenriwei2/article/details/50615753>:

    # 输入数据
    import input_data
    mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)
    
    import tensorflow as tf
    
    # 定义网络超参数
    learning_rate = 0.001
    training_iters = 200000
    batch_size = 64
    display_step = 20
    
    # 定义网络参数
    n_input = 784 # 输入的维度
    n_classes = 10 # 标签的维度
    dropout = 0.8 # Dropout 的概率
    
    # 占位符输入
    x = tf.placeholder(tf.types.float32, [None, n_input])
    y = tf.placeholder(tf.types.float32, [None, n_classes])
    keep_prob = tf.placeholder(tf.types.float32)
    
    # 卷积操作
    def conv2d(name, l_input, w, b):
        return tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(l_input, w, strides=[1, 1, 1, 1], padding='SAME'),b), name=name)
    
    # 最大下采样操作
    def max_pool(name, l_input, k):
        return tf.nn.max_pool(l_input, ksize=[1, k, k, 1], strides=[1, k, k, 1], padding='SAME', name=name)
    
    # 归一化操作
    def norm(name, l_input, lsize=4):
        return tf.nn.lrn(l_input, lsize, bias=1.0, alpha=0.001 / 9.0, beta=0.75, name=name)
    
    # 定义整个网络 
    def alex_net(_X, _weights, _biases, _dropout):
        # 向量转为矩阵
        _X = tf.reshape(_X, shape=[-1, 28, 28, 1])
    
        # 卷积层
        conv1 = conv2d('conv1', _X, _weights['wc1'], _biases['bc1'])
        # 下采样层
        pool1 = max_pool('pool1', conv1, k=2)
        # 归一化层
        norm1 = norm('norm1', pool1, lsize=4)
        # Dropout
        norm1 = tf.nn.dropout(norm1, _dropout)
    
        # 卷积
        conv2 = conv2d('conv2', norm1, _weights['wc2'], _biases['bc2'])
        # 下采样
        pool2 = max_pool('pool2', conv2, k=2)
        # 归一化
        norm2 = norm('norm2', pool2, lsize=4)
        # Dropout
        norm2 = tf.nn.dropout(norm2, _dropout)
    
        # 卷积
        conv3 = conv2d('conv3', norm2, _weights['wc3'], _biases['bc3'])
        # 下采样
        pool3 = max_pool('pool3', conv3, k=2)
        # 归一化
        norm3 = norm('norm3', pool3, lsize=4)
        # Dropout
        norm3 = tf.nn.dropout(norm3, _dropout)
    
        # 全连接层，先把特征图转为向量
        dense1 = tf.reshape(norm3, [-1, _weights['wd1'].get_shape().as_list()[0]]) 
        dense1 = tf.nn.relu(tf.matmul(dense1, _weights['wd1']) + _biases['bd1'], name='fc1') 
        # 全连接层
        dense2 = tf.nn.relu(tf.matmul(dense1, _weights['wd2']) + _biases['bd2'], name='fc2') # Relu activation
    
        # 网络输出层
        out = tf.matmul(dense2, _weights['out']) + _biases['out']
        return out
    
    # 存储所有的网络参数
    weights = {
        'wc1': tf.Variable(tf.random_normal([3, 3, 1, 64])),
        'wc2': tf.Variable(tf.random_normal([3, 3, 64, 128])),
        'wc3': tf.Variable(tf.random_normal([3, 3, 128, 256])),
        'wd1': tf.Variable(tf.random_normal([4\*4\*256, 1024])),
        'wd2': tf.Variable(tf.random_normal([1024, 1024])),
        'out': tf.Variable(tf.random_normal([1024, 10]))
    }
    biases = {
        'bc1': tf.Variable(tf.random_normal([64])),
        'bc2': tf.Variable(tf.random_normal([128])),
        'bc3': tf.Variable(tf.random_normal([256])),
        'bd1': tf.Variable(tf.random_normal([1024])),
        'bd2': tf.Variable(tf.random_normal([1024])),
        'out': tf.Variable(tf.random_normal([n_classes]))
    }
    
    # 构建模型
    pred = alex_net(x, weights, biases, keep_prob)
    
    # 定义损失函数和学习步骤
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(pred, y))
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)
    
    # 测试网络
    correct_pred = tf.equal(tf.argmax(pred,1), tf.argmax(y,1))
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
    
    # 初始化所有的共享变量
    init = tf.initialize_all_variables()
    
    # 开启一个训练
    with tf.Session() as sess:
        sess.run(init)
        step = 1
        # Keep training until reach max iterations
        while step \* batch_size < training_iters:
            batch_xs, batch_ys = mnist.train.next_batch(batch_size)
            # 获取批数据
            sess.run(optimizer, feed_dict={x: batch_xs, y: batch_ys, keep_prob: dropout})
            if step % display_step == 0:
                # 计算精度
                acc = sess.run(accuracy, feed_dict={x: batch_xs, y: batch_ys, keep_prob: 1.})
                # 计算损失值
                loss = sess.run(cost, feed_dict={x: batch_xs, y: batch_ys, keep_prob: 1.})
                print "Iter " + str(step\*batch_size) + ", Minibatch Loss= " + "{:.6f}".format(loss) + ", Training Accuracy= " + "{:.5f}".format(acc)
            step += 1
        print "Optimization Finished!"
        # 计算测试精度
        print "Testing Accuracy:", sess.run(accuracy, feed_dict={x: mnist.test.images[:256], y: mnist.test.labels[:256], keep_prob: 1.})
    

基于mnist 构建alexnet，这里的input可以去tensorflow的github上去找找，这一版代码比较简单。

后来发现了tflearn里面有一个alexnet来分类Oxford的例子，好开心，在基于tflearn对一些日常layer的封装，代码量只有不到50行，看了下内部layer的实现，挺不错的，写代码的时候可以多参考参考，代码地址<https://github.com/tflearn/tflearn/blob/master/examples/images/alexnet.py>.

    from __future__ import division, print_function, absolute_import
    
    import tflearn
    from tflearn.layers.core import input_data, dropout, fully_connected
    from tflearn.layers.conv import conv_2d, max_pool_2d
    from tflearn.layers.normalization import local_response_normalization
    from tflearn.layers.estimator import regression
    
    import tflearn.datasets.oxflower17 as oxflower17
    X, Y = oxflower17.load_data(one_hot=True, resize_pics=(227, 227))
    
    # Building 'AlexNet'
    network = input_data(shape=[None, 227, 227, 3])
    network = conv_2d(network, 96, 11, strides=4, activation='relu')
    network = max_pool_2d(network, 3, strides=2)
    network = local_response_normalization(network)
    network = conv_2d(network, 256, 5, activation='relu')
    network = max_pool_2d(network, 3, strides=2)
    network = local_response_normalization(network)
    network = conv_2d(network, 384, 3, activation='relu')
    network = conv_2d(network, 384, 3, activation='relu')
    network = conv_2d(network, 256, 3, activation='relu')
    network = max_pool_2d(network, 3, strides=2)
    network = local_response_normalization(network)
    network = fully_connected(network, 4096, activation='tanh')
    network = dropout(network, 0.5)
    network = fully_connected(network, 4096, activation='tanh')
    network = dropout(network, 0.5)
    network = fully_connected(network, 17, activation='softmax')
    network = regression(network, optimizer='momentum',
                         loss='categorical_crossentropy',
                         learning_rate=0.001)
    
    # Training
    model = tflearn.DNN(network, checkpoint_path='model_alexnet',
                        max_checkpoints=1, tensorboard_verbose=2)
    model.fit(X, Y, n_epoch=1000, validation_set=0.1, shuffle=True,
              show_metric=True, batch_size=64, snapshot_step=200,
              snapshot_epoch=False, run_id='alexnet_oxflowers17')
    

使用tflearn版本的alexnet来做实验，从TensorBoard上到到的基本效果如下： ![][7] alexnet graph 如下： ![][8]

## Reference

1,[ImageNet Classification with Deep Convolutional Neural Networks][1] 2,<http://blog.csdn.net/chenriwei2/article/details/50615753> 3,<http://www.cs.toronto.edu/~guerzhoy/tf_alexnet/> 4,<https://github.com/tflearn/tflearn/blob/master/examples/images/alexnet.py> 5,<https://github.com/BVLC/caffe/blob/master/python/draw_net.py>

 [1]: http://www.cs.toronto.edu/~fritz/absps/imagenet.pdf
 [2]: ./images/alexnet_architecture.jpg
 [3]: ./images/alext_architecture2.png
 [4]: ./images/alexnet_ReLU.jpg
 [5]: ./images/alexnet_LRU.jpg
 [6]: ./images/alexnet_Dropout.jpg
 [7]: ./images/alexnet_tb1.jpg
 [8]: ./images/alexnet_graph.pngture.jpg
 [3]: ./images/alext_architecture2.png
 [4]: ./images/alexnet_ReLU.jpg
 [5]: ./images/alexnet_LRU.jpg
 [6]: ./images/alexnet_Dropout.jpg
 [7]: ./images/alexnet_tb1.jpg
 [8]: ./images/alexnet_graph.png