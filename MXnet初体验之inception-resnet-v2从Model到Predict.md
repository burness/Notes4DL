## 前言

**如何找到自己实用的丹炉，是一个深度修真之人至关重要的，丹炉的好坏直接关系到炼丹的成功与否，道途千载，寻一合适丹炉也不妨这千古悠悠的修真（正）之路**

为什么学mxnet？ 熟悉本人博客的都知道，前段时间一直在关注TensorFlow也安利了很多次TFlearn，为什么这次突然会写MXnet的东西呢？原因是没钱呀，TensorFlow计算力要求太高，虽然使用方便，但是显存占用太高，计算也不够快，做公司项目还好，自己玩一些好玩的东西时太费时间，不过现在MXnet资源相对少一点，基于MXnet的有意思的开源项目也相对少一点，不过没关系，都不是问题，另外一点就是造MXnet都是一群说得上名字的大牛，能和大牛们玩一样的东西，想想都很兴奋。

MXnet的文档一直被一些爱好者喷，确实文档比较少，不过考虑到开发者都是业余时间造轮子（不，造丹炉！），很那像其他的框架有那么熟悉的文档，不过还好，在cv这块还是比较容易下手的。 这里有我从最近开始接触MXnet（其实很早就听说一直没有用过），学习的一些代码还有笔记[mxnet 101][1],没有特别细致研究，只是了解怎么用在CV上，完整的做一个项目。

## 新的丹方--inception-resnet-v2

**每一付新的丹方，无不是深度前辈们多年经验的结晶，丹方，很多时候在同样炼丹材料表现天差地别，也成为传奇前辈们的一个个标志**

一看到这个名字就知道和resnet和inception（googlenet 即是inception-v1）逃脱不了干系，就是一个比较复杂的网络结构，具体多复杂？！玩过tflearn的去看看我写的代码，run下 然后从tensorboard的graph打开看看，（之前一个被merge的版本后来发现没有batch normalization）改了的提了PR但是在写博客的时候还没有被merge[add inception-resnet-v2 in branch inception-resnet-v2 #450][2]。总之就是"丹方"特别复杂，具体去结合[Inception-v4, Inception-ResNet and the Impact of Residual Connections on Learning][3]，了解过resnet和googlenet的网络结构的小伙伴应该很容易弄明白，以下tflearn的代码参考[tf.slim下inception-resnet-v2][4]。 基本的代码结构：

    # -*- coding: utf-8 -*-
    
    """ inception_resnet_v2.
    
    Applying 'inception_resnet_v2' to Oxford's 17 Category Flower Dataset classification task.
    
    References:
        Inception-v4, Inception-ResNet and the Impact of Residual Connections
        on Learning
      Christian Szegedy, Sergey Ioffe, Vincent Vanhoucke, Alex Alemi.
    
    Links:
        http://arxiv.org/abs/1602.07261
    
    """
    
    from __future__ import division, print_function, absolute_import
    import tflearn
    from tflearn.layers.core import input_data, dropout, flatten, fully_connected
    from tflearn.layers.conv import conv_2d, max_pool_2d, avg_pool_2d
    from tflearn.utils import repeat
    from tflearn.layers.merge_ops import merge
    from tflearn.data_utils import shuffle, to_categorical
    import tflearn.activations as activations
    import tflearn.datasets.oxflower17 as oxflower17
    def block35(net, scale=1.0, activation='relu'):
        tower_conv = conv_2d(net, 32, 1, normalizer_fn='batch_normalization', activation='relu', name='Conv2d_1x1')
        tower_conv1_0 = conv_2d(net, 32, 1, normalizer_fn='batch_normalization', activation='relu',name='Conv2d_0a_1x1')
        tower_conv1_1 = conv_2d(tower_conv1_0, 32, 3, normalizer_fn='batch_normalization', activation='relu',name='Conv2d_0b_3x3')
        tower_conv2_0 = conv_2d(net, 32, 1, normalizer_fn='batch_normalization', activation='relu', name='Conv2d_0a_1x1')
        tower_conv2_1 = conv_2d(tower_conv2_0, 48,3, normalizer_fn='batch_normalization', activation='relu', name='Conv2d_0b_3x3')
        tower_conv2_2 = conv_2d(tower_conv2_1, 64,3, normalizer_fn='batch_normalization', activation='relu', name='Conv2d_0c_3x3')
        tower_mixed = merge([tower_conv, tower_conv1_1, tower_conv2_2], mode='concat', axis=3)
        tower_out = conv_2d(tower_mixed, net.get_shape()[3], 1, normalizer_fn='batch_normalization', activation=None, name='Conv2d_1x1')
        net += scale * tower_out
        if activation:
            if isinstance(activation, str):
                net = activations.get(activation)(net)
            elif hasattr(activation, '__call__'):
                net = activation(net)
            else:
                raise ValueError("Invalid Activation.")
        return net
    
    def block17(net, scale=1.0, activation='relu'):
        tower_conv = conv_2d(net, 192, 1, normalizer_fn='batch_normalization', activation='relu', name='Conv2d_1x1')
        tower_conv_1_0 = conv_2d(net, 128, 1, normalizer_fn='batch_normalization', activation='relu', name='Conv2d_0a_1x1')
        tower_conv_1_1 = conv_2d(tower_conv_1_0, 160,[1,7], normalizer_fn='batch_normalization', activation='relu',name='Conv2d_0b_1x7')
        tower_conv_1_2 = conv_2d(tower_conv_1_1, 192, [7,1], normalizer_fn='batch_normalization', activation='relu',name='Conv2d_0c_7x1')
        tower_mixed = merge([tower_conv,tower_conv_1_2], mode='concat', axis=3)
        tower_out = conv_2d(tower_mixed, net.get_shape()[3], 1, normalizer_fn='batch_normalization', activation=None, name='Conv2d_1x1')
        net += scale * tower_out
        if activation:
            if isinstance(activation, str):
                net = activations.get(activation)(net)
            elif hasattr(activation, '__call__'):
                net = activation(net)
            else:
                raise ValueError("Invalid Activation.")
        return net
    
    
    def block8(net, scale=1.0, activation='relu'):
        """
        """
        tower_conv = conv_2d(net, 192, 1, normalizer_fn='batch_normalization', activation='relu',name='Conv2d_1x1')
        tower_conv1_0 = conv_2d(net, 192, 1, normalizer_fn='batch_normalization', activation='relu', name='Conv2d_0a_1x1')
        tower_conv1_1 = conv_2d(tower_conv1_0, 224, [1,3], normalizer_fn='batch_normalization', name='Conv2d_0b_1x3')
        tower_conv1_2 = conv_2d(tower_conv1_1, 256, [3,1], normalizer_fn='batch_normalization', name='Conv2d_0c_3x1')
        tower_mixed = merge([tower_conv,tower_conv1_2], mode='concat', axis=3)
        tower_out = conv_2d(tower_mixed, net.get_shape()[3], 1, normalizer_fn='batch_normalization', activation=None, name='Conv2d_1x1')
        net += scale * tower_out
        if activation:
            if isinstance(activation, str):
                net = activations.get(activation)(net)
            elif hasattr(activation, '__call__'):
                net = activation(net)
            else:
                raise ValueError("Invalid Activation.")
        return net
    
    # Data loading and preprocessing
    import tflearn.datasets.oxflower17 as oxflower17
    X, Y = oxflower17.load_data(one_hot=True, resize_pics=(299, 299))
    
    num_classes = 17
    dropout_keep_prob = 0.8
    
    network = input_data(shape=[None, 299, 299, 3])
    conv1a_3_3 = conv_2d(network, 32, 3, strides=2, normalizer_fn='batch_normalization', padding='VALID',activation='relu',name='Conv2d_1a_3x3')
    conv2a_3_3 = conv_2d(conv1a_3_3, 32, 3, normalizer_fn='batch_normalization', padding='VALID',activation='relu', name='Conv2d_2a_3x3')
    conv2b_3_3 = conv_2d(conv2a_3_3, 64, 3, normalizer_fn='batch_normalization', activation='relu', name='Conv2d_2b_3x3')
    maxpool3a_3_3 = max_pool_2d(conv2b_3_3, 3, strides=2, padding='VALID', name='MaxPool_3a_3x3')
    conv3b_1_1 = conv_2d(maxpool3a_3_3, 80, 1, normalizer_fn='batch_normalization', padding='VALID',activation='relu', name='Conv2d_3b_1x1')
    conv4a_3_3 = conv_2d(conv3b_1_1, 192, 3, normalizer_fn='batch_normalization', padding='VALID',activation='relu', name='Conv2d_4a_3x3')
    maxpool5a_3_3 = max_pool_2d(conv4a_3_3, 3, strides=2, padding='VALID', name='MaxPool_5a_3x3')
    
    tower_conv = conv_2d(maxpool5a_3_3, 96, 1, normalizer_fn='batch_normalization', activation='relu', name='Conv2d_5b_b0_1x1')
    
    tower_conv1_0 = conv_2d(maxpool5a_3_3, 48, 1, normalizer_fn='batch_normalization', activation='relu', name='Conv2d_5b_b1_0a_1x1')
    tower_conv1_1 = conv_2d(tower_conv1_0, 64, 5, normalizer_fn='batch_normalization', activation='relu', name='Conv2d_5b_b1_0b_5x5')
    
    tower_conv2_0 = conv_2d(maxpool5a_3_3, 64, 1, normalizer_fn='batch_normalization', activation='relu', name='Conv2d_5b_b2_0a_1x1')
    tower_conv2_1 = conv_2d(tower_conv2_0, 96, 3, normalizer_fn='batch_normalization', activation='relu', name='Conv2d_5b_b2_0b_3x3')
    tower_conv2_2 = conv_2d(tower_conv2_1, 96, 3, normalizer_fn='batch_normalization', activation='relu',name='Conv2d_5b_b2_0c_3x3')
    
    tower_pool3_0 = avg_pool_2d(maxpool5a_3_3, 3, strides=1, padding='same', name='AvgPool_5b_b3_0a_3x3')
    tower_conv3_1 = conv_2d(tower_pool3_0, 64, 1, normalizer_fn='batch_normalization', activation='relu',name='Conv2d_5b_b3_0b_1x1')
    
    tower_5b_out = merge([tower_conv, tower_conv1_1, tower_conv2_2, tower_conv3_1], mode='concat', axis=3)
    
    net = repeat(tower_5b_out, 10, block35, scale=0.17)tower_conv2_2 = conv_2d(tower_conv2_1, 96, 3, normalizer_fn='batch_normalization', activation='relu',name='Conv2d_5b_b2_0c_3x3')
    
    tower_pool3_0 = avg_pool_2d(maxpool5a_3_3, 3, strides=1, padding='same', name='AvgPool_5b_b3_0a_3x3')
    tower_conv3_1 = conv_2d(tower_pool3_0, 64, 1, normalizer_fn='batch_normalization', activation='relu',name='Conv2d_5b_b3_0b_1x1')
    
    tower_5b_out = merge([tower_conv, tower_conv1_1, tower_conv2_2, tower_conv3_1], mode='concat', axis=3)
    
    net = repeat(tower_5b_out, 10, block35, scale=0.17)
    
    tower_conv = conv_2d(net, 384, 3, normalizer_fn='batch_normalization', strides=2,activation='relu', padding='VALID', name='Conv2d_6a_b0_0a_3x3')
    tower_conv1_0 = conv_2d(net, 256, 1, normalizer_fn='batch_normalization', activation='relu', name='Conv2d_6a_b1_0a_1x1')
    tower_conv1_1 = conv_2d(tower_conv1_0, 256, 3, normalizer_fn='batch_normalization', activation='relu', name='Conv2d_6a_b1_0b_3x3')
    tower_conv1_2 = conv_2d(tower_conv1_1, 384, 3, normalizer_fn='batch_normalization', strides=2, padding='VALID', activation='relu',name='Conv2d_6a_b1_0c_3x3')
    tower_pool = max_pool_2d(net, 3, strides=2, padding='VALID',name='MaxPool_1a_3x3')
    net = merge([tower_conv, tower_conv1_2, tower_pool], mode='concat', axis=3)
    net = repeat(net, 20, block17, scale=0.1)
    
    tower_conv = conv_2d(net, 256, 1, normalizer_fn='batch_normalization', activation='relu', name='Conv2d_0a_1x1')
    tower_conv0_1 = conv_2d(tower_conv, 384, 3, normalizer_fn='batch_normalization', strides=2, padding='VALID', activation='relu',name='Conv2d_0a_1x1')
    
    tower_conv1 = conv_2d(net, 256, 1, normalizer_fn='batch_normalization', padding='VALID', activation='relu',name='Conv2d_0a_1x1')
    tower_conv1_1 = conv_2d(tower_conv1,288,3, normalizer_fn='batch_normalization', strides=2, padding='VALID',activation='relu', name='COnv2d_1a_3x3')
    
    tower_conv2 = conv_2d(net, 256,1, normalizer_fn='batch_normalization', activation='relu',name='Conv2d_0a_1x1')
    tower_conv2_1 = conv_2d(tower_conv2, 288,3, normalizer_fn='batch_normalization', name='Conv2d_0b_3x3',activation='relu')
    tower_conv2_2 = conv_2d(tower_conv2_1, 320, 3, normalizer_fn='batch_normalization', strides=2, padding='VALID',activation='relu', name='Conv2d_1a_3x3')
    
    tower_pool = max_pool_2d(net, 3, strides=2, padding='VALID', name='MaxPool_1a_3x3')
    net = merge([tower_conv0_1, tower_conv1_1,tower_conv2_2, tower_pool], mode='concat', axis=3)
    
    net = repeat(net, 9, block8, scale=0.2)
    net = block8(net, activation=None)
    
    net = conv_2d(net, 1536, 1, normalizer_fn='batch_normalization', activation='relu', name='Conv2d_7b_1x1')
    net = avg_pool_2d(net, net.get_shape().as_list()[1:3],strides=2, padding='VALID', name='AvgPool_1a_8x8')
    net = flatten(net)
    net = dropout(net, dropout_keep_prob)
    loss = fully_connected(net, num_classes,activation='softmax')
    
    
    network = tflearn.regression(loss, optimizer='RMSprop',
                         loss='categorical_crossentropy',
                         learning_rate=0.0001)
    model = tflearn.DNN(network, checkpoint_path='inception_resnet_v2',
                        max_checkpoints=1, tensorboard_verbose=2, tensorboard_dir="./tflearn_logs/")
    model.fit(X, Y, n_epoch=1000, validation_set=0.1, shuffle=True,
              show_metric=True, batch_size=32, snapshot_step=2000,
              snapshot_epoch=False, run_id='inception_resnet_v2_17flowers')
    
    tower_conv = conv_2d(net, 384, 3, normalizer_fn='batch_normalization', strides=2,activation='relu', padding='VALID', name='Conv2d_6a_b0_0a_3x3')
    tower_conv1_0 = conv_2d(net, 256, 1, normalizer_fn='batch_normalization', activation='relu', name='Conv2d_6a_b1_0a_1x1')
    tower_conv1_1 = conv_2d(tower_conv1_0, 256, 3, normalizer_fn='batch_normalization', activation='relu', name='Conv2d_6a_b1_0b_3x3')
    tower_conv1_2 = conv_2d(tower_conv1_1, 384, 3, normalizer_fn='batch_normalization', strides=2, padding='VALID', activation='relu',name='Conv2d_6a_b1_0c_3x3')
    tower_pool = max_pool_2d(net, 3, strides=2, padding='VALID',name='MaxPool_1a_3x3')
    net = merge([tower_conv, tower_conv1_2, tower_pool], mode='concat', axis=3)
    net = repeat(net, 20, block17, scale=0.1)
    
    tower_conv = conv_2d(net, 256, 1, normalizer_fn='batch_normalization', activation='relu', name='Conv2d_0a_1x1')
    tower_conv0_1 = conv_2d(tower_conv, 384, 3, normalizer_fn='batch_normalization', strides=2, padding='VALID', activation='relu',name='Conv2d_0a_1x1')
    
    tower_conv1 = conv_2d(net, 256, 1, normalizer_fn='batch_normalization', padding='VALID', activation='relu',name='Conv2d_0a_1x1')
    tower_conv1_1 = conv_2d(tower_conv1,288,3, normalizer_fn='batch_normalization', strides=2, padding='VALID',activation='relu', name='COnv2d_1a_3x3')
    
    tower_conv2 = conv_2d(net, 256,1, normalizer_fn='batch_normalization', activation='relu',name='Conv2d_0a_1x1')
    tower_conv2_1 = conv_2d(tower_conv2, 288,3, normalizer_fn='batch_normalization', name='Conv2d_0b_3x3',activation='relu')
    tower_conv2_2 = conv_2d(tower_conv2_1, 320, 3, normalizer_fn='batch_normalization', strides=2, padding='VALID',activation='relu', name='Conv2d_1a_3x3')
    
    tower_pool = max_pool_2d(net, 3, strides=2, padding='VALID', name='MaxPool_1a_3x3')
    net = merge([tower_conv0_1, tower_conv1_1,tower_conv2_2, tower_pool], mode='concat', axis=3)
    
    net = repeat(net, 9, block8, scale=0.2)
    net = block8(net, activation=None)
    
    net = conv_2d(net, 1536, 1, normalizer_fn='batch_normalization', activation='relu', name='Conv2d_7b_1x1')
    net = avg_pool_2d(net, net.get_shape().as_list()[1:3],strides=2, padding='VALID', name='AvgPool_1a_8x8')
    net = flatten(net)
    net = dropout(net, dropout_keep_prob)
    loss = fully_connected(net, num_classes,activation='softmax')
    
    
    network = tflearn.regression(loss, optimizer='RMSprop',
                         loss='categorical_crossentropy',
                         learning_rate=0.0001)
    model = tflearn.DNN(network, checkpoint_path='inception_resnet_v2',
                        max_checkpoints=1, tensorboard_verbose=2, tensorboard_dir="./tflearn_logs/")
    model.fit(X, Y, n_epoch=1000, validation_set=0.1, shuffle=True,
              show_metric=True, batch_size=32, snapshot_step=2000,
              snapshot_epoch=False, run_id='inception_resnet_v2_17flowers')
    

想要run下的可以去使用下tflearn，注意更改conv_2d里面的内容，我这里在本身conv_2d上加了个normalizer_fn，来使用batch_normalization。

## MXnet 炼丹

**不同的丹炉，即使是相同的丹方，炼丹的方式都不仅相同** 在打算用MXnet实现inception-resnet-v2之前，除了mxnet-101里面的代码，基本没有写过mxnet，但是没关系，不怕，有很多其他大神写的丹方，这里具体参考了[symbol_inception-bn.py][5]。首先，为了减少代码条数，参考创建一个ConvFactory，但是和inception-bn不同的是，inception-resnet-v2要考虑是否要激活函数的版本。所以inception-resnet-v2的ConvFactory如下：

    def ConvFactory(data, num_filter, kernel, stride=(1,1), pad=(0, 0), act_type="relu", mirror_attr={},with_act=True):
        conv = mx.symbol.Convolution(data=data, num_filter=num_filter, kernel=kernel, stride=stride, pad=pad)
        bn = mx.symbol.BatchNorm(data=conv)
        if with_act:
            act = mx.symbol.Activation(data = bn, act_type=act_type, attr=mirror_attr)
            return act
        else:
            return bn
    

然后就简单了，按照网络一路往下写：

    def get_symbol(num_classes=1000，input_data_shape=(64,3,299,299)):
        data = mx.symbol.Variable(name='data')
        conv1a_3_3 = ConvFactory(data=data, num_filter=32, kernel=(3,3), stride=(2, 2))
        conv2a_3_3 = ConvFactory(conv1a_3_3, 32, (3,3))
        conv2b_3_3 = ConvFactory(conv2a_3_3, 64, (3,3), pad=(1,1))
        maxpool3a_3_3 = mx.symbol.Pooling(data=conv2b_3_3, kernel=(3, 3), stride=(2, 2), pool_type='max')
        conv3b_1_1 = ConvFactory(maxpool3a_3_3, 80 ,(1,1))
        conv4a_3_3 = ConvFactory(conv3b_1_1, 192, (3,3))
        maxpool5a_3_3 = mx.symbol.Pooling(data=conv4a_3_3, kernel=(3,3), stride=(2,2), pool_type='max')
    
        tower_conv = ConvFactory(maxpool5a_3_3, 96, (1,1))
        tower_conv1_0 = ConvFactory(maxpool5a_3_3, 48, (1,1))
        tower_conv1_1 = ConvFactory(tower_conv1_0, 64, (5,5), pad=(2,2))
    
        tower_conv2_0 = ConvFactory(maxpool5a_3_3, 64, (1,1))
        tower_conv2_1 = ConvFactory(tower_conv2_0, 96, (3,3), pad=(1,1))
        tower_conv2_2 = ConvFactory(tower_conv2_1, 96, (3,3), pad=(1,1))
    
        tower_pool3_0 = mx.symbol.Pooling(data=maxpool5a_3_3, kernel=(3,3), stride=(1,1),pad=(1,1), pool_type='avg')
        tower_conv3_1 = ConvFactory(tower_pool3_0, 64, (1,1))
        tower_5b_out = mx.symbol.Concat(*[tower_conv, tower_conv1_1, tower_conv2_2, tower_conv3_1])
    

然后就不对了，要重复条用一个block35的结构，repeat函数很容易实现，给定调用次数，调用函数，参数， 多次调用就好了：

    def repeat(inputs, repetitions, layer, *args, **kwargs):
        outputs = inputs
        for i in range(repetitions):
            outputs = layer(outputs, *args, **kwargs)
        return outputs
    

这里很简单，但是block35就有问题啦，这个子结构的目的要输出与输入同样大小的channel数，之前因为在tensorflow下写的，很容易拿到一个Variable的shape，但是在MXnet上就很麻烦，这里不知道怎么做，提了个issue [How can i get the shape with the net?][6]，然后就去查api，发现有个infer_shape，mxnet客服部小伙伴也让我用这个去做， 试了试，挺管用能够拿到shape，但是必须给入一个4d的tensor的shape，比如(64,3,299,299)，他会在graph运行时infer到对应symbol的shape，然后就这么写了:

    def block35(net, input_data_shape, scale=1.0, with_act=True, act_type='relu', mirror_attr={}):
        assert len(input_data_shape) == 4, 'input_data_shape should be len of 4, your \
                                input_data_shape is len of %d'%len(input_data_shape)
        _, out_shape,_ = net.infer_shape(data=input_data_shape)
        tower_conv = ConvFactory(net, 32, (1,1))
        tower_conv1_0 = ConvFactory(net, 32, (1,1))
        tower_conv1_1 = ConvFactory(tower_conv1_0, 32, (3, 3), pad=(1,1))
        tower_conv2_0 = ConvFactory(net, 32, (1,1))
        tower_conv2_1 = ConvFactory(tower_conv2_0, 48, (3, 3), pad=(1,1))
        tower_conv2_2 = ConvFactory(tower_conv2_1, 64, (3, 3), pad=(1,1))
        tower_mixed = mx.symbol.Concat(*[tower_conv, tower_conv1_1, tower_conv2_2])
        tower_out = ConvFactory(tower_mixed, out_shape[0][1], (1,1), with_act=False)
    
        net += scale * tower_out
        if with_act:
            act = mx.symbol.Activation(data = net, act_type=act_type, attr=mirror_attr)
            return act
        else:
            return net
    

大家是不是感到很别扭，我也觉得很别扭，但是我一直是个『不拘小节』的工程师，对这块不斤斤计较，所以，写完这块之后也觉得就成了接下来就是block17， block8， 这里都很简单的很类似，就不提了。

然后就接下来一段，很快就完成了：

    net = repeat(tower_5b_out, 10, block35, scale=0.17, input_num_channels=320)
    tower_conv = ConvFactory(net, 384, (3,3),stride=(2,2))
    tower_conv1_0 = ConvFactory(net, 256, (1,1))
    tower_conv1_1 = ConvFactory(tower_conv1_0, 256, (3,3), pad=(1,1))
    tower_conv1_2 = ConvFactory(tower_conv1_1, 384, (3,3),stride=(2,2))
    tower_pool = mx.symbol.Pooling(net, kernel=(3,3), stride=(2,2), pool_type='max')
    net = mx.symbol.Concat(*[tower_conv, tower_conv1_2, tower_pool])
    net = repeat(net, 20, block17, scale=0.1, input_num_channels=1088)
    tower_conv = ConvFactory(net, 256, (1,1))
    tower_conv0_1 = ConvFactory(tower_conv, 384, (3,3), stride=(2,2))
    tower_conv1 = ConvFactory(net, 256, (1,1))
    tower_conv1_1 = ConvFactory(tower_conv1, 288, (3,3), stride=(2,2))
    tower_conv2 = ConvFactory(net, 256, (1,1))
    tower_conv2_1 = ConvFactory(tower_conv2, 288, (3,3), pad=(1,1))
    tower_conv2_2 = ConvFactory(tower_conv2_1, 320, (3,3),  stride=(2,2))
    tower_pool = mx.symbol.Pooling(net, kernel=(3,3), stride=(2,2), pool_type='max')
    net = mx.symbol.Concat(*[tower_conv0_1, tower_conv1_1, tower_conv2_2, tower_pool])
    
    net = repeat(net, 9, block8, scale=0.2, input_num_channels=2080)
    net = block8(net, with_act=False, input_num_channel=2080)
    
    net = ConvFactory(net, 1536, (1,1))
    net = mx.symbol.Pooling(net, kernel=(1,1), global_pool=True, stride=(2,2), pool_type='avg')
    net = mx.symbol.Flatten(net)
    net = mx.symbol.Dropout(data=net,p= 0.8)
    net = mx.symbol.FullyConnected(data=net,num_hidden=num_classes)
    softmax = mx.symbol.SoftmaxOutput(data=net, name='softmax')
    

感觉很开心，写完了，这么简单，大家先忽略先忽悠所有的pad值，因为找不到没有pad的版本，所以大家先忽略下。然后就是写样例测试呀，又是17flowers这个数据集，参考mxnet-101中如何把dataset转换为binary， 首先写个py来get到所有的图像list，index还有他的label_index，这个很快就解决了。具体参考我这里的[mxnet-101][1] 然后就是拿数据开始run啦，Ready？ Go！

咦,车开不起来，不对，都是些什么鬼？ infer_shape 有问题？ 没事 查api tensorflow中padding是"valid"和"same"，mxnet中没有， 没有...，要自己计算，什么鬼？没有valid，same，我不会呀！！！写了这么久，就不写了？不行，找下怎么搞定，看了tensorflow的文档，翻了资料，same就是保证input与output保持一致，valid就无所谓，不需要设置pad，所以当tensorflow中有same的时候，就需要在mxnet中设置对应的pad值，kernel为3的时候pad=1， kernel=5，pad=2。这里改来改去，打印出每一层网络后的shape，前后花了我大概6个小时，终于让我一步一步debug出来了，但是不对，在repeat 10次block35后，怎么和tf.slim的inception-resnet-v2的注释的shape不同？我了个擦，当时已经好像快凌晨4点了，本以为run起来了，怎么就解释不通呢？不会tensorflow的注释有问题吧？我了个擦，老美真是数学有点问题，提了个issue，很快就有人fix然后commit了 [may be an error in slim.nets.inception_resnet_v2 #634][7]，不过貌似到现在还没有被merge。

一切ok，开始run了，用17flowers，很快可以收敛，没有更多的资源来测试更大的数据集，就直接提交了，虽然代码很烂，但怎么着也是一步一步写出来的，可是，始终确实是有点问题，后来经过github的好心人指点肯定也是一个大牛，告诉我Pooling 有个global_pool来做全局的池化，我了个擦，这么好的东西，tensorflow上可没有，所以tensorflow上用的是通过get_shape拿到对应的tensor的width和height来做pooling，我也二笔的在mxne它里面这样用，所以需要input_shape_shape来infer到所在layer的shape，来做全局池化，有了这个，我还infer_shape个什么鬼，blockxx里面也不需要了，channel数可以直接手工计算，传一个channel数就好了，get_symbol也可以保持和原来一样不需要传什么input_data_shape啦！！！感谢[zhreshold][8]的提示，一切都ok，更改了，但是后面mxnet的大神在重构一些代码，还没有merge，不过没有关系，等他们ok了 我再把inception-resnet-v2整理下，再提pr（教练，我想当mxnet contributor）。

    def block35(net, input_num_channels, scale=1.0, with_act=True, act_type='relu', mirror_attr={}):
        tower_conv = ConvFactory(net, 32, (1,1))
        tower_conv1_0 = ConvFactory(net, 32, (1,1))
        tower_conv1_1 = ConvFactory(tower_conv1_0, 32, (3, 3), pad=(1,1))
        tower_conv2_0 = ConvFactory(net, 32, (1,1))
        tower_conv2_1 = ConvFactory(tower_conv2_0, 48, (3, 3), pad=(1,1))
        tower_conv2_2 = ConvFactory(tower_conv2_1, 64, (3, 3), pad=(1,1))
        tower_mixed = mx.symbol.Concat(*[tower_conv, tower_conv1_1, tower_conv2_2])
        tower_out = ConvFactory(tower_mixed, input_num_channels, (1,1), with_act=False)
    
        net += scale * tower_out
        if with_act:
            act = mx.symbol.Activation(data = net, act_type=act_type, attr=mirror_attr)
            return act
        else:
            return net
    

一直到这里，inception-resnet-v2就写出来了，但是只是测试了小数据集，后来在zhihu上偶遇李沐大神，果断上去套近乎，最后拿到一个一台机器，就在测大一点的数据集，其实也不大，102flowers，之后会请沐神帮忙扩展一个大点的盘来放下ImageNet，测试一下性能，不过现在102flowers也还行，效果还不错。

## 丹成

**金丹品阶高低，以丹纹记，不同炼丹材料丹纹不同，评判标准也不同，acc是最常用判断金丹品阶高低的手段**

将102flower按9：1分成训练集和验证集，设置300个epoch(数据集比较小，貌似设置多点epoch才能有比较好的性能，看有小伙伴用inception-bn在imagenet上只需要50个epoch)，网络inception-resnet-v2确实大，如此小的数据集上300 epoch大概也需要1天，不过对比tensorflow那是快多了。 ![][9]

## 编不下去了（predict）

这里，会简单地写一个inference的例子，算作学习如果使用训练好的model，注意还是最好使用python的opencv，因为mxnet官方是用的opencv，使用cv2这个库，我在网上找的使用skimage的库，做出来的始终有问题，应该是brg2rgb的问题，使用cv2的`cv2.cvtColor(img, cv2.COLOR_BGR2RGB`之后会成功:

    import mxnet as mx
    import logging
    import numpy as np
    import cv2
    import scipy.io as sio
    
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    num_round = 260
    prefix = "102flowers"
    model = mx.model.FeedForward.load(prefix, num_round, ctx=mx.cpu(), numpy_batch_size=1)
    # synset = [l.strip() for l in open('Inception/synset.txt').readlines()]
    
    
    def PreprocessImage(path, show_img=False):
            # load image
            img = cv2.imread(path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            mean_img = mx.nd.load('mean.bin').values()[0].asnumpy()
            print img.shape
            print mean_img.shape
            img = cv2.resize(img,(299,299))
            img = np.swapaxes(img, 0, 2)
            img = np.swapaxes(img, 1, 2)
            img = img -mean_img
            img = img[np.newaxis, :]
            print img.shape
    
            return img
    
        right = 0
        sum = 0
        with open('test.lst', 'r') as fread:
            for line in fread.readlines()[:20]:
                sum +=1
                batch  = '../day2/102flowers/' + line.split("\t")[2].strip("\n")
                print batch
                batch = PreprocessImage(batch, False)
                prob = model.predict(batch)[0]
                pred = np.argsort(prob)[::-1]
                # # Get top1 label
                # top1 = synset[pred[0]]
                top_1 = pred[0]
                if top_1 == int(line.split("\t")[1]):
                    print 'top1 right'
                    right += 1
    
        print 'top 1 accuracy: %f '%(right/(1.0*sum))
    

使用第260个epoch的模型weight，这里因为手贱删除了9：1时的test.lst，只能用7：3是的test.lst暂时做计算，最后accuracy应该会比较偏高，不过这不是重点。

## 总结（继续编不下去了）

在这样一次畅快淋漓的mxnet之旅后，总结一下遇到的几个坑，与大家分享：

*   无法直接拿到tensor的shape信息，通过infer_shape，在设计代码时走了很多；
*   im2rec时，准备的train.lst, test.lst未shuffle，在102flowers上我都没有发觉，在后面做鉴黄的training的时候发现开始training accuracy，分析可能是train.lst未shuffle的问题（以为在ImageRecordIter中有shuffle参数，就不需要），改了后没有training accuracy从开始就为1的情况；
*   pad值的问题，翻阅了很多资料才解决，文档也没有特别多相关的，对于我这种从tensorflow转mxnet的小伙伴来说是个比较大的坑；
*   predict的问题，找了mxnet github的源上的example，并不能成功，在找官网上的example发现使用的是cv2，并不是一些例子当中的skimage，考虑到mxnet在安装时需要opencv，可能cv2和skimage在一些标准上有差异，就改用cv2的predict版本，还有读入图片之后要`cv2.cvtColor(img, cv2.COLOR_BGR2RGB)`.
*   还是predict的问题，在mxnet中，构造ImageRecordIter时没有指定mean.bin，但是并不是说计算的时候不会减到均值图片在训练，开始误解为不需要减到均值图片，后来发现一直不正确，考虑到train的时候会自己生成mean.bin，猜测可能是这里的问题，通过`mean_img = mx.nd.load('mean.bin').values()[0].asnumpy()`读入后，在原始图片减去均值图，结果ok；但整个流程相对于tf.slim的predict还是比较复杂的。

优点：

*   速度快，速度快，速度快，具体指没有做测量，但是相对于tensorflow至少两到三倍；
*   占用内存低， 同样batch和模型，12g的显存，tf会爆，但是mxnet只需要占用7g多点；
*   im2rec很方便，相对于tensorflow下tfrecord需要写部分代码，更容易入手，但是切记自己生成train.lst, test.lst的时候要shuffle；
*   Pooling下的global_pool是个好东西，tensorflow没有；

之后会在ImageNet Dataset做一下测试，感觉会更有意思

 [1]: https://github.com/burness/mxnet-101
 [2]: https://github.com/tflearn/tflearn/pull/450
 [3]: https://arxiv.org/pdf/1602.07261.pdf
 [4]: https://github.com/tensorflow/models/blob/master/slim/nets/inception_resnet_v2.py
 [5]: https://github.com/burness/mxnet/blob/master/example/image-classification/symbol_inception-bn.py
 [6]: https://github.com/dmlc/mxnet/issues/3796
 [7]: https://github.com/tensorflow/models/issues/634#issuecomment-260399899
 [8]: https://github.com/zhreshold
 [9]: ./images/inception-resnet-v2-102flowers.png