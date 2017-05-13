## 前言

前面几篇文章讲述了在Computer Vision领域里面常用的模型，接下来一段时间，我会花精力来学习一些TensorFlow在Computer Vision领域的应用，主要是分析相关pape和源码，今天会来详细了解下[fast neural style][1]的相关工作，前面也有文章分析neural style的内容，那篇算是neural style的起源，但是无法应用到实际工作上，为啥呢？它每次都需要指定好content image和style image，然后最小化content loss 和style loss去生成图像，时间花销很大，而且无法保存某种风格的model，所以每次生成图像都是训练一个model的过程，而[fast neural style][1]中能够将训练好的某种style的image的模型保存下来，然后对content image 进行transform，当然文中还提到了image transform的另一个应用方向：Super-Resolution，利用深度学习的技术将低分辨率的图像转换为高分辨率图像，现在在很多大型的互联网公司，尤其是视频网站上也有应用。

## Paper原理

几个月前，就看了Neural Style相关的文章[TensorFlow之深入理解Neural Style][2],[A Neural Algorithm of Aritistic Style][3]中构造了一个多层的卷积网络，通过最小化定义的content loss和style loss最后生成一个结合了content和style的图像，很有意思，而[Perceptual Losses for Real-Time Style Transfer and Super-Resolution][1]，通过使用perceptual loss来替代per-pixels loss使用pre-trained的vgg model来简化原先的loss计算，增加一个transform Network，直接生成Content image的style版本， 如何实现的呢，请看下图，容我道来： ![][4] 整个网络是由部分组成：image transformation network、 loss netwrok；Image Transformation network是一个deep residual conv netwrok，用来将输入图像（content image）直接transform为带有style的图像；而loss network参数是fixed的，这里的loss network和[A Neural Algorithm of Aritistic Style][3]中的网络结构一致，只是参数不做更新（**评论中的hanz和s123提出的很对，neural style的weight也是常数，不同的是像素级loss和per loss的区别，neural style里面是更更新像素，得到最后的合成后的照片，这里我之前可能理解有点小问题，谢谢指出**），只用来做content loss 和style loss的计算，这个就是所谓的perceptual loss，作者是这样解释的为Image Classification的pretrained的卷积模型已经很好的学习了perceptual和semantic information（场景和语义信息），所以后面的整个loss network仅仅是为了计算content loss和style loss，而不像[A Neural Algorithm of Aritistic Style][3]做更新这部分网络的参数，这里更新的是前面的transform network的参数，所以从整个网络结构上来看输入图像通过transform network得到转换的图像，然后计算对应的loss，整个网络通过最小化这个loss去update前面的transform network，是不是很简单？

loss的计算也和之前的都很类似，content loss： ![][5] style loss: ![][5] style loss中的gram matrix: ![][6] Gram Matrix是一个很重要的东西，他可以保证y^hat和y之间有同样的shape。 Gram的说明具体见paper这部分，我这也解释不清楚，相信读者一看就明白： ![][7]

相信看到这里就基本上明白了这篇paper在fast neural style是如何做的，总结一下：

*   transform network 网络结构为deep residual network，将输入image转换为带有特种风格的图像，网络参数可更新。
*   loss network 网络结构同之前paper类似，这里主要是计算content loss和style loss， 注意不像neural style里面会对图像做像素级更新更新。
*   Gram matrix的提出，让transform之后的图像与最后经过loss network之后的图像不同shape时计算loss也很方便。

## fast neural style on tensorflow

代码参考<https://github.com/OlavHN/fast-neural-style>，但是我跑了下，代码是跑不通的，原因大概是tensorflow在更新之后，local_variables之后的一些问题，具体原因可以看这个issue:<https://github.com/tensorflow/tensorflow/issues/1045#issuecomment-239789244>.还有这个项目的代码都写在一起，有点杂乱，我将train和最后生成style后的图像的代码分开了，项目放到了我的个人的github [neural_style_tensorflow][8]，项目基本要求：

*   python 2.7.x
*   [Tensorflow r0.10][9]
*   [VGG-19 model][10]
*   [COCO dataset][11]

Transform Network网络结构

    import tensorflow as tf
    
    def conv2d(x, input_filters, output_filters, kernel, strides, padding='SAME'):
        with tf.variable_scope('conv') as scope:
    
            shape = [kernel, kernel, input_filters, output_filters]
            weight = tf.Variable(tf.truncated_normal(shape, stddev=0.1), name='weight')
            convolved = tf.nn.conv2d(x, weight, strides=[1, strides, strides, 1], padding=padding, name='conv')
    
            normalized = batch_norm(convolved, output_filters)
    
            return normalized
    
    def conv2d_transpose(x, input_filters, output_filters, kernel, strides, padding='SAME'):
        with tf.variable_scope('conv_transpose') as scope:
    
            shape = [kernel, kernel, output_filters, input_filters]
            weight = tf.Variable(tf.truncated_normal(shape, stddev=0.1), name='weight')
    
            batch_size = tf.shape(x)[0]
            height = tf.shape(x)[1] * strides
            width = tf.shape(x)[2] * strides
            output_shape = tf.pack([batch_size, height, width, output_filters])
            convolved = tf.nn.conv2d_transpose(x, weight, output_shape, strides=[1, strides, strides, 1], padding=padding, name='conv_transpose')
    
            normalized = batch_norm(convolved, output_filters)
            return normalized
    
    def batch_norm(x, size):
        batch_mean, batch_var = tf.nn.moments(x, [0, 1, 2], keep_dims=True)
        beta = tf.Variable(tf.zeros([size]), name='beta')
        scale = tf.Variable(tf.ones([size]), name='scale')
        epsilon = 1e-3
        return tf.nn.batch_normalization(x, batch_mean, batch_var, beta, scale, epsilon, name='batch')
    
    def residual(x, filters, kernel, strides, padding='SAME'):
        with tf.variable_scope('residual') as scope:
            conv1 = conv2d(x, filters, filters, kernel, strides, padding=padding)
            conv2 = conv2d(tf.nn.relu(conv1), filters, filters, kernel, strides, padding=padding)
    
            residual = x + conv2
    
            return residual
    
    def net(image):
        with tf.variable_scope('conv1'):
            conv1 = tf.nn.relu(conv2d(image, 3, 32, 9, 1))
        with tf.variable_scope('conv2'):
            conv2 = tf.nn.relu(conv2d(conv1, 32, 64, 3, 2))
        with tf.variable_scope('conv3'):
            conv3 = tf.nn.relu(conv2d(conv2, 64, 128, 3, 2))
        with tf.variable_scope('res1'):
            res1 = residual(conv3, 128, 3, 1)
        with tf.variable_scope('res2'):
            res2 = residual(res1, 128, 3, 1)
        with tf.variable_scope('res3'):
            res3 = residual(res2, 128, 3, 1)
        with tf.variable_scope('res4'):
            res4 = residual(res3, 128, 3, 1)
        with tf.variable_scope('res5'):
            res5 = residual(res4, 128, 3, 1)
        with tf.variable_scope('deconv1'):
            deconv1 = tf.nn.relu(conv2d_transpose(res5, 128, 64, 3, 2))
        with tf.variable_scope('deconv2'):
            deconv2 = tf.nn.relu(conv2d_transpose(deconv1, 64, 32, 3, 2))
        with tf.variable_scope('deconv3'):
            deconv3 = tf.nn.tanh(conv2d_transpose(deconv2, 32, 3, 9, 1))
    
        y = deconv3 * 127.5
    
        return y
    

使用deep residual network来训练COCO数据集，能够在保证性能的前提下，训练更深的模型。 而Loss Network是有pretrained的VGG网络来计算，网络结构：

    import tensorflow as tf
    import numpy as np
    import scipy.io
    from scipy import misc
    
    
    def net(data_path, input_image):
        layers = (
            'conv1_1', 'relu1_1', 'conv1_2', 'relu1_2', 'pool1',
    
            'conv2_1', 'relu2_1', 'conv2_2', 'relu2_2', 'pool2',
    
            'conv3_1', 'relu3_1', 'conv3_2', 'relu3_2', 'conv3_3',
            'relu3_3', 'conv3_4', 'relu3_4', 'pool3',
    
            'conv4_1', 'relu4_1', 'conv4_2', 'relu4_2', 'conv4_3',
            'relu4_3', 'conv4_4', 'relu4_4', 'pool4',
    
            'conv5_1', 'relu5_1', 'conv5_2', 'relu5_2', 'conv5_3',
            'relu5_3', 'conv5_4', 'relu5_4'
        )
    
        data = scipy.io.loadmat(data_path)
        mean = data['normalization'][0][0][0]
        mean_pixel = np.mean(mean, axis=(0, 1))
        weights = data['layers'][0]
    
        net = {}
        current = input_image
        for i, name in enumerate(layers):
            kind = name[:4]
            if kind == 'conv':
                kernels, bias = weights[i][0][0][0][0]
                # matconvnet: weights are [width, height, in_channels, out_channels]
                # tensorflow: weights are [height, width, in_channels, out_channels]
                kernels = np.transpose(kernels, (1, 0, 2, 3))
                bias = bias.reshape(-1)
                current = _conv_layer(current, kernels, bias, name=name)
            elif kind == 'relu':
                current = tf.nn.relu(current, name=name)
            elif kind == 'pool':
                current = _pool_layer(current, name=name)
            net[name] = current
    
        assert len(net) == len(layers)
        return net, mean_pixel
    
    
    def _conv_layer(input, weights, bias, name=None):
        conv = tf.nn.conv2d(input, tf.constant(weights), strides=(1, 1, 1, 1),
                            padding='SAME', name=name)
        return tf.nn.bias_add(conv, bias)
    
    
    def _pool_layer(input, name=None):
        return tf.nn.max_pool(input, ksize=(1, 2, 2, 1), strides=(1, 2, 2, 1),
                              padding='SAME', name=name)
    
    
    def preprocess(image, mean_pixel):
        return image - mean_pixel
    
    
    def unprocess(image, mean_pixel):
        return image + mean_pixel
    

Content Loss：

    def compute_content_loss(content_layers,net):
        content_loss = 0
        # tf.app.flags.DEFINE_string("CONTENT_LAYERS", "relu4_2", "Which VGG layer to extract content loss from")
        for layer in content_layers:
            generated_images, content_images = tf.split(0, 2, net[layer])
            size = tf.size(generated_images)
            content_loss += tf.nn.l2_loss(generated_images - content_images) / tf.to_float(size)
        content_loss = content_loss / len(content_layers)
    
        return content_loss
    

Style Loss：

    def compute_style_loss(style_features_t, style_layers,net):
        style_loss = 0
        for style_gram, layer in zip(style_features_t, style_layers):
            generated_images, _ = tf.split(0, 2, net[layer])
            size = tf.size(generated_images)
            for style_image in style_gram:
                style_loss += tf.nn.l2_loss(tf.reduce_sum(gram(generated_images) - style_image, 0)) / tf.to_float(size)
        style_loss = style_loss / len(style_layers)
        return style_loss
    

gram：

    def gram(layer):
        shape = tf.shape(layer)
        num_images = shape[0]
        num_filters = shape[3]
        size = tf.size(layer)
        filters = tf.reshape(layer, tf.pack([num_images, -1, num_filters]))
        grams = tf.batch_matmul(filters, filters, adj_x=True) / tf.to_float(size / FLAGS.BATCH_SIZE)
    
        return grams
    

在train_fast_neural_style.py main()中，net, _ = vgg.net(FLAGS.VGG_PATH, tf.concat(0, [generated, images]))这部分有点疑问，按我的想法来说应该分别把generated、images作为input给到vgg.net，然后在后面去计算与content image 和style的loss，但是这里是直接首先把generated和原来的content images先concat by axis=0（具体可以查下tf.concat）然后在输出进行tf.split得到对应网络的输出，这个很有意思，想想CNN在做卷积的时候某个位置的值只与周围相关，共享weight之后，该层彼此之间不相关（可能在generated和image之间就边缘地方有些许pixels的影响，基本可以忽略，我的解释可能就是这样，有其他更合理的解释的小伙伴，请在本文下方留言），这个技巧感觉挺有用的，以后写相关代码的时候可以采纳。

代码图像生成效果不是很好，我怀疑是content和 style之间的weight大小的关系，还有就是可能是epoch数大小的问题，之后我会好好改下权重，看看能不能有点比较好的结果。

--试过好多不同的版本，fast neural style确实效果要差一点。

 [1]: http://arxiv.org/pdf/1603.08155v1.pdf
 [2]: http://hacker.duanshishi.com/?p=1651
 [3]: http://arxiv.org/abs/1508.06576
 [4]: ./images/fast_neural_style_overview.png
 [5]: ./images/fast_neural_network_content_loss.png
 [6]: ./images/fast_neural_network_gram.png
 [7]: ./images/fast_neural_network_gram2.png
 [8]: https://github.com/burness/neural_style_tensorflow
 [9]: https://www.tensorflow.org/
 [10]: http://www.vlfeat.org/matconvnet/models/beta16/imagenet-vgg-verydeep-19.mat
 [11]: http://msvocds.blob.core.windows.net/coco2014/train2014.zip