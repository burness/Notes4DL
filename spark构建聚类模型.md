##构建聚类模型
在Machine Learning领域中，我们常会遇到聚类模型这个概念，和分类与回归模型不同，聚类model是属于无监督模型，无须label信息。

聚类模型在实际中有很多应用的case，比如：

 - 对用户或者消费者群体进行用户行为或者元信息的聚类
 - 社区关系的群体发现
 - 商品或者网页的聚类

spark在聚类这块有些相关的工作：

本文会基本介绍两种聚类模型，并且演示如何在spark中快速构造一个聚类模型


### 聚类模型
聚类模型在Machine Learning中是一块很复杂的领域，最常用的就是kmeans了，另外的还有基于密度的模型（DBSCAN、OPTICS），分布模型（GMM），图聚类模型（clique），具体可以去看看wiki:[Cluster_analysis](https://en.wikipedia.org/wiki/Cluster_analysis).
原理我就不说了，看看wiki就够了，接下来直接开始撸码

### 特征提取
提取电影的属性信息：

	genres = sc.textFile("../data/ML_spark/MovieLens/u.genre")
    for line in genres.take(5):
    	print line
    # 输出如下：
    unknown|0
	Action|1
	Adventure|2
	Animation|3
	Children's|4

得到一个电影的属性映射map,对应的genre_id和电影类型名：

	genre_map = genres.filter(lambda x: len(x) > 0).map(lambda line : line.split('|')).map(lambda x:(x[1],x[0])).collectAsMap()
    print genre_map
    # 输出如下：
	{u'0': u'unknown',
     u'1': u'Action',
     u'10': u'Film-Noir',
     u'11': u'Horror',
     u'12': u'Musical',
     u'13': u'Mystery',
     u'14': u'Romance',
     u'15': u'Sci-Fi',
     u'16': u'Thriller',
     u'17': u'War',
     u'18': u'Western',
     u'2': u'Adventure',
     u'3': u'Animation',
     u'4': u"Children's",
     u'5': u'Comedy',
     u'6': u'Crime',
     u'7': u'Documentary',
     u'8': u'Drama',
     u'9': u'Fantasy'}
     
提取电影的title和genres：

	movies = sc.textFile('../data/ML_spark/MovieLens/u.item')
	print movies.first()
    def func1(array):
        genres = array[5:]
        genres_assigned = zip(genres, range(len(genres)))
        index_1=[]
        for x,y in genres_assigned:
            if x=='1':
                index_1.append(y)
        index_1_val = [genre_map[str(i)] for i in index_1]
        index_1_val_str = ','.join(index_1_val)
        return (int(array[0]),array[1]+','+index_1_val_str)
    #     return array[0]+','+array[1]+','+
    titles_and_genres = movies.map(lambda x: x.split('|')).map(lambda x:func1(x))
    titles_and_genres.first()
    # (1, u"Toy Story (1995),Animation,Children's,Comedy")
    
    
    
训练recommendation model，具体可以看看我前面的文章[Machine Learning With Spark Note 2：构建一个简单的推荐系统](http://hacker.duanshishi.com/?p=1345)这里我是为了得到电影的隐变量矩阵，这个矩阵可以认为是电影的向量表示：

	from pyspark.mllib.recommendation import ALS
    from pyspark.mllib.recommendation import Rating
    raw_data = sc.textFile("../data/ML_spark/MovieLens/u.data")
    raw_ratings = raw_data.map(lambda x:x.split('\t')[:3])
    ratings = raw_ratings.map(lambda x: Rating(x[0], x[1], x[2]))
    ratings.cache()
    als_model = ALS.train(ratings,50,5,0.1)
    from pyspark.mllib.linalg import Vectors
    movie_factors = als_model.productFeatures().map(lambda (id,factor): (id,Vectors.dense(factor)))
	print movie_factors.first()
	movie_vectors = movie_factors.map(lambda (id,vec):vec)
	user_factors = als_model.userFeatures().map(lambda (id,factor):(id,Vectors.dense(factor)))
	user_vectors = user_factors.map(lambda (id, vec):vec)
	print user_vectors.first()
    # 输出如下：
    (1, DenseVector([0.1189, 0.154, -0.1281, 0.0743, 0.3372, -0.0218, -0.1564, -0.0752, -0.3558, -0.129, -0.2035, 0.425, 0.2254, 0.0389, -0.16, 0.1132, -0.0508, -0.2512, 0.3065, -0.3016, 0.2264, -0.1025, 0.594, 0.4342, 0.0976, -0.2594, 0.4988, -0.1878, -0.543, -0.2482, -0.2286, -0.2257, -0.3169, 0.5306, -0.2114, 0.1968, 0.1103, -0.1596, 0.446, 0.13, -0.2431, -0.1562, -0.2451, 0.2605, -0.5239, -0.1533, -0.078, -0.18, 0.0902, -0.2976]))
    [0.287010610104,-0.306130200624,-0.0110167916864,-0.100282646716,0.402284443378,0.133642598987,-0.17621190846,0.188554614782,-0.327551275492,-0.263691723347,-0.457682311535,0.524626433849,0.15720166266,-0.0829833373427,-0.295744478703,0.105343133211,0.277225226164,-0.273413777351,0.335160762072,-0.185756832361,0.445180237293,-0.600775659084,0.723579525948,-0.00662225764245,0.0986614897847,-0.320296704769,0.743609786034,-0.180224940181,-0.503776729107,-0.422970384359,-0.56777215004,-0.231761977077,0.00380780920386,1.10723686218,-0.27037063241,-0.0452572144568,0.418190091848,-0.00451346393675,0.329894691706,-0.272329092026,-0.151863947511,0.103571020067,-0.465166419744,0.201156660914,-0.603282809258,-0.0489130392671,0.0569526553154,-0.0179597213864,0.0932706743479,0.100327283144]
    
训练聚类模型

	from pyspark.mllib.clustering import KMeans
    num_clusters = 5
	num_iterations = 20
	num_runs =3
    movie_cluster_model = KMeans.train(movie_vectors,num_clusters, num_iterations, num_runs)
    movie_cluster_model_coverged = KMeans.train(movie_vectors,num_clusters,100)
    # user cluster model
    user_cluster_model = KMeans.train(user_vectors,num_clusters,num_iterations, num_runs)
    predictions = movie_cluster_model.predict(movie_vectors)
    print ",".join([str(i) for i in predictions.take(10)])
    # 输出如下：
    4,0,0,3,0,4,3,4,4,3
    
    
### 模型评估与调优
一般对聚类模型，会计算WCSS(within-cluster sum of squares)来表明聚类模型的好坏，spark kmeans模型里面有computeCost函数来计算WCSS：

	movie_cost = movie_cluster_model.computeCost(movie_vectors)
	user_cost = user_cluster_model.computeCost(user_vectors)
	print "WCSS for movies: %f"%movie_cost
	print "WCSS for users: %f"%user_cost
    # 输出如下：
    WCSS for movies: 2172.650469
	WCSS for users: 1458.771774
    
选定模型评估标准后，我们就可以做参数调优了：

	train_test_split_movies = movie_vectors.randomSplit([0.6,0.4],123)
    train_movies = train_test_split_movies[0]
    test_movies = train_test_split_movies[1]
    for k in [2,3,4,5,10,20]:
        k_model = KMeans.train(train_movies,num_iterations,k,num_runs)
        cost = k_model.computeCost(test_movies)
        print 'WCSS for k=%d : %f'%(k,cost)
    # costs_moives = [2,3,4,5,10,20].map(lambda k:(k, KMeans.train(train_movies,num_iterations,k,num_runs)).compute_cost(test_movies))
    # 输出如下：
    WCSS for k=2 : 790.686228
    WCSS for k=3 : 785.881720
    WCSS for k=4 : 784.198163
    WCSS for k=5 : 788.684923
    WCSS for k=10 : 771.914133
    WCSS for k=20 : 778.678835
    
同样，用户聚类：

	train_test_split_movies = user_vectors.randomSplit([0.6,0.4],123)
    train_users = train_test_split_movies[0]
    test_users = train_test_split_movies[1]
    for k in [2,3,4,5,10,20]:
        k_model = KMeans.train(train_users,num_iterations,k,num_runs)
        cost = k_model.computeCost(test_users)
        print 'WCSS for k=%d : %f'%(k,cost)
        
    # 输出如下：
    WCSS for k=2 : 547.122121
    WCSS for k=3 : 551.845096
    WCSS for k=4 : 551.888517
    WCSS for k=5 : 555.971549
    WCSS for k=10 : 546.884437
    WCSS for k=20 : 539.705653
    
可能，有些同学看到这里，觉得聚类了，能干啥呢？在推荐这块很多时候，比如上文中做mf推荐后，可以做隐向量的聚类，然后我们在计算相似度的时候，只需要在对应类里面进行计算，另外有了隐向量后，我们可以做用户的群体划分、电影的划分等等相关的工作，很多时候会有很有意思的东西，作者在项目当中，曾遇到对1400w+的sku来做相似品的推荐，取计算一个1400w+*1400w+的相似矩阵，即使在spark上也是特别难得，何况，大部分数据都是不需要的，我们的方法是先保存kmeans之后的数据点和对应的数据簇，这样，在计算某个主品的相似品时，只需要计算主品与该簇下的商品之间的相似性，大大地减少计算量。很多时候，某一类模型可能并不去直接解决一个问题，而且简化其他方法，使其变得更加简单。


