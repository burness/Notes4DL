## Spark Streaming简介
何为streaming？在实际中，有一类数据是连续性生产，流式方法主要就是为了解决这块的问题，将连续性数据切分离散型单元来进行处理的方法。生活中比较熟悉的是视频或者音频的流式服务就是将连续性的数据分为一个个块来传输到用户计算机。
下图为Spark Streaming的示意图：
![](http://7xr3b9.com1.z0.glb.clouddn.com/mac_blogs_streaming_workflow.png)
连续性的RDD在Spark Streaming中的数据表示为DStream，称为Discretized Stream。每一个DStream数据都相应地有一个receiver。receiver从数据源接入数据，然后将其存储到Spark集群的内存中，常见的数据源包括以下两类：
 - 基本数据源包括文件、socket链接
 - 高级数据源包括Kafka、Flume

在Streaming应用中，batch interval是一个十分重要的优化因素，理论上越快的处理数据，batch interval越小，还有window length和sliding interval，window length表明窗口持续时间，sliding interval，假定window length为60s，sliding interval为10s也就是说1min中的有50s的数据是上一个窗口，然后10s的数据为新数据
### word count using Streaming
我们从socket接受数据，然后做一个简单的wordcount

	val conf = new SparkConf().setAppName("Spark Streaming").setMaster("local[2]")
    val sc = new SparkContext(conf)
    Logger.getRootLogger.setLevel(Level.WARN)
    val ssc = new StreamingContext(sc, Seconds(20))// 20 seconds as a batch interval
    val lines = ssc.socketTextStream("localhost",8585,MEMORY_ONLY)
    val wordsFlatMap = lines.flatMap(_.split(" "))
    val wordsMap = wordsFlatMap.map( w => (w,1))
    val wordCount = wordsMap.reduceByKey( (a,b) => (a+b))
    wordCount.print
    ssc.start()
    ssc.awaitTermination()

开启shell，输入nc -lk 8585,然后输入数据：
![](http://7xr3b9.com1.z0.glb.clouddn.com/mac_blogs_Spark_Streaming_Socket_result1.png)
![](http://7xr3b9.com1.z0.glb.clouddn.com/mac_blogs_Spark_Streaming_Socket_result2.png)
### Streaming From Web service: Twitter
开始你必须要在Twitter注册一个app，然后才有相应地使用Twitter api的权限，拿到权限后，我们这里通过Streaming来拿一些tweets（语言为en）：
![](http://7xr3b9.com1.z0.glb.clouddn.com/mac_blogs_spark_streaming_twitter.png)
然后在Keys And Access Tokens中拿到以下四个字段的值：
Consumer Key、Consumer Secret、Access Token、Access Token Secret。
接着在你的maven项目中加入以下依赖：

	<dependency>
        <groupId>org.twitter4j</groupId>
        <artifactId>twitter4j-core</artifactId>
        <version>4.0.4</version>
    </dependency>
    <dependency>
        <groupId>org.twitter4j</groupId>
        <artifactId>twitter4j-stream</artifactId>
        <version>4.0.4</version>
    </dependency>
    <dependency>
        <groupId>org.apache.spark</groupId>
        <artifactId>spark-streaming-twitter_2.11</artifactId>
        <version>1.6.1</version>
    </dependency>
代码如下：

	val ssc = new StreamingContext(sc, Seconds(10))
    val cb = new ConfigurationBuilder
    cb.setDebugEnabled(true)
    .setOAuthConsumerKey("XXXXXXXXXXXXXXXXXXXXXXX")
    .setOAuthConsumerSecret("XXXXXXXXXXXXXXXXXXXXXXX")
    .setOAuthAccessToken("XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX")
    .setOAuthAccessTokenSecret("XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX")
    val auth = new OAuthAuthorization(cb.build())
    val tweets = TwitterUtils.createStream(ssc, Some(auth))
    val englishTweets = tweets.filter(_.getLang=="en")
    val status = englishTweets.map(status=>status.getText)
    status.print
    ssc.checkpoint(getClass.getResource("/checkpoint").toString.split(":")(1))
    ssc.start
    ssc.awaitTermination
    
![](http://7xr3b9.com1.z0.glb.clouddn.com/mac_blogs_spark_streaming_twitter1.png)
有没有发现这里我们设置了一个checkpoint的目录，有什么用呢？因为Spark Streaming一直是连续性数据，当Spark engine出现问题时，如果没有cp，很难知道从哪里恢复，而常规的spark应用只需要replay，但是sreaming不行，前面因为是wordcount示例而已所以没有设置，checkpoint的具体应用可以见[Spark Streaming的还原药水——Checkpoint](http://www.jianshu.com/p/00b591c5f623),这篇文章中讲的十分清楚。

另外一个比较细的从twitter拿tweets的代码如下：

	val filters = Array("ps3", "ps4", "playstation", "sony", "vita", "psvita")
    //val filers = "ThisIsSparkStreamingFilter_100K_per_Second"

    val delimeter = "|"

    System.setProperty("twitter4j.oauth.consumerKey", "e7KiYQz1koMZOuxNtyxu9pjyK")
    System.setProperty("twitter4j.oauth.consumerSecret", "6bHUHyQwPdxQlIOiKSVyFHNAEI2cel6qibaat3wQk2RV0ls0FO")
    System.setProperty("twitter4j.oauth.accessToken", "969272604-iw8OzM90fFCDHoHGQrBQuMXXd1q2wISXtZKj5THz")
    System.setProperty("twitter4j.oauth.accessTokenSecret", "mBkDJe2aq1HWBX1LXRK6Vs0Mz8HvgrOGhccbFItUgUISq")
    System.setProperty("twitter4j.http.useSSL", "true")

    val conf = new SparkConf().setAppName("TwitterApp").setMaster("local[4]")

    val sc = new SparkContext(conf)
    val ssc = new StreamingContext(sc, Seconds(5))

    val tweetStream = TwitterUtils.createStream(ssc, None, filters)

    val tweetRecords = tweetStream.map(status => {

      def getValStr(x: Any): String = {
        if (x != null && !x.toString.isEmpty()) x.toString + "|" else "|"
      }


      var tweetRecord =
        getValStr(status.getUser().getId()) +
          getValStr(status.getUser().getScreenName()) +
          getValStr(status.getUser().getFriendsCount()) +
          getValStr(status.getUser().getFavouritesCount()) +
          getValStr(status.getUser().getFollowersCount()) +
          getValStr(status.getUser().getLang()) +
          getValStr(status.getUser().getLocation()) +
          getValStr(status.getUser().getName()) +
          getValStr(status.getId()) +
          getValStr(status.getCreatedAt()) +
          getValStr(status.getGeoLocation()) +
          getValStr(status.getInReplyToUserId()) +
          getValStr(status.getPlace()) +
          getValStr(status.getRetweetCount()) +
          getValStr(status.getRetweetedStatus()) +
          getValStr(status.getSource()) +
          getValStr(status.getInReplyToScreenName()) +
          getValStr(status.getText())

      tweetRecord

    })

    tweetRecords.print

    //    tweetRecords.filter(t => (t.getLength() > 0)).saveAsTextFiles("/user/hive/warehouse/social.db/tweeter_data/tweets", "data")

    ssc.start()
    ssc.awaitTermination()
    
![](http://7xr3b9.com1.z0.glb.clouddn.com/mac_blogs_spark_streaming_twitter2.png)

### Streaming using Kafka
Kafka是LinkedIn开发的分布式消息系统，很多开源分布式处理系统如Cloudera、Apache Storm、Spark支持与Kafka集成。具体的东西我也不是特别清楚，想更深入了解可以看看这个info的介绍文档[Kafka剖析系列文章](http://www.infoq.com/cn/articles/kafka-analysis-part-1)然后去看相关的项目

首先，在自己机器上安装Kafka以及捆绑的zookeeper，下载地址[Kafka](http://apache.fayea.com/kafka/0.8.2.1/kafka_2.10-0.8.2.1.tgz)

zookeeper启动

	bin/zookeeper-server-start.sh config/zookeeper.properties

kafka启动

	bin/kafka-server-start.sh config/server.properties
	bin/kafka-topics.sh --create --zookeeper localhost:2181 --replication-factor 1 --partitions 1 --topic test

生成消息：

	bin/kafka-console-producer.sh --broker-list localhost:9092 --topic test
    
代码如下：

	val conf = new SparkConf().setAppName("Spark Streaming With Kafka").setMaster("local[2]")
    val sc = new SparkContext(conf)
    Logger.getRootLogger.setLevel(Level.WARN)
    val ssc = new StreamingContext(sc, Seconds(2))
    val zkQuorum = "localhost:2181"
    val group = "test-group"
    val topics = "test"
    val numThreads = 1
    val topicMap = topics.split(",").map((_,numThreads.toInt)).toMap
    val lineMap = KafkaUtils.createStream(ssc, zkQuorum, group, topicMap)
    val lines = lineMap.map(_._2)
    val words = lines.flatMap(_.split(" "))
    val pair = words.map(x=>(x,1))
    val wordCounts = pair.reduceByKeyAndWindow(_+_,_-_,Minutes(10),Seconds(2),2)
    wordCounts.print
    ssc.checkpoint(getClass.getResource("/checkpointKafka").toString.split(":")(1))
    ssc.start
    ssc.awaitTermination

![](http://7xr3b9.com1.z0.glb.clouddn.com/mac_blogs_Spark_streaming_kafka.png)
