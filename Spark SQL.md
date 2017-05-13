##简介
Spark SQL最早合并Shark，一个尝试将Hive跑在Spark上的project。Hive是Hadoop生态圈的一个重要组成部分，内部主要是讲SQL转换为MapReduce Job。
![](http://7xr3b9.com1.z0.glb.clouddn.com/mac_blogs_spark_sql_hive.png)
Shark将其中的MR部分用Spark改写：
![](http://7xr3b9.com1.z0.glb.clouddn.com/mac_blogs_spark_sql.png)
开始时，Shark十分有效，但是渐渐地，开发者发现到瓶颈，无法再进一步优化，最后他们决定从头开始写一个sql引擎，然后就诞生了Spark SQL
![](http://7xr3b9.com1.z0.glb.clouddn.com/mac_blogs_sparksql.png)
Spark SQL的诞生不仅仅服务于运行sql语句，更大的原因是帮助开发人员使用更少的语句更加方便的开发spark应用，Spark SQL使用DataFrame。对比RDD和DataFrame还有DataSet的一篇文章[Apache Spark: RDD, DataFrame or Dataset?](http://www.kdnuggets.com/2016/02/apache-spark-rdd-dataframe-dataset.html)


###Spark HiveContext

	val conf = new SparkConf().setAppName("Spark SQL").setMaster("local[1]")
	val sc = new SparkContext(conf)
	val hc = new HiveContext(sc)
	hc.sql("create table if not exists person(first_name
       string, last_name string, age int) row format delimited fields
       terminated by ','")
	hc.sql("load data local inpath \"/home/hduser/person\" into
       table person")
	# or load the data from hdfs
	hc.sql("load data local inpath \"/home/hduser/person\" into
       table person")
	val persons = hc.sql("from person select first_name,last_
       name,age")
    persons.collect.foreach(println)
    hc.sql("create table person2 as select first_name, last_
           name from person;")
    hc.sql("create table person2 like person location '/user/
           hive/warehouse/person'")
    hc.sql("create table people_by_last_name(last_name
           string,count int)")
    hc.sql("create table people_by_age(age int,count int)")
    hc.sql("""from person
             insert overwrite table people_by_last_name
               select last_name, count(distinct first_name)
               group by last_name
           insert overwrite table people_by_age
               select age, count(distinct first_name)
               group by age; """)

因为这块我使用local，而且我机器上也没有hdfs，这里不好演示，之后到公司机器试下

### Spark SQL using case classes
先创建一个case class:

	case class Person(first_name: String, last_name: String, age: Int)


	val conf = new SparkConf().setAppName("Spark SQL").setMaster("local[1]")
    val sc = new SparkContext(conf)
    val sqlContext = new SQLContext(sc)
    val filePath = getClass.getResource("/person/person.txt").toString
    import sqlContext.implicits._
    val p = sc.textFile(filePath)
    val personDF = p.map(_.split(",")).map(pp=>Person(pp(0),pp(1),pp(2).toInt)).toDF()
    personDF.registerTempTable("person")
    val persons = sqlContext.sql("select * from person order by age desc")
    persons.collect.foreach(println)

![](http://7xr3b9.com1.z0.glb.clouddn.com/mac_blogs_spark_case_class.png)

### Spark SQL specifying the schema

    val conf = new SparkConf().setAppName("Spark SQL").setMaster("local[1]")
    val sc = new SparkContext(conf)
    val sqlContext = new SQLContext(sc)
    val filePath = getClass.getResource("/person/person.txt").toString
    import sqlContext.implicits._
    val p = sc.textFile(filePath)
    val pmap = p.map(_.split(","))
    val personData = pmap.map(p=>Row(p(0),p(1),p(2).toInt))
    val schema = StructType(
        Array(StructField("first_name",StringType,true),
        StructField("last_name",StringType,true),
        StructField("age",IntegerType,true)
        ))
    val personDF = sqlContext.createDataFrame(personData, schema)
    personDF.registerTempTable("person")
    val persons = sqlContext.sql("select * from person")
    persons.collect.foreach(println)

![](http://7xr3b9.com1.z0.glb.clouddn.com/mac_blogs_spark_schema.png)
StructType定义数据表的结构，等同于关系型数据表的结构。StructType由一组StructField对象组成

	StructType(fields: Array[StructField])
	StructField(name: String, dataType: DataType, nullable: Boolean = true, metadata: Metadata = Metadata.empty)


### Loading and saving data using the Parquet format
Apache Parquet是面向分析性业务的列式存储，由Twitter和Cloudera合作开发。
Parquet的优点可以参考这篇文章：[深入分析Parquet列式存储格式](http://www.infoq.com/cn/articles/in-depth-analysis-of-parquet-column-storage-format)

    val personRDD = sc.textFile(filePath).map(_.split(",")).map(p=>Person(p(0),p(1),p(2).toInt))
    val person = personRDD.toDF()
    person.registerTempTable("person")
    val sixtyPlus = sqlContext.sql("select * from person where age > 60")
    sixtyPlus.collect().foreach(println)
    sixtyPlus.saveAsParquetFile("/Users/burness/git_repository/simpleScripts/src/main/resources/sp.parquet")
    val parquetDF = sqlContext.load("/Users/burness/git_repository/simpleScripts/src/main/resources/sp.parquet")
    parquetDF.registerTempTable("sixtyPlus")
    val parquetSixtyPlus = sqlContext.sql("select * from sixtyPlus")
    parquetSixtyPlus.collect().foreach(println)

保存的路径下文件：
![](http://7xr3b9.com1.z0.glb.clouddn.com/mac_blogs_Parquet_format.png)

### Loading and saving data using the json format

    val person = sqlContext.read.json(filePath)
    person.registerTempTable("person")
    val sixtyPlus = sqlContext.sql("select * from person where age > 60")
    sixtyPlus.collect.foreach(println)
    val savePath2 = getClass.getResource("/person/").toString.split(":")(1)+"person.Sp"
    val savePath = "/Users/burness/git_repository/simpleScripts/src/main/resources/person/personSp"
    println(savePath2)
    import sqlContext.implicits._
    println(savePath)
    sixtyPlus.toJSON.saveAsTextFile(savePath)
    sixtyPlus.toJSON.saveAsTextFile(savePath2)
    sc.stop()

![](http://7xr3b9.com1.z0.glb.clouddn.com/mac_blogs_spark_json_result.png)
### Loading and saving data from relational databases

    val url="jdbc:mysql://192.168.35.235:3306/mysql"
    val prop = new java.util.Properties
    prop.setProperty("user","root")
    prop.setProperty("password","qweasdzxc")
    val people = sqlContext.read.jdbc(url,"person",prop)
    people.show
    al males = sqlContext.read.jdbc(url,"person",Array("gender ='M'"),prop)
    males.show
    val first_names = people.select("first_name")
    first_names.show
    val below60 = people.filter(people("age") < 60)
    below60.show
    //    val grouped = people.groupBy("gender")
    //    val gender_count = grouped.count
    people.write.json(getClass.getResource("/person/personDB/").toString.split(":")(1)+"person.json")
    people.write.parquet(getClass.getResource("/person/personDB/").toString.split(":")(1)+"person.parquet")
    // 输出如下：
    +---------+----------+---------+------+---+
    |person_id|first_name|last_name|gender|age|
    +---------+----------+---------+------+---+
    |        1|       asd|      qwe|     m|123|
    |        3|    Barack|    Obama|     M| 53|
    |        4| Barack123|    Obama|     M| 53|
    |        5|Barack1234|    Obama|     M| 53|
    |        6|Barack1234|    Obama|     M| 53|
    |        7| Barack124|    Obama|     M| 53|
    |        8| Barack134|    Obama|     M| 53|
    +---------+----------+---------+------+---+
    +----------+
    |first_name|
    +----------+
    |       asd|
    |     12asd|
    |    Barack|
    | Barack123|
    |Barack1234|
    |Barack1234|
    | Barack124|
    | Barack134|
    +----------+
    +---------+----------+---------+------+---+
    |person_id|first_name|last_name|gender|age|
    +---------+----------+---------+------+---+
    |        3|    Barack|    Obama|     M| 53|
    |        4| Barack123|    Obama|     M| 53|
    |        5|Barack1234|    Obama|     M| 53|
    |        6|Barack1234|    Obama|     M| 53|
    |        7| Barack124|    Obama|     M| 53|
    |        8| Barack134|    Obama|     M| 53|
    +---------+----------+---------+------+---+
![](http://7xr3b9.com1.z0.glb.clouddn.com/mac_blogs_spark_mysql_result.png)

### Loading and saving data from an arbitrary
spark能够很有效地读入文件，包括hdfs、local json、local parquet等等，常用的api也就那么几个，因为我是在自己的小本上用local跑的hdfs的无法演示

更多的可以来看这里[Spark sql-programming-guide](http://spark.apache.org/docs/latest/sql-programming-guide.html)

其中有个Datasets，以后再做总结

