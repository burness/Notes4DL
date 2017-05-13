## Eigen
Eigen项目地址: [http://eigen.tuxfamily.org/index.php?title=Main_Page](http://eigen.tuxfamily.org/index.php?title=Main_Page)

```
Eigen is a C++ template library for linear algebra: matrices, vectors, numerical solvers, and related algorithms.
```

## Tutorial

第一个使用Eigen的代码，定义一个arbitrary size的矩阵类型MatrixXd，给定大小m(2,2)，然后给各个元素赋值
    #include <iostream>
    #include <Eigen/Dense>
    using Eigen::MatrixXd;
    
    int main()
    {
    MatrixXd m(2,2);
    m(0,0) = 3;
    m(1,0) = 2.5;
    m(0,1) = -1;
    m(1,1) = m(1,0) + m(0,1);
    std::cout << m << std::endl;
    }

    /******
    *  3   -1
    * 2.5  1.5
    ******/

对比MatrixXd和MatrixXd的代码段

    #include <iostream>                                #include <iostream>
    #include <Eigen/Dense>                             #include <Eigen/Dense>
    using namespace Eigen;                             using namespace Eigen;
    using namespace std;                               using namespace std;
    int main()                                         int main()
    {                                                  {
    MatrixXd m = MatrixXd::Random(3,3);                 Matrix3d m = Matrix3d::Random();
    m = (m + MatrixXd::Constant(3,3,1.2)) * 50;         m = (m + Matrix3d::Constant(1.2)) * 50;
    cout << "m =" << endl << m << endl;                 cout << "m =" << endl << m << endl;
    VectorXd v(3);                                      Vector3d v(1,2,3);
    v << 1, 2, 3;
    cout << "m * v =" << endl << m * v << endl;         cout << "m * v =" << endl << m * v << endl;
    }                                                  }
    /**********                                        /**********
    * m =                                              * m =
    * 10.0008  55.865 14.7045                          * 10.0008  55.865 14.7045
    * 23.1538 63.2767 77.8865                          * 23.1538 63.2767 77.8865
    * 85.5605 31.8959 77.9296                          * 85.5605 31.8959 77.9296
    * m * v =                                          * m * v =
    * 165.844                                          * 165.844
    * 383.367                                          * 383.367
    * 383.141                                          * 383.141
    ***********/                                       ***********/
测试了两部分代码的
`bash test_speed.sh 01-martix-vector 10000  16.76s user 11.86s system 70% cpu 40.471 total`
`bash test_speed.sh 02-martix-vector 10000  16.60s user 11.89s system 70% cpu 40.342 total`

编译生成的可执行文件02-martix-verctor比01-matrix-vector大一些
    ➜  eigen_study ll
    -rw-r--r--  1 burness  staff   203B  4  4 14:13 00-simple_program.cpp
    -rwxr-xr-x  1 burness  staff   142K  4  4 14:34 01-martix-vector
    -rw-r--r--  1 burness  staff   316B  4  4 14:33 01-martix-vector.cpp
    -rwxr-xr-x  1 burness  staff   161K  4  4 14:34 02-martix-vector
    -rw-r--r--  1 burness  staff   286B  4  4 14:33 02-martix-vector.cpp
    -rwxr-xr-x  1 burness  staff    56K  4  4 14:03 simple_program
    -rw-r--r--  1 burness  staff    94B  4  4 14:42 test_speed.sh

### Matrix
在Eigen中，所有的matrix和vector都是Matrix对象，通过不同模板来实现不同类型的Matrix，Vectors只是matrix的一个特点类（一列或者一行）

可以使用很多typedef 来定义特点的类：

    typedef Matrix<float, 4, 4> Matrix4f;

### Vectors

    typedef Matrix<float, 3, 1> Vector3f;
    typedef Matrix<int, 1, 3> RowVector3i;

在编译时，维度不可知的也可以使用Dynamic来定义行或者列的长度：

    typedef Matrix<double, Dynamic, Dynamic> MatrixXd;

### 基本操作
#### Coefficient accessors

    MatrixXd m(2,2)
给定值后，可以通过index来读取对应位置的数， 但是和python的不太一样，m(0)不能读取第0行的，而是第0行的第0个元素

#### Comma-initialization
一种给Matrix初始化的方法：

    Matrix3f m;
    m << 1, 2, 3, 4, 5, 6, 7, 8, 9;

#### resize

    #include <iostream>
    #include <Eigen/Dense>
    using namespace Eigen;
    int main()
    {
        MatrixXd m(2,5);
        m.resize(4,3);
        std::cout << "The matrix m is of size "
                << m.rows() << "x" << m.cols() << std::endl;
        std::cout << "It has " << m.size() << " coefficients" << std::endl;
        VectorXd v(2);
        v.resize(5);
        std::cout << "The vector v is of size " << v.size() << std::endl;
        std::cout << "As a matrix, v is of size "
                << v.rows() << "x" << v.cols() << std::endl;
    }

    The matrix m is of size 4x3
    It has 12 coefficients
    The vector v is of size 5
    As a matrix, v is of size 5x1

#### Assignment and resizing

    MatrixXf a(2,2);
    std::cout << "a is of size " << a.rows() << "x" << a.cols() << std::endl;
    MatrixXf b(3,3);
    a = b;
    std::cout << "a is now of size " << a.rows() << "x" << a.cols() << std::endl;

    a is of size 2x2
    a is now of size 3x3

#### Fixed vs. Dynamic size

    Matrix4f mymatrix;
    MatrixXf mymatrix[16];

dynamic-size matrix分配在heap


#### Optional template parameters

Matrix 有6中模板参数配置：

    Matrix<typename Scalar,
       int RowsAtCompileTime,
       int ColsAtCompileTime,
       int Options = 0,
       int MaxRowsAtCompileTime = RowsAtCompileTime,
       int MaxColsAtCompileTime = ColsAtCompileTime>

MaxRowsAtCompileTime表示在编译时间不确定row长度时，指定最大的row长度

#### Convenience typedefs

    MatrixNt for Matrix<type, N, N>. For example, MatrixXi for Matrix<int, Dynamic, Dynamic>.
    VectorNt for Matrix<type, N, 1>. For example, Vector2f for Matrix<float, 2, 1>.
    RowVectorNt for Matrix<type, 1, N>. For example, RowVector3d for Matrix<double, 1, 3>.

### 矩阵向量计算

#### 加减乘除

#### Transposition and conjugation

对于实数矩阵，conjugate不作任何影响，adjoint()等价于transpose()
注意千万不能有a=a.transpose()，要想完成相应的功能只需要a.transposeInPlace()即可

#### Matrix-matrix and matrix-vector multiplication

矩阵相乘是比较特殊的，不会产生 aliasing issues

#### Dot product and cross product

叉乘还记得吗 ？

点积： (x1 , y1 , z1 ) .( x2 , y2 , z2 ) = x1x2 + y1y2 + z1z2
叉乘： ( x1 , y1 , z1 ) X ( x2 , y2 , z2 ) =( y1z2 - z1y2 , z1x2 - x1z2 , x1y2 - y1x2 )

#### Basic arithmetic reduction operations

    Matrix2d mmat;
    mmat << 1, 2, 3, 4;
    double matsum = mmat.sum();
    double matprod = mmat.prod();
    double matmean = mmat.mean();
    double matminCoeff = mmat.minCoeff();
    double matmaxCoeff = mmat.maxCoeff();
    double matTrace = mmat.trace();

    std::cout << "Here is mat.sum():    " << matsum << std::endl;
    std::cout << "Here is mat.prod():    " << matprod << std::endl;
    std::cout << "Here is mat.mean():    " << matmean << std::endl;
    std::cout << "Here is mat.minCoeff():    " << matminCoeff << std::endl;
    std::cout << "Here is mat.maxCoeff():    " << matmaxCoeff << std::endl;
    std::cout << "Here is mat.trace():    " << matTrace << std::endl;

    Matrix3f mmm = Matrix3f::Random();
    std::ptrdiff_t i, j;

    float minofM = mmm.minCoeff(&i, &j);

    std::cout << "Here is the matrix m: \n" << mmm << std::endl;
    std::cout << "Its minimum coefficient ( " << minofM << ") is at position (" << i << "," << j << ")\n\n";

    RowVector4i vvv = RowVector4i::Random();
    int maxOfVvv = vvv.maxCoeff(&i);
    std::cout << "Here is the vector vvv: " << vvv << std::endl;
    std::cout << "Its maximum coefficient (" << maxOfVvv << ") is at position" << i << std::endl; 

    /****************
    * Here is mat.sum():    10
    * Here is mat.prod():    24
    * Here is mat.mean():    2.5
    * Here is mat.minCoeff():    1
    * Here is mat.maxCoeff():    4
    * Here is mat.trace():    5
    * Here is the matrix m:
    * 0.358593 0.0388328 -0.893077
    * 0.869386  0.661931 0.0594004
    * -0.232996 -0.930856  0.342299
    * Its minimum coefficient ( -0.930856) is at position (2,1)

    * Here is the vector vvv: -1057210095  -250362984  -930199212  -177197521
    * Its maximum coefficient ( -177197521) is at position 3
    ******************/


### Array

Array是在Eigen使用很多的库，提供更方便的方式来做数组操作， 操作和Matrix感觉很类似

#### Accessing values inside an Array
    ArrayXXf m(2, 2);
    m(0, 0) = 1.0;
    m(0, 1) = 2.0;
    m(1, 0) = 3.0;
    m(1, 1) = m(0, 1) + m(1, 0);
    cout << m << endl;
    m << 1.0, 2.0, 3.0, 4.0;
    cout << m << endl ;

#### Addition and subtraction

    ArrayXXf a(3, 3);
    ArrayXXf b(3, 3);
    a << 1, 2, 3, 4, 5, 6, 7, 8, 9;
    b << 1, 2, 3, 1, 2, 3, 1, 2, 3;

    cout << "a + b = " << endl << a + b << endl << endl;
    cout << "a - 2 = " << endl << a - 2 << endl;

#### Array multiplication
两个数组只有当有相同的维度时才能够相乘：

    ArrayXXf aa(2, 2);
    ArrayXXf bb(2, 2);

    aa << 1.0, 2.0, 3.0, 4.0;
    bb << 5.0, 6.0, 7.0, 8.0;

    cout << "a * b = " << endl << aa * bb << endl;

#### Other coefficient-wise operations

    ArrayXf aaa = ArrayXf::Random(5);
    aaa *= 2;
    cout << "aaa =" << endl << aaa << endl;
    cout << "aaa.abs() =" << endl << aaa.abs() << endl;
    cout << "aaa.abs().sqrt() =" << endl << aaa.abs().sqrt() << endl;
    cout << "aaa.min(aaa.abs().sqrt()) =" << endl << aaa.min(aaa.abs().sqrt()) << endl;

min()在给定的两个数组中比较小的元素

#### Converting between array and matrix expressions

当需要使用线代操作时，一般使用Matrix，当元素级操作时，使用Array；当然当你想同时使用Matrix和Array操作时，只需要将Matrix转换为Array，或者Array转换为Matrix即可。

    Matrix==>Array Matrix对象使用.array()即可
    Array==>Matrix Array对象使用.matrix()即可

在Eigen，混合使用Matrix和Array是被禁止的，无法直接将matrix和array对象相加