    因为scipy.sparse模块缺少关于稀疏矩阵的实现原理，所以查阅了维基百科：https://en.wikipedia.org/wiki/Sparse_matrix
    
    所谓稀疏矩阵，就是矩阵中大多数的元素都为零，反之则为稠密矩阵。这样一来，我们可以思考如何在计算机中高效的存储稀疏矩阵。
    一个矩阵的典型存储方式为一个二维数组，通过索引i，j来访问元素a[i,j]，习惯上i为行索引，j为列索引。那么对于一个m*n矩阵来说，其花费的存储空间为O(mn)
    
    但对于稀疏矩阵来说，因为绝大多数元素都是零，我们可以考虑仅存储非零元素来节省大量的存储空间，有多种算法来实现这个目的，代价会是访问元素更加的复杂，并且需要一些额外的数据结构来正确还原成初始的矩阵。
    
    这些算法可以分为两组：
    
    1. 支持高效的修改操作，比如DoK(Dictionary of Keys),LIL(List of Lists), COO(Coordinate List), 这些都是用来创建矩阵的典型方法
    
    2. 支持高效的访问和矩阵运算操作，例如CSR(Compressed Sparse Row), CSC(Compressed Sparse Column)
    
    Dictionary of Keys(DoK): DoK就是一个字典，key是(row, column)对，value就是对应的元素值，key值里面存储的是所有非零元素的行列索引。显然这个字典里面找不到的元素就都是零元素了，这个算法很适合增量的创建一个稀疏矩阵，但它是无序的，在遍历的时候如果对顺序有要求则不行
    
    List of lists(LIL): LIL用嵌套数组的方式来构建稀疏矩阵，这个也很好理解，每一行的非零元素作为一个内嵌的数组存进来就可以，相比DoK它是有序的，也很适合用于增量的更新矩阵
    
    Coordinate List(COO): COO就是一个list，每个元素就是一个(row, column, value)的tuple，一般会按照row，column来排序，以改善随机访问的速度，也很适合用于增量的更新矩阵
    
    Compressed Sparse Row(CSR, CRS or Yale format): 
    
    它将一个稀疏矩阵M(m*n)存储为三个一维矩阵，分别为：
    
    A：存储所有的非零元素，假设M中非零元素的个数为NN(Number of nonzeros)，则A的长度为NNZ，存储的顺序按从左到右，从上到下遍历
    
    IA：长度为m + 1， 它有个递归的定义：
    
        IA[0] = 0
        IA[i + 1] = IA[i] + (M中第i行的非零元素的数量)
    
    JA: M中每个非零元素的列索引，显然长度也为NNZ
    
    通过下面这个例子可以看到，使用这种结构可以节省存储空间，并且反推回原矩阵时不会具有二义性：
    
    矩阵M：
    
    0  0  0  0
    5  8  0  0
    0  0  3  0
    0  6  0  0
    
    那么根据CSR的算法：
    
    A  = [5, 8, 3, 6]
    IA = [0, 0, 2, 3, 4]
    JA = [0, 1, 2, 1]
    
    这样，通过IA可以计算出每一行所包含的非零元素，再结合JA可得到列索引，然后取A中对应的值，可还原出整个初始矩阵。原矩阵中M需要16个单位的存储空间，CSR只需要13个。CSR节省内存有个条件：NNZ < (m(n-1) - 1)/2
    
    Compressed sparse comlumn(CSC)
    
    与CSR的原理是一致的，只不过A中的元素现在的遍历顺序改为按列遍历，从上到下再从左到右。并且IA中存储的是列的递归，JA中存储的是每个非零元素的行索引，例如上面的矩阵M，其CSC的表示形式为：
    A  = [5, 8, 6, 3]     -> 对应scipy.csc_matrix里面的data
    IA = [0, 1, 3, 4, 0]  -> 对应scipy.csc_matrix里面的indices
    JA = [1, 1, 3, 2]     -> 对应scipy.csc_matrix里面的indptr
    
