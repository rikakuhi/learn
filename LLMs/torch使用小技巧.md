# 1.Tensor.reshape() 和 Tensor.view()之间的区别
- view返回一个新向量，与原向量共享内存，且需要向量在内存上是连续的才可以。修改原向量或者新向量的时候，另外一个向量也会随之修改。
- reshape返回一个新的向量，并不要求原始向量内存是连续的。
