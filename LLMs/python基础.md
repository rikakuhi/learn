# 1.函数的参数
- 位置参数
- 关键字参数： key：value的形式
- *args：可变参数，主要用为未匹配上的位置参数和关键字参数，放到一个元组中。
- **kwargs：可变关键字参数，主要接受原本没有的key=value的参数，放到一个字典中。
# 2.变量的作用域
    只有class、函数等才会有新的域，其中定义的变量都会重新分配内存，并且只能在当前作用域内进行调用。
    在一个函数内，嵌套另外一个函数，内层函数可以调用外层函数的变量，若内嵌函数中有和外层函数一样的变量名，那么优先内层函数。
- 变量查找的过程就是，从内到外一直找的过程，直到找到对应的变量，否则就会报错。
# 3. 闭包函数
- 满足如下这三个条件：
    - 存在函数嵌套
    - 内层函数调用了外层函数中的变量。
    - 返回的内层函数的变量名。
- 作用：当返回内层函数的时候，会保存其用到的变量的一起打包返回，这样相当于保存了外层函数中的一个变量。
# 4.python中的作用域
- L （Locals）局部作用域，或作当前作用域。
- E （Enclosing）闭包函数外的函数中
- G （Globals）全局作用域
- B （Built-ins）内建作用域

查找方式是从上到下。
- global：声明当前变量从全局作用域中找。
- nonlocal：只能用在内嵌函数中，指明当前变量从外层函数的作用域中去寻找。
# 5.装饰器
## 1.装饰器函数
- 不含参数的函数装饰器
    ```python
    def log(func):
        def wrapper(*args, **kwargs):
            ret = func(*args, **kwargs)
            logging.debug('%s is called' % func.__name__)
            return ret
        return wrapper

    func = log(func)
    func(0)
    ```
- 包含参数的函数装饰器
    ```python
    def log(level='debug'):
        def decorator(func):
            def wrapper(*args, **kwargs):
                ret = func(*args, **kwargs)
                if level == 'warning':
                    logging.warning("{} is called".format(func.__name__))
                else:
                    logging.debug("{} is called".format(func.__name__))
                return ret
            return wrapper
        return decorator

    @log(level="warning") # 添加带参数的装饰器 log()
    def func(n):
        print("from func(), n is %d!" % (n), flush=True)

    func(0)
    ```
    - 在装饰器外面在加上一层函数，用于传递参数。
# 类方法装饰器使用方法和这个一样，只不过需要在内层函数添加上一个 self 参数。
## 2.装饰器类
pass
# 3.生成器
    若一个列表中的元素可以以某种方法推算出来，就不用先创建整个列表，而是用到了哪个元素，在创建对应的元素。--->可以降低内存的开销。
在使用列表生成式的时候，换成小括号，就得到是一个生成器。
```python
list_generator0 = (x * x for x in range(5))
print(list_generator0)  # 打印出来的是一个生成器对象。
# 要打印出列表中的元素，可以用python的内置函数 next()
print(next(list_generator0))

# 另外生成器也是一个可以迭代对象，能够通过for循环来使用。

# 生成器函数
def fib_generator(n):
    i, j = 0, 1

    while(i < n):
        yield i  # 每次会停在这里，next()会从当前位置开启下一次迭代。
        i, j = j, i + j

# 生成器函数本身并不是一个生成器，他的返回对象才是一个生成器。
print(type(fib_generator))
print(type(fib_generator(5)))

```
- next()的本质其实是调用对象的__next__方法
# 4.迭代器
- 可迭代对象，内部一定定义了__iter__方法.
- 迭代器一定定义了__iter__和__next__方法。
- 可迭代对象：可以用for循环进行遍历
- 迭代器： 可以用for循环和 next进行迭代，迭代完了之后，就不能再次访问了，相当于销毁了这个迭代器。
- 可以通过iter()函数来将list 等迭代对象变为一个迭代器(接受iter的返回值，是一个迭代器)

# 5.python中的多线程和多进程
- 多线程：
    - 其实是假的多线程，因为python有一个全局锁，目的是保证多个线程中的数据统一。
    - 在I/O密集任务中，还是能够起很大的作用，因为当读取、写入文件的时候，全局锁会释放，然后其他的线程就有机会执行。
    - 同一进程下的多个线程可以互相访问对方的资源。
- 多进程：
    - 不同进程之间不能互相访问对方的资源
    - 不同进程之间的通信是以管道的方式进行通信，是阻塞的。
    