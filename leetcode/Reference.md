# 1.美团暑期实习
### 1.一个长度为n的字符串，删除尽可能少的字符，使得字符串中不含有长度为偶数的回文子串。
```python

```

### 2.一个排列，元素为红色或者白色，每次操作可以任意互换两个红色元素的位置，尽可能少的操作，使得数组变为非降序的情况。leetcode801的变体
  - 排列：指的是从1到n的每个元素恰好出现一次
```python
def solve(n, nums, colors):
    for i in range(n):
        # 如果当一个数字的下标+1不是他的数据，并且他是白色染色体的时候，一定无法实现这个操作
        if nums[i] != i + 1 and colors[i] == 'W':
            return -1
    # 如何次数最少？每次交互必须保证一个能放到指定位置
    count = 0
    for i in range(n):
        while nums[i] != i + 1:  # 当前这个不在正确的位置，需要交换，一直换到他的指定位置为止
            tmp = nums[nums[i] - 1]  # 因为当前这个nums[i]需要放到nums[i] - 1的位置处
            nums[nums[i] - 1] = nums[i]
            nums[i] = tmp
            count += 1
    return count

```
