# 1.美团暑期实习
### 1.一个长度为n的字符串，删除尽可能少的字符，使得字符串中不含有长度为偶数的回文子串。
```python
# 没有长度为偶数的回文子串 等价于 不能有连续的字符相等
def solve(s):
    result = []
    count = 0
    for index, char in enumerate(s):
        if index == 0:
            result.append(char)
        else:
            if char != result[-1]:
                result.append(char)
            else:
                count += 1
    return "".join(result)
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

### 3.点外卖，一共点n天，然后希望每天吃的都不能相同，问最多有多少种点发？
```python
def solve(n, selects):
    """
    n: 点外卖的参数   selects：每天的选择
    """
    # dp[i][j] 表示第i天选择了商家j时的选择方案树。 那么第i-1天就不能选择j。
    # 状态转移方程 dp[i][j] = sum dp[i-1][k] (k!=j)
    if n == 0:
        return 0
    dp_prev_sum = 0  # 用于统计前一天所有商家的方案数之和
    dp_prev = {}  # 存放前一天选择每个商家的方案数
    for shop in selects[0]:  # 首先初始化第一天的选择
        dp_prev[shop] = 1  # 第一天可以任选
        dp_prev_sum += 1
    for i in range(1, n):
        current_shops = selects[i]
        dp_curr = {}
        dp_curr_sum = 0

        for shop in current_shops:
            dp_curr[shop] = (dp_prev_sum - dp_prev.get(shop, 0))  # 当前这天选择shop时候，选择的总数等于，前一天不选shop的总数
            dp_curr_sum = (dp_curr_sum + dp_curr[shop])

        dp_prev = dp_curr
        dp_prev_sum = dp_curr_sum
    return dp_prev_sum
```
### 4.一个长度为n的排列a，将a转换为一个V图(排列先减后增，全增或者全减也属于)每次交换可以交换相邻的两个数字，最少操作几次？