#### LC. 154 寻找旋转排序数组中的最小值 II（略难）

给你一个可能存在 **重复** 元素值的数组 `nums` ，它原来是一个升序排列的数组，并按上述情形进行了多次旋转。请你找出并返回数组中的 **最小元素** 。

<font size=4 color=blue>题解：</font>

​	旋转排序数组 nums 可以被拆分为 2 个排序数组 nums1  , nums2；且 `nums1任一元素 >= nums2任一元素`；因此，考虑二分法寻找此两数组的分界点 nums[i]  (即第 2 个数组的首个元素)。

- 设置 left，right 指针在 nums 数组两端，mid 为每次二分的中点：
  - 当 nums[mid]  > nums[right]  时，mid 一定在第 1 个排序数组中，i 一定满足 mid < i <= right，因此执行 left = mid + 1；
  - 当 nums[mid] <  nums[right]  时，mid  一定在第 2 个排序数组中，i 一定满足 left < i <= mid，因此执行 right = mid；

  - 当 `nums[mid] == nums[right]` 时，是此题的难点（原因是此题中数组的元素**可重复**，难以判断分界点 i 指针区间）；

```
class Solution {
public:
    int findMin(vector<int>& nums) {
        int n=nums.size();
        int l=0,r=n-1;
        while(l<r){
            int mid=l+(r-l)/2;
            if(nums[mid]==nums[r]) r--;
            else if(nums[mid]>nums[r]){
                l=mid+1;
            }
            else{
                r=mid;
            }
        }
        return nums[l];
    }
};
```



#### LC 264. 丑数 II

给你一个整数 `n` ，请你找出并返回第 `n` 个 **丑数** 。**丑数** 就是只包含质因数 `2`、`3` 和/或 `5` 的正整数。

<font size=4 color=blue>题解：</font>

​	定义数组 dp，其中  dp[i]  表示第 i  个丑数，第 n 个丑数即为 dp[n]。

​	由于最小的丑数是 1，因此 dp[1]=1。如何得到其余的丑数呢？定义三个指针 p2，p3，p5 ，表示下一个丑数是当前指针指向的丑数乘以对应的质因数。初始时，三个指针的值都是 1。

```
class Solution {
public:
    int nthUglyNumber(int n) {
        vector<int> dp(n + 1);
        dp[1] = 1;
        int p2 = 1, p3 = 1, p5 = 1;
        for (int i = 2; i <= n; i++) {
            int num2 = dp[p2] * 2, num3 = dp[p3] * 3, num5 = dp[p5] * 5;
            dp[i] = min(min(num2, num3), num5);
            if (dp[i] == num2) {
                p2++;
            }
            if (dp[i] == num3) {
                p3++;
            }
            if (dp[i] == num5) {
                p5++;
            }
        }
        return dp[n];
    }
};
```

#### LC 560 和为K的子数组

给定一个整数数组和一个整数 **k，**你需要找到该数组中和为 **k** 的连续的子数组的个数。

<font size=4 color=blue>题解：</font>

前缀和+哈希

```
class Solution {
public:
    int subarraySum(vector<int>& nums, int k) {
        unordered_map<int,int> map;   // map[i][j] 表示 前缀和i 出现了 j 次
        map[0]=1;
        int presum=0,cnt=0;
        for(auto& num:nums){
            presum+=num;
            if(map.find(presum-k)!=map.end()) cnt+=map[presum-k];
            map[presum]++;
        }
        return cnt;
    }
};
```

#### LC 220 存在重复元素 III

​	在整数数组 nums 中，是否存在两个下标 i 和 j，使得 nums [i] 和 nums [j] 的差的绝对值小于等于 t ，且满足 i 和 j 的差的绝对值也小于等于 ķ 。

<font size=4 color=blue>题解：</font> 桶排序，每个桶大小为 t，只要桶内有元素，就可以返回 true。否则看 相邻桶是否有元素，且 下标间距<=k

```
class Solution {
public:
    bool containsNearbyAlmostDuplicate(vector<int>& nums, int k, int t) {
        if(t<0) return false;
        long bucket_size=t+1L;
        
        unordered_map<long ,long > bucket;
        for(int i=0;i<nums.size();i++)
        {
            int index= nums[i]/ bucket_size;
            //可能nums[i]为负数，比如-4 / 5 以及 -4 / 5都等于0，所以负数要向下移动一位
            if(nums[i] < 0) index--;
         
            if(bucket.find(index)!=bucket.end()) return true;
      
            else if(bucket.find(index-1)!=bucket.end() && abs(nums[i] - buck[index-1]) <= t)
                return true;
            else if(bucket.find(index+1)!=bucket.end() && abs(nums[i] - bucket[index+1]) <= t)
                return true;
                
            bucket[index] = nums[i];
            if(i >= k)   // 桶相当于  滑动窗口，当 i>=k，删除最左边的桶，维持相邻桶元素间隔 k
            {
                bucket.erase(nums[i - k] / bucket_size);
            }
        }
        return false;
    }
};
```

#### LC 26. 删除有序数组中的重复项

​	给你一个有序数组 nums ，请你 原地 删除重复出现的元素，使每个元素 只出现一次 ，返回删除后数组的新长度。不要使用额外的数组空间，你必须在 原地 修改输入数组 并在使用 O(1) 额外空间的条件下完成。

<font size=4 color=blue>题解：</font>首先数组是有序的，那么重复的元素一定会相邻。要求删除重复元素，实际上就是将不重复的元素移到数组的左侧。

双指针，指针p 和q。比较 p 和 q 位置的元素是否相等。如果相等，q 后移 1 位；如果不相等，将 q 位置的元素复制到 p+1 位置上，p 后移一位，q 后移 1 位；重复上述过程，直到 q 等于数组长度。

返回 p + 1，即为新数组长度。

```
class Solution {
public:
    int removeDuplicates(vector<int>& nums) {
        int n=nums.size();
        if(!n) return 0;
        int left=0,right=1;
        while(right<n){
            if(nums[left]!=nums[right]){
                nums[++left]=nums[right];
            }
            right++;
        }
        return left+1;
    }
};
```



#### 剑指Offer 47 礼物的最大价值



## 回溯算法

【 LC优质解答 】 https://leetcode-cn.com/problems/subsets/solution/c-zong-jie-liao-hui-su-wen-ti-lei-xing-dai-ni-gao-/

回溯算法与 DFS 的区别就是   有无状态重置；

**何时使用回溯算法**
		当问题需要 "回头"，以此来查找出所有的解的时候，使用回溯算法。**即满足结束条件或者发现不是正确路径的时候(走不通)，要撤销选择，回退到上一个状态**，继续尝试，直到找出所有解为止；



### （一）组合| 子集（与顺序无关）

#### LC 78. 子集

给你一个整数数组 `nums` ，数组中的元素 **互不相同** 。返回该数组所有可能的子集（幂集）。

<font size=4 color=blue>题解：</font> 

<img src="https://pic.leetcode-cn.com/d8e07f0c876d9175df9f679fcb92505d20a81f09b1cb559afc59a20044cc3e8c-%E5%AD%90%E9%9B%86%E9%97%AE%E9%A2%98%E9%80%92%E5%BD%92%E6%A0%91.png" alt="子集问题递归树.png" style="zoom:50%;" />

```
class Solution {
private:
    vector<vector<int>> ans;
    vector<int> temp;
public:
    void dfs(int start, vector<int>& nums){
        ans.push_back(temp);
        for(int i = start; i < nums.size(); i++){    // start,
            temp.push_back(nums[i]);
            dfs(i + 1, nums);
            temp.pop_back();
        }
    }

    vector<vector<int>> subsets(vector<int>& nums) {
        dfs(0, nums);
        return ans;
    }
};
```

#### LC 90. 子集 II

给你一个整数数组 `nums` ，其中可能**包含重复元素**，请你返回该数组所有可能的子集（幂集）。

```
class Solution {
private:
    vector<vector<int>> res;
    vector<int> path;
public:
    void backtrace(vector<int>& nums,int start){
        res.push_back(path);
        for(int i=start;i<nums.size();i++){    // start，状态变量
            if(i>start && nums[i]==nums[i-1])   // 去重，剪枝
                continue;
            path.push_back(nums[i]);
            backtrace(nums,i+1);
            path.pop_back();
        }
    }
    vector<vector<int>> subsetsWithDup(vector<int>& nums) {
        res.clear();
        path.clear();
        sort(nums.begin(),nums.end());  // 对于重复元素，排序后更方便操作
        backtrace(nums,0);
        return res;
    }
};
```

#### LC 39.  组合总和

​		给定一个**无重复元素的数组 candidates 和一个目标数 target** ，找出 candidates 中所有可以使数字和为 target 的组合。candidates 中的**数字可以无限制重复被选取**。

​      示例：

```
输入：candidates = [2,3,6,7], target = 7,
所求解集为：
[
  [7],
  [2,2,3]
]
```

解答：

```
class Solution {
    vector<int> path;
    vector<vector<int>> res;
public:
    void backtrace(vector<int>& candidates,int start,int target){
        if(target==0){  // 终止搜索| 回溯条件
            res.push_back(path);
            return ;
        }
        for(int i=start;i<candidates.size()&& target-candidates[i]>=0;i++){  // target-candidates[i]>=0 剪枝条件
            path.push_back(candidates[i]);
            backtrace(candidates,i,target-candidates[i]);
            path.pop_back();
        }
    }
    vector<vector<int>> combinationSum(vector<int>& candidates, int target) {
        sort(candidates.begin(),candidates.end());  // 排序，方便剪枝
        backtrace(candidates,0,target);
        return res;
    }
};
```



### （二）排列

【 LC优质解答 】 https://leetcode-cn.com/problems/zi-fu-chuan-de-pai-lie-lcof/solution/c-zong-jie-liao-hui-su-wen-ti-lei-xing-dai-ni-ga-4/

#### LC 46. 全排列

给定一个 **没有重复 数字**的序列，返回其所有可能的全排列。

```
示例:
输入: [1,2,3]
输出:
[
  [1,2,3],
  [1,3,2],
  [2,1,3],
  [2,3,1],
  [3,1,2],
  [3,2,1]
]
```

<font size=4 color=blue>题解：</font> 

<img src="https://pic.leetcode-cn.com/60930c71aa60549ff5c78700a4fc211a7f4304d6548352b7738173eab8d6d7d8.png" alt="在这里插入图片描述" style="zoom:40%;" />

```
class Solution {
private:
    vector<int> path;
    vector<vector<int>> res;
public:
    vector<vector<int>> permute(vector<int>& nums) {
        vector<bool> used(nums.size(),false);   // 使用 used[]
        backtrace(nums,used);
        return res;
    }

    void backtrace(vector<int>& nums,vector<bool> used){
        if(path.size()==nums.size()){
            res.push_back(path);
            return ;
        }
        for(int i=0;i<nums.size();i++){    //
            if(used[i]==true) continue;
            used[i]=true;
            path.push_back(nums[i]);
            backtrace(nums,used);
            path.pop_back();
            used[i]=false;
        }
    }
};
```

<font size=4 color=red>**总结：**</font>

<font size=4>可以发现“排列”类型问题和“子集、组合”问题不同在于：**“排列”问题使用used数组来标识选择列表**，而**“子集、组合”问题则使用start参数**。</font>

#### LC  47.  全排列 II

​		给定一个可**包含重复数字**的序列 `nums` ，按任意顺序 返回所有不重复的全排列。

<font size=4 color=blue>题解：</font> 

参考 LC 90，**对于需要去重的，将 nums 数组排序，让重复的元素并列排在一起。**

```
void backtrack(vector<int>& nums,vector<bool>&used,vector<int>& path)//used初始化全为false
{
    if(path.size()==nums.size())
    {
        res.push_back(path);
        return;
    }
    for(int i=0;i<nums.size();i++)          //从给定的数中除去，用过的数，就是当前的选择列表
    {
        if(!used[i])
        {
            if(i>0&&nums[i]==nums[i-1]&&!used[i-1])    //剪枝，三个条件
                continue;
            path.push_back(nums[i]);                  //做选择
            used[i]=true;                            //设置当前数已用
            backtrack(nums,used,path);              //进入下一层
            used[i]=false;                          //撤销选择
            path.pop_back();                        //撤销选择
        }
    }
}
```

## LC 554	砖墙（Middle）

题目（简单描述）：有一堵矩形的，由 n行砖块组成的砖墙。砖块的高度相同，宽度不同。画一条自顶向下的垂线，求穿过的最少砖块数。（不能沿着墙的两个垂直边缘之一画线）

思路：乍一看，以为 和 求 “穿过气球的最少箭数” 题目一样，先排序再求不 相交的区间数。

​           但不同的是，这个题目不需要排序，关键点在于，求解 “每行中的砖块 距离最左侧的距离”。距离相同的出现最多，即穿过的砖块数最少。（画图好理解，和 贪心无关）

```
class Solution {
public:
    int leastBricks(vector<vector<int>>& wall) {
        unordered_map<int,int> map;
        for(auto& walls:wall){
            int n=walls.size();
            int sum=0;
            for(int i=0;i<n-1;i++){
                sum+=walls[i];
                map[sum]++;
            }
        }
        int maxcnt=0;
        for(auto& [_,c]:map){    // 注意代码写法
            maxcnt=max(maxcnt,c);
        }
        return wall.size()-maxcnt;
    }
};
```

时间复杂度：O(nm)，其中 n是砖墙的高度，m 是每行砖墙的砖的平均数量。

空间复杂度：O(nm)

## LC 452. 用最少数量的箭引爆气球（Middle）

题目（概况一下）：提供 气球直径的开始和结束坐标，求 使得 所有气球全部被引爆，所需的弓箭的最小数量。

思路：Lc 554 是二维坐标，而 LC 452 其实是一维坐标。先根据 气球的结束坐标排序，然后其 区间数

```
class Solution {
public:
    int findMinArrowShots(vector<vector<int>>& points) {
        sort(points.begin(),points.end(),[](vector<int>a,vector<int>b){
            return a[1]<b[1];    // 按结束坐标排序
        });
        int r=points[0][1];
        int cnt=1;
        for(auto& poin:points){
            if(poin[0]>r){
                cnt+=1;
                r=poin[1];
            }
        }
        return cnt;
    }
};
```

时间复杂度：O(n log n)，n 是数组  points  的长度。排序的时间复杂度为 O(n log n)。

空间复杂度：O(log n)，即为排序需要使用的栈空间。

## LC 647. 回文子串

题目：计算字符串中有多少个回文子串。

思路：1、动态规划 dp [ i ] [ j ] ，表示以 i 为起始点，j 为中心点的 回文子串

```
class Solution {
public:
    int countSubstrings(string s) {
        vector<vector<bool>>dp(s.size(),vector<bool>(s.size()));
        int ans=0;
        for(int i=0;i<s.size();i++){   // 枚举可能的回文中心
            for(int j=0;j<=i;j++){
                if(s[i]==s[j]&& (i-j<2 || dp[j+1][i-1])){
                    dp[j][i]=true;
                    ans++;
                }
            }
        }
        return ans;
    }
};
时间复杂度为 O(N^2)；空间复杂度为 O(N^2)
```

2、中心扩散法，双指针

```
    int countSubstrings(string s) {
        int ans=0;
        // 枚举可能的中心位置
        for(int center=0;center<2*s.size()-1;center++){
            int left=center/2;
            int right=left+center%2;  // 可能是 奇数长度也可能是偶数长度
            while(left>=0 && right<s.size()  && s[left]==s[right]){
                 ans+=1;
                 left--;
                 right++;
            }
        }
        return ans;
    }
```

## LC 5. 最长回文子串（Middle）

可以依据 647 返回的所有回文子串中，找出最长的。

```
class Solution {
public:
    string longestPalindrome(string s) {
        string ans="";
        int maxlen=0;
        for(int cen=0;cen<2*s.size()-1;cen++){
            int left=cen/2;
            int right=left+cen%2;
            string temp;
            while(left>=0 && right<s.size() && s[left]==s[right]){
                left--;
                right++;
            }
            if(right-left+1>maxlen){
                ans=s.substr(left+1,right-left-1);
                maxlen=right-left+1;
            }
        }
        return ans;
    }
};
```

## LC 3. 无重复的最长子串 Middle

题目：给定一个字符串，找出其中不含有重复字符的最长子串的长度。

思路：滑动窗口（双指针），使用 unordered_set，判断 滑动窗口内的子串是否重复。

```
class Solution {
public:
    int lengthOfLongestSubstring(string s) {
        if(s.size()==0) return 0;
        unordered_set<char> map;
        int l=0,ans=0                 // 双指针
        for(int r=0;r<s.size();r++){
            while(map.find(s[r])!=map.end()){
                map.erase(s[l]);
                l++;
            }
            map.insert(s[r]);
            ans=max(ans,r-l+1);
        }
        return ans;
    }
};
时间复杂度：O(n)
```

## LC 76. 最小覆盖子串 （Hard）

题目：给你一个字符串 s，一个字符串 t。返回 s 中覆盖 t 所有字符的最小子串。

思路：滑动窗口，先右移完全覆盖 t 子串，再左移缩小边界，然后不断进行这个过程，同时记录最小子串。

```
class Solution {
public:
    string minWindow(string s, string t) {
        vector<int> map(128);   
        int left = 0, right = 0, need = t.size(), minStart = 0, minLen = INT_MAX;
        for(char ch : t)    ++map[ch];      //统计t中字符出现次数      
        while(right < s.size())
        {
            if(map[s[right]] > 0) --need;   //窗口右移，每包含一个t中的字符，need-1
            --map[s[right]];
            ++right;
            while(need == 0)    //完全覆盖子串时
            {
                if(right - left < minLen)   //此时字符被包含在[left,right)中
                {
                    minStart = left;
                    minLen = right - left;
                }
                if(++map[s[left]] > 0) ++need;  //窗口左移
                ++left;
            }
        }
        if(minLen != INT_MAX)   return s.substr(minStart, minLen);
        return "";
    }
};
```

## LC 137. 只出现一次的数字 II（Middle）剑指Offer 56-II

题目：给一个整数数组 nums，除某个元素仅出现一次外，其余每个元素都出现了三次，请返回那个只出现一次的元素。

思路：位运算

```
class Solution {
public:
    int singleNumber(vector<int>& nums) {
        int res=0;
        for(int i=0;i<32;i++){
            int cnt=0;                     // 分别计算每个 bit上的1个数
            for(auto& num:nums){
                if(num &(1<<i)) cnt+=1    // & 运算，计算出各位置上1的个数，并和3 求余
            }
            if((cnt%3)==1)
                res^=(1<<i);             // 用异或 方法生成二进制中的每一位
        }
        return res;
    }
};
```

