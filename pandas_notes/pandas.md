# Note for Pandas

##1. DataFrame的创建

###1.1 通过numpy array创建

```
>>> df2 = pd.DataFrame(np.random.randint(low=0, high=10, size=(5, 5)),
...                    columns=['a', 'b', 'c', 'd', 'e'])
>>> df2
    a   b   c   d   e
0   2   8   8   3   4
1   4   2   9   0   9
2   1   0   7   8   0
3   5   1   7   1   3
4   6   0   2   4   2
```

### 1.2 通过字典创建

```
>>> d = {'col1': [1, 2], 'col2': [3, 4]}
>>> df = pd.DataFrame(data=d)
>>> df
   col1  col2
0     1     3
1     2     4
```



