# -*- coding: UTF-8 -*-

import math
import numpy as np
import pandas as pd
from modules import *


def grow_tree(df, max_height, num_columns, current_height):
    sample_size = df.shape[0]
    if current_height >= max_height or sample_size <= 1:
        return ITreeLeaf(sample_size)
    split_column = np.random.randint(0, num_columns-1)
    col_min = df.iloc[:, split_column].min()
    col_max = df.iloc[:, split_column].max()
    split_value = col_min + np.random.random() * (col_max - col_min)       #随机从col_min到col_max之间取一个数
    X_left = df[df.iloc[:, split_column] < split_value]
    X_right = df[df.iloc[:, split_column] >= split_value]
    return ITreeBranch(
        grow_tree(X_left, max_height, num_columns, current_height + 1),
        grow_tree(X_right, max_height, num_columns, current_height + 1),
        split_column,
        split_value)


def buildForest(data, num_trees=100, subsample_size=256):
    num_columns = len(data.columns)
    max_height = int(math.ceil(math.log(subsample_size)))
    
    data_size = data.count()
    all_row_inds = np.arange(data_size)
    multi_sample_inds = [np.random.choice(all_row_inds, subsample_size, replace=False) for i in range(num_trees)]
    del all_row_inds
    
    data_rdd = data.rdd.zipWithIndex().map(lambda (v,k): (k,v))
    rdds = data_rdd.filter(lambda x: x[0] in multi_sample_inds[0]).map(lambda x: (0, x[1]))
    for i in range(1, num_trees):
        rdds = rdds.union(
            data_rdd.filter(lambda x: x[0] in multi_sample_inds[i]).map(lambda x: (i, x[1])))
    df_rdds = rdds.groupByKey().map(lambda (tree_id, samples): (tree_id, pd.DataFrame(list(samples))))
    
    trained_trees = df_rdds.map(lambda x: grow_tree(x[1], max_height, num_columns, 0)).collect()
    return IsolationForest(subsample_size, trained_trees)



if __name__ == "__main__":
    from pyspark.sql import SparkSession
    session = SparkSession.builder.appName("chenxingwei_test_02").enableHiveSupport().getOrCreate()
    train_data = session.sql('select * from turing.credit_card_test').drop('next_month')
    test_data = train_data

    import time
    t1 = time.time()
    forest = buildForest(train_data, num_trees=100, subsample_size=512)
    t2 = time.time()
    print('\nTime for building forest: ' + str(t2-t1) + ' s')

    t3 = time.time()
    scores = test_data.rdd.map(lambda row: forest.predict(row)).collect()
    print(scores[:10])
    t4 = time.time()
    print('\nTime for prediction: ' + str(t4-t3) + ' s')

    print('\nNumber of anomas:')
    print(pd.Series(scores) >= 0.6).value_counts()
