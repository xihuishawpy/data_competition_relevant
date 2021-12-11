
# 2021科大讯飞-车辆贷款违约预测挑战赛Top1--方案学习

## 简介

车贷违约预测问题，目的是建立风险识别模型来预测可能违约的借款人。预测结果为借款人是否可能违约，属于二分类问题。

偏`数据挖掘`的比赛，关键点是`如何基于对数据的理解抽象归纳出有用的特征`。

站在大佬的视角，尝试学习总结，站在巨人的肩膀上，也许看得会更远一些。

直接进入主题，开始学习套路，芜湖~


## 特征工程

### 1、常用库、数据导入

```python
import pandas as pd
import numpy as np
import lightgbm as lgb
import xgboost as xgb
from sklearn.metrics import roc_auc_score, auc, roc_curve, accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler, QuantileTransformer, KBinsDiscretizer, LabelEncoder, MinMaxScaler, PowerTransformer

from tqdm import tqdm
import pickle
import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
import os
```
后半部分用了一些工具：
- tqdm：一个优雅的进度条显示，方便观测跑数进度以及速度；
- pickle：将对象以文件的形式存放在磁盘上，几乎所有的数据类型都可以用pickle来序列化，一般先dump，后load，类似于写出、导入的意思；作用是，一次结果多次复用，避免重复做功，hhh，比如说A列数据处理得花2h，每次修改过后需重跑其他列数据，但无须修改A列数据，就可以用pickle解决这个问题，快速调取之前的结果；
- logging：控制台输出日志，方便查看运行状态；

```python
logging.info('data loading...')
train = pd.read_csv('../xfdata/车辆贷款违约预测数据集/train.csv')
test = pd.read_csv('../xfdata/车辆贷款违约预测数据集/test.csv')
```

### 2、特征工程

#### 2.1 构造特征 

针对训练集、测试集：
1. 根据业务理解，计算新的特征；
2. 对某些比例特征进行`等宽分箱`（cut），对某些数值特征进行`等频分箱`（qcut），还有一些数值特征进行自定义分箱，划分bin的范围；


```python

def gen_new_feats(train, test):
    '''生成新特征：如年利率/分箱等特征'''
    # Step 1: 合并训练集和测试集
    data = pd.concat([train, test])

    # Step 2: 具体特征工程
    # 计算二级账户的年利率
    data['sub_Rate'] = (data['sub_account_monthly_payment'] * data['sub_account_tenure'] - data[
        'sub_account_sanction_loan']) / data['sub_account_sanction_loan']

    # 计算主账户的年利率
    data['main_Rate'] = (data['main_account_monthly_payment'] * data['main_account_tenure'] - data[
        'main_account_sanction_loan']) / data['main_account_sanction_loan']

    # 对部分特征进行分箱操作
    # 等宽分箱
    loan_to_asset_ratio_labels = [i for i in range(10)]
    data['loan_to_asset_ratio_bin'] = pd.cut(data["loan_to_asset_ratio"], 10, labels=loan_to_asset_ratio_labels)
    # 等频分箱
    data['asset_cost_bin'] = pd.qcut(data['asset_cost'], 10, labels=loan_to_asset_ratio_labels)
    # 自定义分箱
    amount_cols = [
                   'total_monthly_payment',
                   'main_account_sanction_loan',
                   'main_account_disbursed_loan',
                   'sub_account_sanction_loan',
                   'sub_account_disbursed_loan',
                   'main_account_monthly_payment',
                   'sub_account_monthly_payment',
                   'total_sanction_loan'
                ]
    amount_labels = [i for i in range(10)]
    for col in amount_cols:
        total_monthly_payment_bin = [-1, 5000, 10000, 30000, 50000, 100000, 300000, 500000, 1000000, 3000000, data[col].max()]
        data[col + '_bin'] = pd.cut(data[col], total_monthly_payment_bin, labels=amount_labels).astype(int)

    # Step 3: 返回包含新特征的训练集 & 测试集
    return data[data['loan_default'].notnull()], data[data['loan_default'].isnull()]
```

#### 2.2 编码-Target Encoding

Target encoding是一种结合目标值进行特征编码的方式。

在二分类中，对于特征i，target encoding在该特征取值为k时的编码值为类别k对应的目标值期望E(y|xi=xik)。

![20211208003221](https://s2.loli.net/2021/12/08/198MoizYbUcaQ5f.png)

在样本集中一共有10条记录，其中3条记录中特征Trend的取值为Up，我们关注这3条记录。在k=Up时，目标值的期望为2/3 ≈ 0.66，所以将Up编码为0.66。

大佬后面主要是针对id特征进行target encoding。

```python
def gen_target_encoding_feats(train, test, encode_cols, target_col, n_fold=10):
    '''生成target encoding特征'''
    # for training set - cv
    tg_feats = np.zeros((train.shape[0], len(encode_cols)))
    kfold = StratifiedKFold(n_splits=n_fold, random_state=1024, shuffle=True)
    for _, (train_index, val_index) in enumerate(kfold.split(train[encode_cols], train[target_col])):
        df_train, df_val = train.iloc[train_index], train.iloc[val_index]
        for idx, col in enumerate(encode_cols):
            target_mean_dict = df_train.groupby(col)[target_col].mean()
            df_val[f'{col}_mean_target'] = df_val[col].map(target_mean_dict)
            tg_feats[val_index, idx] = df_val[f'{col}_mean_target'].values

    for idx, encode_col in enumerate(encode_cols):
        train[f'{encode_col}_mean_target'] = tg_feats[:, idx]

    # for testing set
    for col in encode_cols:
        target_mean_dict = train.groupby(col)[target_col].mean()
        test[f'{col}_mean_target'] = test[col].map(target_mean_dict)

    return train, test

```
说实话，这段代码还没完全看明白~先用小本本记着，用的时候先直接掏出来，hhh

#### 2.3 近邻欺诈特征

对于风控账户来说，存在风险的账户可能存在同批大量的注册情况，所以id可能是连着的。

这里大佬构建了近邻欺诈特征，就是每个账号的前后10个账户的lable取均值，也就代表着概率，意为可能违约账户聚集的概率，在一定程度上代表该账户可能违约的相关性。


```python
def gen_neighbor_feats(train, test):
    '''产生近邻欺诈特征'''
    if not os.path.exists('../user_data/neighbor_default_probs.pkl'):
        # 该特征需要跑的时间较久，因此将其存成了pkl文件
        neighbor_default_probs = []
        for i in tqdm(range(train.customer_id.max())):
            if i >= 10 and i < 199706:
                customer_id_neighbors = list(range(i - 10, i)) + list(range(i + 1, i + 10))
            elif i < 199706:
                customer_id_neighbors = list(range(0, i)) + list(range(i + 1, i + 10))
            else:
                customer_id_neighbors = list(range(i - 10, i)) + list(range(i + 1, 199706))

            customer_id_neighbors = [customer_id_neighbor for customer_id_neighbor in customer_id_neighbors if
                                     customer_id_neighbor in train.customer_id.values.tolist()]
            neighbor_default_prob = train.set_index('customer_id').loc[customer_id_neighbors].loan_default.mean()
            neighbor_default_probs.append(neighbor_default_prob)

        df_neighbor_default_prob = pd.DataFrame({'customer_id': range(0, train.customer_id.max()),
                                                 'neighbor_default_prob': neighbor_default_probs})
        save_pkl(df_neighbor_default_prob, '../user_data/neighbor_default_probs.pkl')
    else:
        df_neighbor_default_prob = load_pkl('../user_data/neighbor_default_probs.pkl')
    train = pd.merge(left=train, right=df_neighbor_default_prob, on='customer_id', how='left')
    test = pd.merge(left=test, right=df_neighbor_default_prob, on='customer_id', how='left')

    return train, test
```

#### 2.4 特征工程结果输出



```python
TARGET_ENCODING_FETAS = [
                            'employment_type',
                             'branch_id',
                             'supplier_id',
                             'manufacturer_id',
                             'area_id',
                             'employee_code_id',
                             'asset_cost_bin'
                         ]


# 特征工程
logging.info('feature generating...')
train, test = gen_new_feats(train, test)
train, test = gen_target_encoding_feats(train, test, TARGET_ENCODING_FETAS, target_col='loan_default', n_fold=10)
train, test = gen_neighbor_feats(train, test)
```
特征的后续处理，比如一些转换后特征的数据类型转换，一些率值特征的简化，方便后续的模型学习，增强模型的鲁棒性。
```python

# 保存的最终特征名称列表
SAVE_FEATS = [
                 'customer_id',
                 'neighbor_default_prob',
                 'disbursed_amount',
                 'asset_cost',
                 'branch_id',
                 'supplier_id',
                 'manufacturer_id',
                 'area_id',
                 'employee_code_id',
                 'credit_score',
                 'loan_to_asset_ratio',
                 'year_of_birth',
                 'age',
                 'sub_Rate',
                 'main_Rate',
                 'loan_to_asset_ratio_bin',
                 'asset_cost_bin',
                 'employment_type_mean_target',
                 'branch_id_mean_target',
                 'supplier_id_mean_target',
                 'manufacturer_id_mean_target',
                 'area_id_mean_target',
                 'employee_code_id_mean_target',
                 'asset_cost_bin_mean_target',
                 'credit_history',
                 'average_age',
                 'total_disbursed_loan',
                 'main_account_disbursed_loan',
                 'total_sanction_loan',
                 'main_account_sanction_loan',
                 'active_to_inactive_act_ratio',
                 'total_outstanding_loan',
                 'main_account_outstanding_loan',
                 'Credit_level',
                 'outstanding_disburse_ratio',
                 'total_account_loan_no',
                 'main_account_tenure',
                 'main_account_loan_no',
                 'main_account_monthly_payment',
                 'total_monthly_payment',
                 'main_account_active_loan_no',
                 'main_account_inactive_loan_no',
                 'sub_account_inactive_loan_no',
                 'enquirie_no',
                 'main_account_overdue_no',
                 'total_overdue_no',
                 'last_six_month_defaulted_no'
            ]


# 特征工程 后处理
# 简化特征
for col in ['sub_Rate', 'main_Rate', 'outstanding_disburse_ratio']:
     train[col] = train[col].apply(lambda x: 1 if x > 1 else x)
     test[col] = test[col].apply(lambda x: 1 if x > 1 else x)

# 数据类型转换
train['asset_cost_bin'] = train['asset_cost_bin'].astype(int)
test['asset_cost_bin'] = test['asset_cost_bin'].astype(int)
train['loan_to_asset_ratio_bin'] = train['loan_to_asset_ratio_bin'].astype(int)
test['loan_to_asset_ratio_bin'] = test['loan_to_asset_ratio_bin'].astype(int)

# 存储包含新特征的数据集
logging.info('new data saving...')
cols = SAVE_FEATS + ['loan_default', ]
train[cols].to_csv('./train_final.csv', index=False)
test[cols].to_csv('./test_final.csv', index=False)
```


## 模型构建

### 1、模型训练-交叉验证

采用lightgbm、xgboost两种梯度提升树模型，这里不多解释了，下面代码都成了“标准”，DDDD~

```python
def train_lgb_kfold(X_train, y_train, X_test, n_fold=5):
    '''train lightgbm with k-fold split'''
    gbms = []
    kfold = StratifiedKFold(n_splits=n_fold, random_state=1024, shuffle=True)
    oof_preds = np.zeros((X_train.shape[0],))
    test_preds = np.zeros((X_test.shape[0],))

    for fold, (train_index, val_index) in enumerate(kfold.split(X_train, y_train)):
        logging.info(f'############ fold {fold} ###########')
        X_tr, X_val, y_tr, y_val = X_train.iloc[train_index], X_train.iloc[val_index], y_train[train_index], y_train[val_index]
        dtrain = lgb.Dataset(X_tr, y_tr)
        dvalid = lgb.Dataset(X_val, y_val, reference=dtrain)

        params = {
            'objective': 'binary',
            'metric': 'auc',
            'num_leaves': 64,
            'learning_rate': 0.02,
            'min_data_in_leaf': 150,
            'feature_fraction': 0.8,
            'bagging_fraction': 0.7,
            'n_jobs': -1,
            'seed': 1024
        }

        gbm = lgb.train(params,
                        dtrain,
                        num_boost_round=1000,
                        valid_sets=[dtrain, dvalid],
                        verbose_eval=50,
                        early_stopping_rounds=20)

        oof_preds[val_index] = gbm.predict(X_val, num_iteration=gbm.best_iteration)
        test_preds += gbm.predict(X_test, num_iteration=gbm.best_iteration) / kfold.n_splits
        gbms.append(gbm)

    return gbms, oof_preds, test_preds



def train_xgb_kfold(X_train, y_train, X_test, n_fold=5):
    '''train xgboost with k-fold split'''
    gbms = []
    kfold = StratifiedKFold(n_splits=10, random_state=1024, shuffle=True)
    oof_preds = np.zeros((X_train.shape[0],))
    test_preds = np.zeros((X_test.shape[0],))

    for fold, (train_index, val_index) in enumerate(kfold.split(X_train, y_train)):
        logging.info(f'############ fold {fold} ###########')
        X_tr, X_val, y_tr, y_val = X_train.iloc[train_index], X_train.iloc[val_index], y_train[train_index], y_train[val_index]
        dtrain = xgb.DMatrix(X_tr, y_tr)
        dvalid = xgb.DMatrix(X_val, y_val)
        dtest = xgb.DMatrix(X_test)

        params={
            'booster':'gbtree',
            'objective': 'binary:logistic',
            'eval_metric': ['logloss', 'auc'],
            'max_depth': 8,
            'subsample':0.9,
            'min_child_weight': 10,
            'colsample_bytree':0.85,
            'lambda': 10,
            'eta': 0.02,
            'seed': 1024
        }

        watchlist = [(dtrain, 'train'), (dvalid, 'test')]

        gbm = xgb.train(params,
                        dtrain,
                        num_boost_round=1000,
                        evals=watchlist,
                        verbose_eval=50,
                        early_stopping_rounds=20)

        oof_preds[val_index] = gbm.predict(dvalid, iteration_range=(0, gbm.best_iteration))
        test_preds += gbm.predict(dtest, iteration_range=(0, gbm.best_iteration)) / kfold.n_splits
        gbms.append(gbm)

    return gbms, oof_preds, test_preds
```

```python
def train_xgb(train, test, feat_cols, label_col, n_fold=10):
    '''训练xgboost'''
    for col in ['sub_Rate', 'main_Rate', 'outstanding_disburse_ratio']:
        train[col] = train[col].apply(lambda x: 1 if x > 1 else x)
        test[col] = test[col].apply(lambda x: 1 if x > 1 else x)

    X_train = train[feat_cols]
    y_train = train[label_col]
    X_test = test[feat_cols]
    gbms_xgb, oof_preds_xgb, test_preds_xgb = train_xgb_kfold(X_train, y_train, X_test, n_fold=n_fold)

    if not os.path.exists('../user_data/gbms_xgb.pkl'):
        save_pkl(gbms_xgb, '../user_data/gbms_xgb.pkl')

    return gbms_xgb, oof_preds_xgb, test_preds_xgb


def train_lgb(train, test, feat_cols, label_col, n_fold=10):
    '''训练lightgbm'''
    X_train = train[feat_cols]
    y_train = train[label_col]
    X_test = test[feat_cols]
    gbms_lgb, oof_preds_lgb, test_preds_lgb = train_lgb_kfold(X_train, y_train, X_test, n_fold=n_fold)

    if not os.path.exists('../user_data/gbms_lgb.pkl'):
        save_pkl(gbms_lgb, '../user_data/gbms_lgb.pkl')

    return gbms_lgb, oof_preds_lgb, test_preds_lgb
```

输出模型训练结果：

```python
# 读取原始数据集
logging.info('data loading...')
train = pd.read_csv('../xfdata/车辆贷款违约预测数据集/train.csv')
test = pd.read_csv('../xfdata/车辆贷款违约预测数据集/test.csv')

# 特征工程
logging.info('feature generating...')
train, test = gen_new_feats(train, test)
train, test = gen_target_encoding_feats(train, test, TARGET_ENCODING_FETAS, target_col='loan_default', n_fold=10)
train, test = gen_neighbor_feats(train, test)

train['asset_cost_bin'] = train['asset_cost_bin'].astype(int)
test['asset_cost_bin'] = test['asset_cost_bin'].astype(int)
train['loan_to_asset_ratio_bin'] = train['loan_to_asset_ratio_bin'].astype(int)
test['loan_to_asset_ratio_bin'] = test['loan_to_asset_ratio_bin'].astype(int)
train['asset_cost_bin_mean_target'] = train['asset_cost_bin_mean_target'].astype(float)
test['asset_cost_bin_mean_target'] = test['asset_cost_bin_mean_target'].astype(float)

# 模型训练：linux和mac的xgboost结果会有些许不同，以模型文件结果为主
gbms_xgb, oof_preds_xgb, test_preds_xgb = train_xgb(train.copy(), test.copy(),
                                                    feat_cols=SAVE_FEATS,
                                                    label_col='loan_default')
gbms_lgb, oof_preds_lgb, test_preds_lgb = train_lgb(train, test,
                                                    feat_cols=SAVE_FEATS,
                                                    label_col='loan_default')

```

### 2、划分阈值

因为是`0-1二分类`，最终分类的均值，可近似理解为取到loan_default=1的概率。
再通过对cv的预测结果排序，取分位数（1-P(loan_default=1)）对应的概率为预测正负样本的划分的临界点。

为了让结果更精准，采取小步长遍历临界点附近的点，找到局部最优的概率阈值。

```python
def gen_thres_new(df_train, oof_preds):
    df_train['oof_preds'] = oof_preds
    # 可看作训练集取到loan_default=1的概率
    quantile_point = df_train['loan_default'].mean() 
    thres = df_train['oof_preds'].quantile(1 - quantile_point) 
    # 比如 0,1,1,1 mean=0.75 1-mean=0.25,也就是25%分位数取值为0

    _thresh = []
     #  按照理论阈值的上下0.2范围，0.01步长，找到最佳阈值，f1分数最高对应的阈值即为最佳阈值
    for thres_item in np.arange(thres - 0.2, thres + 0.2, 0.01):
        _thresh.append(
            [thres_item, f1_score(df_train['loan_default'], np.where(oof_preds > thres_item, 1, 0), average='macro')])

    _thresh = np.array(_thresh)
    best_id = _thresh[:, 1].argmax() # 找到f1最高对应的行
    best_thresh = _thresh[best_id][0] # 取出最佳阈值

    print("阈值: {}\n训练集的f1: {}".format(best_thresh, _thresh[best_id][1]))
    return best_thresh
```

### 3、模型融合

对xgb、lgb的模型cv结果的分位数进行`加权求和`，再去找融合后的模型0-1的概率阈值。

```python
xgb_thres = gen_thres_new(train, oof_preds_xgb)
lgb_thres =  gen_thres_new(train, oof_preds_lgb)


# 结果聚合
df_oof_res = pd.DataFrame({'customer_id': train['customer_id'],
                            'loan_default':train['loan_default'],
                            'oof_preds_xgb': oof_preds_xgb,
                            'oof_preds_lgb': oof_preds_lgb})

# 模型融合
df_oof_res['xgb_rank'] = df_oof_res['oof_preds_xgb'].rank(pct=True) # percentile rank,返回的是排序后的分位数
df_oof_res['lgb_rank'] = df_oof_res['oof_preds_lgb'].rank(pct=True)

df_oof_res['preds'] = 0.31 * df_oof_res['xgb_rank'] + 0.69 * df_oof_res['lgb_rank']

# 融合后的模型，概率阈值
thres = gen_thres_new(df_oof_res, df_oof_res['preds'])
```

## 预测

按照融模后训练集的概率阈值，对测试集预测结果进行0-1划分，输出最终预测提交结果。

```python

def gen_submit_file(df_test, test_preds, thres, save_path):
    # 按最终模型融合后的阈值进行划分
    df_test['test_preds_binary'] = np.where(test_preds > thres, 1, 0)  
    df_test_submit = df_test[['customer_id', 'test_preds_binary']]
    df_test_submit.columns = ['customer_id', 'loan_default']
    print(f'saving result to: {save_path}')
    df_test_submit.to_csv(save_path, index=False)
    print('done!')
    return df_test_submit



df_test_res = pd.DataFrame({'customer_id': test['customer_id'],
                                'test_preds_xgb': test_preds_xgb,
                                'test_preds_lgb': test_preds_lgb})

df_test_res['xgb_rank'] = df_test_res['test_preds_xgb'].rank(pct=True)
df_test_res['lgb_rank'] = df_test_res['test_preds_lgb'].rank(pct=True)
df_test_res['preds'] = 0.31 * df_test_res['xgb_rank'] + 0.69 * df_test_res['lgb_rank']

# 结果产出
df_submit = gen_submit_file(df_test_res, df_test_res['preds'], thres,
                            save_path='../prediction_result/result.csv')
```


## 总结

大佬的代码风格清晰、简洁，看代码非常流畅，思路也非常清晰，可以好好学习这些工程化的代码，可拓展性强，方便debug。

从赛题角度看，对业务的思考后从id集中度上做了一个“近邻欺诈特征”；在融模操作上，按预测值的ranking值分位数加权。这些小技巧都是可直接复用的~（也是大佬提到的上分点）

下面2个问题，估计很多同学和我一样也都会有些疑惑，我就从b乎直接截图出来：

![20211208161333](https://s2.loli.net/2021/12/08/suALQIGcOlWFY4i.png)


源码：https://github.com/WangliLin/xunfei2021_car_loan_top1

另外，我也整理了个ipynb，方便学习，需要的同学公众号后台回复“1208”获取

---

参考：
1. [logging模块](https://blog.csdn.net/pansaky/article/details/90710751)
2. [pickle模块](https://blog.csdn.net/sinat_29552923/article/details/70833455)
3. [tqdm模块](https://blog.csdn.net/qq_33472765/article/details/82940843)
4. [Target Encoding公式](https://blog.csdn.net/Dynomite/article/details/82886504)
5. [Target Encoding](https://www.cnblogs.com/lryou/p/14627564.html)
6. https://zhuanlan.zhihu.com/p/412337232










