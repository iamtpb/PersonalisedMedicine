from sklearn import *
import sklearn
import pandas as pd
import numpy as np
import xgboost as xgb


#######
train = pd.read_csv('data/training_variants')
test = pd.read_csv('data/stage2_test_variants.csv')
trainx = pd.read_csv('data/training_text', sep="\|\|", engine='python', header=None, skiprows=1, names=["ID","Text"])
testx = pd.read_csv('data/stage2_test_text.csv', sep="\|\|", engine='python', header=None, skiprows=1, names=["ID","Text"])
##
train = pd.merge(train, trainx, how='left', on='ID').fillna('')
train = train.drop(["ID"], axis=1)

test = pd.merge(test, testx, how='left', on='ID').fillna('')
pid = test['ID'].values
##
df_variants_test = pd.read_csv('data/test_variants', usecols=['ID', 'Gene', 'Variation'])
df_text_test = pd.read_csv('data/test_text', sep='\|\|', engine='python',
                           skiprows=1, names=['ID', 'Text'])
df_variants_test['Text'] = df_text_test['Text']
df_test = df_variants_test

# read stage1 solutions
df_labels_test = pd.read_csv('data/stage1_solution_filtered.csv')
df_labels_test['Class'] = pd.to_numeric(df_labels_test.drop('ID', axis=1).idxmax(axis=1).str[5:])

# join with test_data on same indexes
df_test = df_test.merge(df_labels_test[['ID', 'Class']], on='ID', how='left').drop('ID', axis=1)
df_test = df_test[df_test['Class'].notnull()]

# join train and test files
df_stage_2_train = pd.concat([train, df_test])
df_stage_2_train.reset_index(drop=True, inplace=True)
train = df_stage_2_train

y = train['Class'].values
train = train.drop(['Class'], axis=1)
pid = test['ID'].values

df_all = pd.concat((train, test), axis=0, ignore_index=True)
df_all['Gene_Share'] = df_all.apply(lambda r: sum([1 for w in r['Gene'].split(' ') if w in r['Text'].split(' ')]), axis=1)
df_all['Variation_Share'] = df_all.apply(lambda r: sum([1 for w in r['Variation'].split(' ') if w in r['Text'].split(' ')]), axis=1)

for i in range(56):
    df_all['Gene_'+str(i)] = df_all['Gene'].map(lambda x: str(x[i]) if len(x)>i else '')
    df_all['Variation'+str(i)] = df_all['Variation'].map(lambda x: str(x[i]) if len(x)>i else '')


gen_var_lst = sorted(list(train.Gene.unique()) + list(train.Variation.unique()))
print(len(gen_var_lst))
gen_var_lst = [x for x in gen_var_lst if len(x.split(' '))==1]
print(len(gen_var_lst))
i_ = 0
for gen_var_lst_itm in gen_var_lst:
    if i_ % 100 == 0: print(i_)
    df_all['GV_'+str(gen_var_lst_itm)] = df_all['Text'].map(lambda x: str(x).count(str(gen_var_lst_itm)))
    i_ += 1
print('step1')
for c in df_all.columns:
    if df_all[c].dtype == 'object':
        if c in ['Gene','Variation']:
            lbl = preprocessing.LabelEncoder()
            df_all[c+'_lbl_enc'] = lbl.fit_transform(df_all[c].values)
            df_all[c+'_len'] = df_all[c].map(lambda x: len(str(x)))
            df_all[c+'_words'] = df_all[c].map(lambda x: len(str(x).split(' ')))
        elif c != 'Text':
            lbl = preprocessing.LabelEncoder()
            df_all[c] = lbl.fit_transform(df_all[c].values)
        if c=='Text':
            df_all[c+'_len'] = df_all[c].map(lambda x: len(str(x)))
            df_all[c+'_words'] = df_all[c].map(lambda x: len(str(x).split(' ')))

train = df_all.iloc[:len(train)]
test = df_all.iloc[len(train):]
print('step2')
class cust_regression_vals(sklearn.base.BaseEstimator, sklearn.base.TransformerMixin):
    def fit(self, x, y=None):
        return self
    def transform(self, x):
        x = x.drop(['Gene', 'Variation','ID','Text'],axis=1).values
        return x

class cust_txt_col(sklearn.base.BaseEstimator, sklearn.base.TransformerMixin):
    def __init__(self, key):
        self.key = key
    def fit(self, x, y=None):
        return self
    def transform(self, x):
        return x[self.key].apply(str)

print('Pipeline...')
fp = pipeline.Pipeline([
    ('union', pipeline.FeatureUnion(
        n_jobs = -1,
        transformer_list = [
            ('standard', cust_regression_vals()),
            ('pi1', pipeline.Pipeline([('Gene', cust_txt_col('Gene')), ('count_Gene', feature_extraction.text.CountVectorizer(analyzer=u'char', ngram_range=(1, 8))), ('tsvd1', decomposition.TruncatedSVD(n_components=20, n_iter=25, random_state=12))])),
            ('pi2', pipeline.Pipeline([('Variation', cust_txt_col('Variation')), ('count_Variation', feature_extraction.text.CountVectorizer(analyzer=u'char', ngram_range=(1, 8))), ('tsvd2', decomposition.TruncatedSVD(n_components=20, n_iter=25, random_state=12))])),
            ('pi3', pipeline.Pipeline([('Text', cust_txt_col('Text')), ('tfidf_Text', feature_extraction.text.TfidfVectorizer(ngram_range=(1, 2))), ('tsvd3', decomposition.TruncatedSVD(n_components=25, n_iter=20, random_state=12))]))
        ])
    )])
print('step3')
train = fp.fit_transform(train); print(train.shape)
test = fp.transform(test); print(test.shape)

y = y - 1

denom = 0
fold = 5
for i in range(fold):
    params = {
        'eta': 0.03333,
        'max_depth': 4,
        'objective': 'multi:softprob',
        'eval_metric': 'mlogloss',
        'num_class': 9,
        'seed': i,
        'silent': True
    }
    x1, x2, y1, y2 = model_selection.train_test_split(train, y, test_size=0.18, random_state=i)
    watchlist = [(xgb.DMatrix(x1, y1), 'train'), (xgb.DMatrix(x2, y2), 'valid')]
    model = xgb.train(params, xgb.DMatrix(x1, y1), 1000,  watchlist, verbose_eval=50, early_stopping_rounds=100)
    score1 = metrics.log_loss(y2, model.predict(xgb.DMatrix(x2), ntree_limit=model.best_ntree_limit), labels = list(range(9)))
    print(score1)
    #if score < 0.9:
    if denom != 0:
        pred = model.predict(xgb.DMatrix(test), ntree_limit=model.best_ntree_limit+80)
        preds += pred
    else:
        pred = model.predict(xgb.DMatrix(test), ntree_limit=model.best_ntree_limit+80)
        preds = pred.copy()
    denom += 1
    submission = pd.DataFrame(pred, columns=['class'+str(c+1) for c in range(9)])
    submission['ID'] = pid
    submission.to_csv('submission_xgb_fold_'  + str(i) + '.csv', index=False)
preds /= denom
print('step4')
submission = pd.DataFrame(preds, columns=['class'+str(c+1) for c in range(9)])
submission['ID'] = pid
submission.to_csv('submission_xgb.csv', index=False)


"""
/home/xipher/Projects/ml/bin/python /home/xipher/cancer/xgb.py
3578
3397

step1
step2
Pipeline...
step3
(3689, 3584)
(986, 3584)
[0]	train-mlogloss:2.14193	valid-mlogloss:2.14831
Multiple eval metrics have been passed: 'valid-mlogloss' will be used for early stopping.

Will train until valid-mlogloss hasn't improved in 100 rounds.
[50]	train-mlogloss:1.12655	valid-mlogloss:1.28591
[100]	train-mlogloss:0.850921	valid-mlogloss:1.08847
[150]	train-mlogloss:0.713036	valid-mlogloss:1.01956
[200]	train-mlogloss:0.618859	valid-mlogloss:0.981633
[250]	train-mlogloss:0.548723	valid-mlogloss:0.96192
[300]	train-mlogloss:0.490403	valid-mlogloss:0.9486
[350]	train-mlogloss:0.444124	valid-mlogloss:0.941121
[400]	train-mlogloss:0.404225	valid-mlogloss:0.936599
[450]	train-mlogloss:0.369022	valid-mlogloss:0.932737
[500]	train-mlogloss:0.337283	valid-mlogloss:0.932471
[550]	train-mlogloss:0.308373	valid-mlogloss:0.933672
Stopping. Best iteration:
[454]	train-mlogloss:0.366327	valid-mlogloss:0.932371

0.932370906974
[0]	train-mlogloss:2.14236	valid-mlogloss:2.14479
Multiple eval metrics have been passed: 'valid-mlogloss' will be used for early stopping.

Will train until valid-mlogloss hasn't improved in 100 rounds.
[50]	train-mlogloss:1.12951	valid-mlogloss:1.23812
[100]	train-mlogloss:0.853616	valid-mlogloss:1.03694
[150]	train-mlogloss:0.710123	valid-mlogloss:0.960808
[200]	train-mlogloss:0.619246	valid-mlogloss:0.924804
[250]	train-mlogloss:0.54841	valid-mlogloss:0.902785
[300]	train-mlogloss:0.490213	valid-mlogloss:0.888703
[350]	train-mlogloss:0.439826	valid-mlogloss:0.878176
[400]	train-mlogloss:0.396904	valid-mlogloss:0.870031
[450]	train-mlogloss:0.360832	valid-mlogloss:0.863673
[500]	train-mlogloss:0.328437	valid-mlogloss:0.859742
[550]	train-mlogloss:0.300626	valid-mlogloss:0.858792
[600]	train-mlogloss:0.275422	valid-mlogloss:0.860255
Stopping. Best iteration:
[529]	train-mlogloss:0.311912	valid-mlogloss:0.85858

0.858579588984
[0]	train-mlogloss:2.14222	valid-mlogloss:2.14635
Multiple eval metrics have been passed: 'valid-mlogloss' will be used for early stopping.

Will train until valid-mlogloss hasn't improved in 100 rounds.
[50]	train-mlogloss:1.13294	valid-mlogloss:1.24326
[100]	train-mlogloss:0.860897	valid-mlogloss:1.04274
[150]	train-mlogloss:0.717377	valid-mlogloss:0.969304
[200]	train-mlogloss:0.620384	valid-mlogloss:0.931273
[250]	train-mlogloss:0.54785	valid-mlogloss:0.903654
[300]	train-mlogloss:0.489024	valid-mlogloss:0.884044
[350]	train-mlogloss:0.43721	valid-mlogloss:0.872305
[400]	train-mlogloss:0.394807	valid-mlogloss:0.862066
[450]	train-mlogloss:0.35931	valid-mlogloss:0.855991
[500]	train-mlogloss:0.328029	valid-mlogloss:0.851434
[550]	train-mlogloss:0.301566	valid-mlogloss:0.848773
[600]	train-mlogloss:0.278056	valid-mlogloss:0.846488
[650]	train-mlogloss:0.256693	valid-mlogloss:0.845488
[700]	train-mlogloss:0.236355	valid-mlogloss:0.844992
[750]	train-mlogloss:0.217565	valid-mlogloss:0.844855
[800]	train-mlogloss:0.199898	valid-mlogloss:0.845408
Stopping. Best iteration:
[721]	train-mlogloss:0.228058	valid-mlogloss:0.844449

0.84444884973
"""