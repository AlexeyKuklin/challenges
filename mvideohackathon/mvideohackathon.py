# -*- coding: utf-8 -*-
import pandas as pd
import hashlib
from scipy.sparse import csr_matrix
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import SGDClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import LabelEncoder


def ng(s, n, rid, rows, cols, data):
    ngrams = lambda n: [s[i:i+n] for i in range(0, len(s)-n+1)]   

    for s1 in ngrams(n):
        h = hashlib.md5(s1.strip().encode('utf8'))
        cid = abs(int(h.hexdigest(), 16)%2**25)
        rows.append(rid)
        cols.append(cid)
        data.append(1)
    return rows, cols, data              


def get_data():
    rows, cols, data, y = [], [], [], []

    df = pd.read_csv('X_train.csv')
    df = df[['reting',
             'comment',
             'commentNegative',
             'commentPositive']]

    for rid, r in df.iterrows():
        y.append(int(r['reting']))
        
        s = r['comment'].lower() + ' ' + str(r['commentNegative']).lower() + ' ' + str(r['commentPositive']).lower()
        rows, cols, data = ng(s, 8, rid, rows, cols, data)

    rows.append(rows[-1])
    cols.append(2**25+1)
    data.append(1)

    X = csr_matrix((data, (rows, cols)))
    return X, y


clf = SGDClassifier(alpha=0.0001,
                    average=False, 
                    class_weight=None, 
                    epsilon=0.1,
                    eta0=0.0, 
                    fit_intercept=True, 
                    l1_ratio=0.15,
                    learning_rate='optimal', 
                    loss='hinge', 
                    n_iter=100,
                    n_jobs=1,
                    penalty='l2', 
                    power_t=0.5, 
                    random_state=42,
                    shuffle=True,
                    verbose=0,
                    warm_start=False)

clf = OneVsRestClassifier(clf)
le = LabelEncoder()

X_train, y_train = get_data()
y_train = le.fit_transform(y_train)
print(cross_val_score(clf, X_train, y_train, 
                      groups=None, 
                      scoring=None, 
                      cv=5, 
                      n_jobs=1, 
                      verbose=0, 
                      fit_params=None))
