# -*- coding: utf-8 -*-
"""
Created on Mon Jul 22 06:49:31 2019

@author: Tadhg Lynch
"""

### Mapping on normalized gold standard

import numpy as np


from sklearn.feature_extraction.text import TfidfVectorizer


from Fuzzy_Normalizing import cleaning

from sklearn.neighbors import NearestNeighbors



def mapping(df_train, train_column, df_test, test_column, dist_threshold):
    
    df_train = cleaning(df_train, train_column)
    df_test = cleaning(df_test,test_column)

    
    vectorizer_train = TfidfVectorizer(min_df=1, analyzer="char", ngram_range=(3,3)).fit(df_train['clean_brand'].fillna('Not Specified'))
    tfidf_train = vectorizer_train.transform(df_train['clean_brand'].fillna('Not Specified'))

    df_nn = df_test
    hjk = df_nn['clean_brand'].fillna('Not Specified')
    asd = vectorizer_train.transform(hjk)
    nn = NearestNeighbors(n_neighbors=1, metric="cosine").fit(tfidf_train)
    dist, nearest = nn.kneighbors(asd)


    #defo a better way to to this, ask later... literally just the list of index values with len = 
    ind = list()
    x=0
    for i in dist:
        ind.append(x)
        x = x+1
    ind = np.asarray(ind)


    op = np.column_stack((ind,nearest,dist))
    
    
    kl = [possible for possible in op if possible[2]<dist_threshold]
    kl = [possible for possible in kl if possible[2]>0]

    fg = list()
    gh = list()
    for i in kl:
        fg.append(int(i[0]))
        gh.append(int(i[1]))
    
    jk = dict(zip(fg,gh))
    
    dict_final = {k:df_train[train_column][v] for k,v in jk.items()}
    
    #maybe make a dictionary of starting ind_str k to dict_final
    
    df_nn['ind_str'] = df_nn.index.values
    df_nn['ind_str'] = df_nn.ind_str.apply(int)
    
    
    df_nn['final_output'] = df_nn['ind_str'].map(dict_final)
    
    df_nn_final = df_nn[[test_column, 'final_output']]
    
    df_nn_changes = df_nn_final.dropna(axis = 0)

    return df_nn, df_nn_final, df_nn_changes




