# -*- coding: utf-8 -*-
"""
Created on Mon Jul 22 06:18:16 2019

@author: Tadhg Lynch
"""



import pandas as pd
import numpy as np
import re

from sklearn.feature_extraction.text import TfidfVectorizer

from scipy.sparse import csr_matrix
import sparse_dot_topn.sparse_dot_topn as ct



#cleans
def cleaning(df_input, brands):
    
    df_input['clean_brand']= df_input[brands].str.lower().replace('([^a-zA-Z0-9& \(\)]+?)', ' ', regex=True)

    #replace stop words with nothing
    stop_words = ['industries','inc','inc.','cor','enterprises','international','studios','llc','ltd','systems','housewares',
                  'incorporation','corporation', 'electronics','group','usa', 'entertainment','distributors',',','exclusive','artist',
                  'illustrator', 'actor', 'director', 'director','editor', 'translator', 'creator', 'photographer', 'ntributor',
                  'tm','writer','publisher','producer', 'performer', 'pl', 'author', 'co', 'announcement']
    for i in stop_words:
        df_input['clean_brand'] = df_input.clean_brand.replace(r'\b%s\b' % i,'', regex=True) #\b%s\b removes i not attached to anything else

    ## remove values contained in paranthesis
    df_input['clean_brand'] = df_input.clean_brand.replace("\([^\)]+\)", "", regex=True)

    #clean up white spaces 
    df_input['clean_brand']=df_input.clean_brand.apply(lambda x: re.sub('(\s{2,})',' ',str(x))).str.strip()
    
    return df_input


def cosine_sim_matrix(A, B, ntop, lower_bound = 0):
    A = A.tocsr()
    B = B.tocsr()
    M, _ = A.shape
    _, N = B.shape
    
    idx_dtype = np.int32
    
    nnz_max = M*ntop
    
    indptr = np.zeros(M+1, dtype=idx_dtype)
    indices = np.zeros(nnz_max, dtype=idx_dtype)
    data = np.zeros(nnz_max, dtype=A.dtype)
    
    ct.sparse_dot_topn(
        M, N, np.asarray(A.indptr, dtype=idx_dtype),
        np.asarray(A.indices, dtype=idx_dtype),
        A.data,
        np.asarray(B.indptr, dtype=idx_dtype),
        np.asarray(B.indices, dtype=idx_dtype),
        B.data,
        ntop,
        lower_bound,
        indptr, indices, data)
    
    return csr_matrix((data,indices,indptr),shape=(M,N))



def get_matches_df(sparse_matrix, name_vector, top):   # top is set to capture all that fit criteria above
    non_zeros = sparse_matrix.nonzero()
    
    sparserows = non_zeros[0]
    sparsecols = non_zeros[1]
    
    if top:
        nr_matches = top
    else:
        nr_matches = sparsecols.size
    
    left_side = np.empty([nr_matches], dtype=object)
    right_side = np.empty([nr_matches], dtype=object)
    similarity = np.zeros(nr_matches)
    
    for index in range(0, nr_matches):
        left_side[index] = name_vector[sparserows[index]]
        right_side[index] = name_vector[sparsecols[index]]
        similarity[index] = sparse_matrix.data[index]
    
    return pd.DataFrame({'left_side': left_side.tolist(),
                          'right_side': right_side.tolist(),
                           'similarity': similarity.tolist()})
    

    
#chooses normalized value off of highest count
def highest_count(dframe,df_input,counts):
    dict_count = df_input[counts].to_dict()
    dict_output = {}
    for i in dframe.right_side:
        split = i.split(',')
        max_split = []
        for j in split:
            max_split.append(dict_count[int(j)])
        output = split[np.argmax(max_split)]
        list_output = [output]*len(split)
        dict_output.update((zip(split,list_output)))
    return dict_output

  
#keeps mapping brand names to normalized name until all normalized
def mappingsomedicts(original_dict, mapping_dict):
    mapped_dict = {}
    for i in original_dict.keys():
        if i in mapping_dict.keys():
            mapped_dict[i] = mapping_dict[i]
                                                  
    y = list()
    for i in mapped_dict.keys():
         if mapped_dict[i] != original_dict[i]:
                y.append(i)
    if len(y) > 0 :
        mappingsomedicts(mapped_dict, mapping_dict)
    return mapped_dict


#df_input, brands and counts as string
def fuzziest_of_matches(df_input, brands, counts, threshold):
    
    df_input = cleaning(df_input, brands)

    
    vectorizer = TfidfVectorizer(min_df=1, analyzer="char", ngram_range=(3,3)).fit(df_input.clean_brand)
    tfidf = vectorizer.transform(df_input.clean_brand)
    
    matches = cosine_sim_matrix(tfidf, tfidf.transpose(), 500, threshold)  #top 500 matches over 0.8 similarity if they exist (so all)

    ## calculate similarity as dataframe
    df_sim = get_matches_df(matches, df_input.index, top=matches.nnz)
    
    
    ## create field holding index value
    df_input['ind_str'] = df_input.index.values
    df_input['ind_str'] = df_input.ind_str.apply(str)
    
    
    
    #groupby left side
    df_sim.left_side = df_sim.left_side.apply(str)
    df_sim.right_side = df_sim.right_side.apply(str)
    df_sim = df_sim.groupby('left_side')['right_side'].agg(','.join).reset_index()


    dict_output = highest_count(df_sim, df_input, counts)
    ind_dict = df_input.ind_str.to_dict()
    
    ind_dict_int = {int(k):int(v) for k,v in ind_dict.items()}
    dict_output_int = {int(k):int(v) for k,v in dict_output.items()}
    
    
    output = mappingsomedicts(ind_dict_int, dict_output_int)
    
    
    changes = {}
    for i in output:
        if i != output[i]:
            changes[i] = output[i]
            
    dict_final = df_input.set_index(df_input.ind_str)[brands].to_dict()
    dict_final_int = {int(k):v for k,v in dict_final.items()}
    
    
    for i in changes:
        dict_final_int[i] = dict_final_int[changes[i]]
    
    
    
    df_input['final_output'] = dict_final_int.values()
    
    df_done = df_input[[brands, 'final_output', counts]]
    final_groupings = df_done.groupby('final_output')[counts].sum()
    df_output = pd.DataFrame(data = final_groupings)
    df_output = df_output.sort_values(by=[counts], ascending = False)

    df_output['final_output'] = df_output.index.values
    
    df_changes = df_done.loc[~(df_done['final_output'] == df_done[brands])]
    
    
    return (df_input,df_output,df_changes,df_done)
    




