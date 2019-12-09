# -*- coding: utf-8 -*-
"""
Created on Thu Jul 11 15:47:43 2019

@author: Tadhg Lynch
"""

import pandas as pd
import numpy as np
import re

from scipy.sparse import csr_matrix
import sparse_dot_topn.sparse_dot_topn as ct
from sklearn.neighbors import NearestNeighbors
from sklearn.feature_extraction.text import TfidfVectorizer

from Fuzzy_Normalizing import fuzziest_of_matches

#################################################

# Parameters:
in_file = "C:/Users/Henry Lynch/Documents/Fuzzy Matching Project/umi_brand_project_apparel.csv"
brand_column = "updated_brand"
count_column = "umi_count"
threshold = 0.85

#################################################

def main():
    
    df_infile = pd.read_csv(in_file)
    df_new, df_output, df_changes, df_done = fuzziest_of_matches(df_infile, brand_column, count_column, threshold)
    print(df_changes.head(50))
    
    
    
if __name__ == "__main__":
    main()
    
    

    