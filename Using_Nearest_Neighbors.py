# -*- coding: utf-8 -*-
"""
Created on Tue Jul 30 14:51:39 2019

@author: Tadhg Lynch
"""

import pandas as pd

from Fuzzy_nearest_neighbors import mapping

#################################################

# Parameters:
in_file_train = "C:/Users/Henry Lynch/Normalized_POS_done.csv"
train_column = "final_output"
in_file_test = "Y:/ProjectManagement/Data_Science/Brand_match/UMI_brands.csv"
test_column = "CURRENT_BRAND"
dist_threshold = 0.2

#################################################

def main():
    
    df_train = pd.read_csv(in_file_train)
    df_test = pd.read_csv(in_file_test, encoding='ISO-8859-1')
    df_nn_path_new, df_nn_final_new, df_nn_changes_new = mapping(df_train, train_column, df_test, test_column, dist_threshold)
    
    df_nn_path_new.to_excel(r"C:\Users\Henry Lynch\NN_UMI_new.xlsx", index=None, header = True)
    df_nn_final_new.to_excel(r"C:\Users\Henry Lynch\NN_UMI_final_new.xlsx", index=None, header = True)
    df_nn_changes_new.to_excel(r"C:\Users\Henry Lynch\NN_UMI_changes_new.xlsx", index=None, header = True)
    
    return print("done")
    
if __name__ == "__main__":
    main()
    
    
    
    
    
    
    
    
    
    
    
    
    