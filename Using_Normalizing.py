# -*- coding: utf-8 -*-
"""
Created on Thu Jul 11 15:47:43 2019

@author: Tadhg Lynch
"""

import pandas as pd

from Fuzzy_Normalizing import fuzziest_of_matches

#################################################

# Parameters:
in_file = "C:/Users/Henry Lynch/Documents/Fuzzy Matching Project/Copy of UMI Brand Consolidation_08082019.csv"
brand_column = "updated_brand"
count_column = "CountDistinct_UMI_ID"
threshold = 0.85

#################################################

def main():
    
    df_in_file = pd.read_csv(in_file, encoding='ISO-8859-1')
    df_new, df_output_new, df_changes_new, df_done_new = fuzziest_of_matches(df_in_file, brand_column, count_column, threshold)
    
    df_new.to_csv(r"C:\Users\Henry Lynch\Normalized_UMI_rob.csv", index=None, header = True)
    df_output_new.to_csv(r"C:\Users\Henry Lynch\Normalized_UMI_rob_done.csv", index=None, header = True)
    df_changes_new.to_csv(r"C:\Users\Henry Lynch\Normalized_UMI_rob_changes.csv", index=None, header = True)
    df_done_new.to_csv(r"C:\Users\Henry Lynch\Normalized_UMI_rob_final_output.csv", index=None, header = True)
    
    print("done")
    
if __name__ == "__main__":
    main()
    
    

    