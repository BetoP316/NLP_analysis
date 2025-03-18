#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep 14 18:35:36 2024
@TITLE: AUX_FUNCTION
@author: beto
"""

def update_lexicon_key(lexicon_dict,new_key,old_key):
    if old_key in lexicon_dict:
        lexicon_dict[new_key] = lexicon_dict[old_key]
        del lexicon_dict[old_key]
        print(f"Key '{old_key}' has been changed to '{new_key}'.")
    else:
        print(f"Key '{old_key}' not found in the lexicon.")



def convert_column_to_list(df, column_name):
    return df[column_name].dropna().tolist()

