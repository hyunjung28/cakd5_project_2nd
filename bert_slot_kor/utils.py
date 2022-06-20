# -*- coding: utf-8 -*-
import pandas as pd
from itertools import chain
import os

def flatten(y):
    return list(chain.from_iterable(y))

class Reader:
    
    def __init__(self):
        pass
    
    def read(dataset_folder_path):
        text_arr = []
        tags_arr = []
        
        ########################### TODO ###############################
        # seq.in과 seq.out을 읽어들여서 리스트로 만들기
        # (텍스트 파일의 한 라인이 리스트의 한 요소가 되도록)
        with open(f'{dataset_folder_path}/seq.in','r') as f:
            text_arr = f.readlines()
        with open(f'{dataset_folder_path}/seq.out','r') as f:
            tags_arr = f.readlines()
        ################################################################

        assert len(text_arr) == len(tags_arr)
        return text_arr, tags_arr
