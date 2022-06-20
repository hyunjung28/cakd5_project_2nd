# -*- coding: utf-8 -*-

import os
import pickle
import argparse

############################### TODO ##########################################
# 필요한 모듈 불러오기
import sys
from tokenizationK import FullTokenizer
from to_array.bert_to_array import BERTToArray
from to_array.tags_to_array import TagsToArray
from models.bert_slot_model import BertSlotModel
import re
from prepare_data import process_line
###############################################################################

import tensorflow as tf


if __name__ == "__main__":
    # Reads command-line parameters
    parser = argparse.ArgumentParser("Evaluating the BERT NLU model")
    parser.add_argument("--model", "-m",
                        help="Path to BERT NLU model",
                        type=str,
                        required=True)
    
    args = parser.parse_args()
    load_folder_path = args.model
    

    # this line is to disable gpu
    os.environ["CUDA_VISIBLE_DEVICES"]="-1"

    config = tf.ConfigProto(intra_op_parallelism_threads=1,
                            inter_op_parallelism_threads=1,
                            allow_soft_placement=True,
                            device_count = {"CPU": 1})
    sess = tf.compat.v1.Session(config=config)

    ################################ TODO 경로 고치기 ##################
    bert_model_hub_path = "/content/drive/MyDrive/Colab_Notebooks/2nd_project/dataset/model"
    ####################################################################
    
    ############################### TODO ###############################
    # 모델과 기타 필요한 것들 불러오기
    tokenizer = FullTokenizer(vocab_file="/content/drive/MyDrive/Colab_Notebooks/2nd_project/dataset/vocab.korean.rawtext.list")
    vocab_file = os.path.join(bert_model_hub_path,"assets/vocab.korean.rawtext.list")
    bert_to_array = BERTToArray(vocab_file)

    # loading models
    print("Loading models ...")
    if not os.path.exists(load_folder_path):
        print("Folder `%s` not exist" % load_folder_path)
    
    tags_to_array_path = os.path.join(load_folder_path, "tags_to_array.pkl")
    with open(tags_to_array_path, "rb") as handle:
        tags_to_array = pickle.load(handle)
        slots_num = len(tags_to_array.label_encoder.classes_)
        
    model = BertSlotModel.load(load_folder_path, sess)
    ####################################################################
    
    
    while True:
        print("\nEnter your sentence: ")
        try:
            input_text = input().strip()
        except:
            continue
            
        if input_text == "quit":
            break
        else:
            text_arr = tokenizer.tokenize(input_text)
            input_ids, input_mask, segment_ids = bert_to_array.transform([' '.join(text_arr)])
            inferred_tags, slots_score = model.predict_slots([input_ids,input_mask,segment_ids],tags_to_array)
            print(text_arr)
            print(inferred_tags)
            print(slots_score)
    ############################### TODO ###############################
    # 사용자가 입력한 한 문장을 슬롯태깅 모델에 넣어서 결과 뽑아내기
    ####################################################################
    
    tf.compat.v1.reset_default_graph()
    
