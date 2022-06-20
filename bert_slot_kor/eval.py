# -*- coding: utf-8 -*-

import os
import pickle
import argparse

import tensorflow as tf
from sklearn import metrics

from utils import Reader
from to_array.bert_to_array import BERTToArray
from to_array.tags_to_array import TagsToArray
from models.bert_slot_model import BertSlotModel
from utils import flatten
import numpy as np

if __name__ == "__main__":
    # Reads command-line parameters
    parser = argparse.ArgumentParser("Evaluating the BERT NLU model")
    parser.add_argument("--model", "-m",
                        help="Path to BERT NLU model",
                        type=str,
                        required=True)
    parser.add_argument("--data", "-d",
                        help="Path to data",
                        type=str,
                        required=True)
    
    args = parser.parse_args()
    load_folder_path = args.model
    data_folder_path = args.data
    
    # this line is to disable gpu
    os.environ["CUDA_VISIBLE_DEVICES"]="-1"
    
    config = tf.ConfigProto(intra_op_parallelism_threads=8, 
                            inter_op_parallelism_threads=0,
                            allow_soft_placement=True,
                            device_count = {"CPU": 8})
    sess = tf.Session(config=config)
    
    
    #################################### TODO 경로 고치기 ##################
    bert_model_hub_path = "/content/drive/MyDrive/Colab_Notebooks/2nd_project/dataset/model"
    ########################################################################
    
    vocab_file = os.path.join(bert_model_hub_path,
                              "assets/vocab.korean.rawtext.list")
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
    
    #################################### TODO #############################
    # test set 데이터 불러오기
    print("reading test set")
    bert_vocab_path = os.path.join(bert_model_hub_path,
                                   "assets/vocab.korean.rawtext.list")
    text_arr, tags_arr = Reader.read(data_folder_path)
    
    print("text_arr[0:2] :", text_arr[0:2])
    print("tags_arr[0:2] :", tags_arr[0:2])
    
    bert_to_array = BERTToArray(bert_vocab_path) 
    # bert_to_array MUST NOT tokenize input !!!
    
    input_ids, input_mask, segment_ids = bert_to_array.transform(text_arr)
    
    # tags_to_array = TagsToArray()
    # tags_to_array.fit(tags_arr)
    data_tags_arr = tags_arr
    # data_tags_arr = tags_to_array.transform(tags_arr, input_ids)
    
    print("tags :", data_tags_arr[0:2])
    
    print("input shape :", input_ids.shape, input_ids[0:2])
    print("t_input_mask :", input_mask.shape, input_mask[0:2])
    print("t_segment_ids :", segment_ids.shape, segment_ids[0:2])
    print("data_tags_arr :", np.array(data_tags_arr).shape, data_tags_arr[0:2])
    ########################################################################
    
    def get_results(input_ids, input_mask, segment_ids, tags_arr, tags_to_array):
        inferred_tags, slots_score = model.predict_slots([input_ids,
                                                          input_mask,
                                                          segment_ids],
                                                         tags_to_array)
        
        gold_tags = [x.split() for x in tags_arr]
        # x.split()을 x.tostring().split()으로 변환

        f1_score = metrics.f1_score(flatten(gold_tags), flatten(inferred_tags),
                                    average="micro")

        tag_incorrect = ""
        for i, sent in enumerate(input_ids):
            if inferred_tags[i] != gold_tags[i]:
                tokens = bert_to_array.tokenizer.convert_ids_to_tokens(input_ids[i])
                tag_incorrect += "sent {}\n".format(tokens)
                tag_incorrect += ("pred: {}\n".format(inferred_tags[i]))
                tag_incorrect += ("score: {}\n".format(slots_score[i]))
                tag_incorrect += ("ansr: {}\n\n".format(gold_tags[i]))

        return f1_score, tag_incorrect

    f1_score, tag_incorrect = get_results(input_ids, input_mask,
                                          segment_ids, data_tags_arr,
                                          tags_to_array)
    
    # 테스트 결과를 모델 디렉토리의 하위 test_results에 저장해 준다.
    result_path = os.path.join(load_folder_path, "test_results")
    
    if not os.path.isdir(result_path):
        os.mkdir(result_path)
    
    with open(os.path.join(result_path, "tag_incorrect.txt"), "w") as f:
        f.write(tag_incorrect)
    
    with open(os.path.join(result_path, "test_total.txt"), "w") as f:
        f.write("Slot f1_score = {}\n".format(f1_score))
    
    tf.compat.v1.reset_default_graph()
    print("complete")
