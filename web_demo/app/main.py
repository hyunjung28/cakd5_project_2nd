# -*- coding: utf-8 -*-
from flask import Flask, render_template, request
from flask_ngrok import run_with_ngrok
import os
import re
import sys
import pickle
import tensorflow as tf
sys.path.append('/content/drive/MyDrive/cakd5/project2/2팀_자연어2/final/web_demo/app')
from recommend import recommend
import random

# enable GPU
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
config = tf.compat.v1.ConfigProto(
    intra_op_parallelism_threads=1,
    inter_op_parallelism_threads=1,
    allow_soft_placement=True,
    device_count={"GPU": 0},
)
sess = tf.compat.v1.Session(config=config)
graph = tf.compat.v1.get_default_graph()

# bert_slot_kor 경로 설정
sys.path.append(os.path.join(os.getcwd(), "/content/drive/MyDrive/cakd5/project2/2팀_자연어2/final/bert_slot_kor"))
from to_array.bert_to_array import BERTToArray
from models.bert_slot_model import BertSlotModel
from to_array.tokenizationK import FullTokenizer

# pretrained model path
bert_model_hub_path = '/content/drive/MyDrive/cakd5/project2/2팀_자연어2/final/dataset/model'

# fine-tuned model path
load_folder_path = "/content/drive/MyDrive/cakd5/project2/2팀_자연어2/final/dataset/saved_model"

# tokenizer vocab file path
vocab_file = os.path.join(bert_model_hub_path, "assets/vocab.korean.rawtext.list")
bert_to_array = BERTToArray(vocab_file)

# 슬롯태깅 모델과 토크나이저 불러오기
tags_to_array_path = os.path.join(load_folder_path, "tags_to_array.pkl")
with open(tags_to_array_path, "rb") as handle:
    tags_to_array = pickle.load(handle)
    slots_num = len(tags_to_array.label_encoder.classes_)

model = BertSlotModel.load(load_folder_path, sess)

tokenizer = FullTokenizer(vocab_file=vocab_file)


# 슬롯 리스트
sweetness =['안달','달지않은','달지않고','드라이','안단','안달고','달지않고','달지않은','달달한','달달하고','달달하지만','달짝지근','단','달콤한','스위트','많이단',
'달지않은거','달지않은거로','달지않은걸로','달지않으며','달지않게','달달하지않으면','달달하지않았으면','달달하지않게','달달하지않은거',
'달달하지않은걸로','달달하지않은거로','드라이하게','드라이한거','드라이한걸로','드라이한거로','드라이했으면','안달달하게','안달달했으면','안달달하고',
'안달달한거','안달달한걸로','안달달한거로','안달달한','안단거','안달게','안달며','안단걸로','안단거로','안단걸로','달달','달달한거',
'달달한걸로','달달한거로','달달했으면','달달하며','달달하게','달짝지근하게','달짝지근하며','달짝지근한거','달짝지근한거로','달짝지근한걸로',
'달짝지근했으면','달짝지근하고','달짝지근','덜단','덜단거','단거','단걸로','단거로','단게','달게','달며','달고','달콤하며','달콤하고','달콤한거',
'달콤한걸로','달콤하게','달콤한거로','달콤','달콤했으면','스위트하고','스위트하며','스위트한거','스위트하게','스위트한걸로','스위트한거로','스위트한',
'스위트했으면','스위트로']
body = ['가벼운','라이트','가볍','상쾌한','청량한','가볍지만','가볍지않은','미디엄','진한','진하고','무거운','무겁고','헤비','풀','풀바디','끈적한',
'무겁지만','가벼운걸로','가벼운거','가벼우며','가벼운거로','가볍게','가볍','가볍게','가벼웠으면','라이트한거','라이트한걸로','라이트한거로','라이트하며',
'라이트하게','라이트하고','라이트한','라이트했으면','라이트로','무겁지않았으면','무겁지않은걸로','무겁지않은거로','무겁지않은','무겁지않고','무겁지않은거',
'무겁지않으며','무겁지않게','안무거운','안무거운거','안무겁게','안무거우며','안무거운걸로','안무거운거로','안무거웠으면','안무겁고','상쾌하며',
'상쾌하게','상쾌하고','상쾌한걸로','상쾌한거','상쾌한거로','상쾌','상쾌했으면','청량하며','청량하게','청량하고','청량한걸로','청량한거','청량한거로','청량',
'청량했으면','끈적하게','끈적하고','끈적하며','끈적했으면','끈적한거','끈적한걸로','끈적한거로','끈적','가볍지않고','가볍지않은거','가볍지않은걸로',
'가볍지않은거로','가볍지않게','가볍지않으며','가볍지않았으면','진한거','진한걸로','진한거로','진했으면','진하게','진하며','무거운거','무거운걸로','무거운거로',
'무거웠으면','무거우며','무겁게','헤비한거','헤비한걸로','헤비한거로','헤비했으면','헤비하게','헤비하고','헤비한','헤비하며','헤비로','풀바디한','풀바디로',
'풀바디였으면','풀바디한걸로','풀바디한거','풀바디한거로','풀바디하게','풀바디하며','풀바디하고','풀로','풀이였으면','풀인걸로','풀인거','풀인거로','풀하고',
'묵직한', '묵직', '묵직하고']
sourness = ['안신','안시고','시지않은','시지않고','새콤한','상큼한','시지만','조금시큼한','시고','신','시큼한','시큼하고','안신거','안신걸로','안신거로',
'안시었으면','안셨으면','안시며','안시게','시지않게','시지않은걸로','시지않으며','시지않은거','시지않았으면','시지않은거로','상큼하고','상큼한거',
'상큼한거로','상큼한걸로','상큼하며','상큼하게','상큼','상큼했으면','새콤하고','새콤한거','새콤한거로','새콤한걸로','새콤하며','새콤하게','새콤','세콤했으면',
'조금시큼한거로','조금시큼한걸로','조금시큼한거','조금시큼하게','조금시큼하며','조금시큼','조금시큼하고','신거','신걸로','신거로','시고','셨으면','시게',
'시었으면','시며','시게','시큼한거','시큼한걸로','시큼한거로','시큼했으면','시큼','시큼하고','조금신']
wine_type = ['레드', '화이트', '스파클링', '샴페인', '로제','레드로', '화이트로', '스파클링으로', '샴페인으로','로제로']
price = ['1만대','1만원대','1만정도','1만원정도','1만원쯤','1만원?','1만?','1만이하','1만원이하','1만원이하','1만이하','2만대','2만원대',
'2만정도','2만원정도','2만원쯤','2만원?','2만?','2만이하','2만원이하','2만원이하','2만이하','3만대','3만원대','3만정도','3만원정도','3만원쯤',
'3만원?','3만?','3만이하','3만원이하','3만원이하','3만이하','4만대','4만원대','4만정도','4만원정도','4만원쯤','4만원?','4만?','4만이하',
'4만원이하','4만원이하','4만이하','5만대','5만원대','5만정도','5만원정도','5만원쯤','5만원?','5만?','5만이하','5만원이하','5만원이하',
'5만이하','6만대','6만원대','6만정도','6만원정도','6만원쯤','6만원?','6만?','6만이하','6만원이하','6만원이하','6만이하','7만대','7만원대',
'7만정도','7만원정도','7만원쯤','7만원?','7만?','7만이하','7만원이하','7만원이하','7만이하','8만대','8만원대','8만정도','8만원정도','8만원쯤',
'8만원?','8만?','8만이하','8만원이하','8만원이하','8만이하','9만대','9만원대','9만정도','9만원정도','9만원쯤','9만원?','9만?','9만이하',
'9만원이하','9만원이하','9만이하','10만대','10만원대','10만정도','10만원정도','10만원쯤','10만원?','10만?','10만이하','10만원이하',
'10만원이하','10만이하','20만대','20만원대','20만정도','20만원정도','20만원쯤','20만원?','20만?','20만이하','20만원이하','20만원이하',
'20만이하','30만대','30만원대','30만정도','30만원정도','30만원쯤','30만원?','30만?','30만이하','30만원이하','30만원이하','30만이하','40만대',
'40만원대','40만정도','40만원정도','40만원쯤','40만원?','40만?','40만이하','40만원이하','40만원이하','40만이하','50만대','50만원대','50만정도',
'50만원정도','50만원쯤','50만원?','50만?','50만이하','50만원이하','50만원이하','50만이하','60만대','60만원대','60만정도','60만원정도',
'60만원쯤','60만원?','60만?','60만이하','60만원이하','60만원이하','60만이하','70만대','70만원대','70만정도','70만원정도','70만원쯤','70만원?',
'70만?','70만이하','70만원이하','70만원이하','70만이하','80만대','80만원대','80만정도','80만원정도','80만원쯤','80만원?','80만?','80만이하',
'80만원이하','80만원이하','80만이하','90만대','90만원대','90만정도','90만원정도','90만원쯤','90만원?','90만?','90만이하','90만원이하',
'90만원이하','90만이하','100만대','100만원대','100만정도','100만원정도','100만원쯤','100만원?','100만?','100만이하','100만원이하',
'100만원이하','100만이하','1만대의', '1만원대의', '1만정도의', '1만원쯤의', '2만대의', '2만원대의', '2만정도의', '2만원쯤의', '3만대의', 
'3만원대의', '3만정도의', '3만원쯤의', '4만대의', '4만원대의', '4만정도의', '4만원쯤의', '5만대의', '5만원대의', '5만정도의', '5만원쯤의', 
'6만대의', '6만원대의', '6만정도의', '6만원쯤의', '7만대의', '7만원대의', '7만정도의', '7만원쯤의', '8만대의', '8만원대의', '8만정도의', 
'8만원쯤의', '9만대의', '9만원대의', '9만정도의', '9만원쯤의', '10만대의', '10만원대의', '10만정도의', '10만원쯤의', '20만대의', 
'20만원대의', '20만정도의', '20만원쯤의', '30만대의', '30만원대의', '30만정도의', '30만원쯤의', '40만대의', '40만원대의', '40만정도의',
 '40만원쯤의', '50만대의', '50만원대의', '50만정도의', '50만원쯤의', '60만대의', '60만원대의', '60만정도의', '60만원쯤의', 
 '70만대의', '70만원대의', '70만정도의', '70만원쯤의', '80만대의', '80만원대의', '80만정도의', '80만원쯤의', '90만대의',
 '90만원대의', '90만정도의', '90만원쯤의', '100만대의', '100만원대의', '100만정도의', '100만원쯤의']
 ##############################################################

# 슬롯 사전
dic = {
    "당도" : sweetness,
    "바디감" : body,
    "산미": sourness,
    "종류" : wine_type,
    "금액" : price
}


# 명령어 설정 ( 챗봇 사용자가 문장 앞에 !를 붙이면 명령어로 인식 )
cmds = {
    "명령어" : ["명령어",'설명' ,"당도", "바디감", "산미", "종류", "금액"],
    '설명' : '''이 와인 챗봇은 초심자를 위한 와인 챗봇입니다.
    종류, 당도, 바디감, 산미, 금액을 입력하시면 그에 맞는 와인을 추천해드려요.
    어떤것이 있는지 모르시겠다면 !명령어를 입력하시면 어떤 명령어가 있는지 보실수 있어요.
    !당도 같이 입력하시면 그것에 대한 설명을 알려드릴게요.''',
    "당도" : '''당도는 와인의 단 정도를 뜻해요. 단맛이 없는것을 원하신다면 안단거, 달지않은 와인
    같이 입력하시고 어느정도 단것을 원하신다면 달달한거,달콤한거 같이 입력해주세요.''',
    "바디감" : '''바디감은 와인을 머금었을 때 입안에서 느껴지는 '묵직함' 을 의미하는데요.
    바디감이 진한수록 도수도 더 높아지는 경향이 있어요. 크림같이 진한느낌을 원하신다면 
    진한거, 무거운거같이 입력하시고 전지분유처럼 가벼운 느낌이 좋으시다면 가벼운와인, 라이트한 와인
    이라고 입력해주세요.''',
    "산미" : '''산미는 식초와 같이 톡 쏘는 시큼한 맛을 뜻해요.상큼,새콤한 느낌부터 톡쏘는 시큼한 느낌까지
    다양해요. 시지않은것을 원하시면 시지않은거나 안신거처럼 입력하시고 어느정도 새콤한것은 상큼한 와인, 새콤한와인처럼
    입력하시고 좀 시큼한 와인을 원하시면 신거, 시큼한 와인처럼 입력해주세요. ''',
    "종류" : '''저희가 추천해드리는 와인의 종류는 레드, 화이트, 샴페인, 스파클링, 로제와인의 5가지 종류가 있어요.''',
    "금액" : '''저희는 1~10만원대는 만원단위로 추천을 해드리고 그 이상부터는 10만원단위로 추천해드려요.
    만약 X만원 이하 같이 금액을 말씀해주시면 그 금액 이하의 와인을 추천해드릴게요.''',
    "member" : '''권혁종, 이현정, 이동연, 김민성, 오주완, 박종석''',
    "권혁종" : '''팀장<br>git : https://github.com/gitHek''',
    "이현정" : '''팀원<br>git : https://github.com/hyunjung28''',
    "이동연" : '''팀원<br>git : https://github.com/movingkite''',
    "김민성" : '''팀원<br>git : https://github.com/nycticebus0915''',
    "오주완" : '''팀원<br>git : https://github.com/joowaun93''' ,
    "박종석" : '''팀원<br>git : https://github.com/blazestar95''' 
}


# 슬롯이라고 인식한 토큰을 slot_text에 저장하기
def catch_slot(i, inferred_tags, text_arr, slot_text):
    if not inferred_tags[0][i] == "O":
        word_piece = re.sub("_", "", text_arr[i])
        slot_text[inferred_tags[0][i]] += word_piece


# 슬롯이 다 채워지면 챗봇 유저에게 확인 메세지 보내기
def check_order_msg(app):
    order = []
    for slot, option in app.slot_dict.items():
        order.append(f"{slot}: {option}")
    # br 태그는 html에서 줄바꿈을 의미함
    order = "<br />\n".join(set(order))

    message = f"""
        {order} <br />
        위의 와인으로 추천해드릴까요? (예 or 아니오)
        """
    return message


# 슬롯 초기화 함수
def init_app(app):
    app.slot_dict = {
        "당도": "",
        "바디감": "",
        "산미": "",
        "종류": "",
        "금액": ""
    }
    app.confirm = 0


app = Flask(__name__)

# colaboratory에서 실행 시
run_with_ngrok(app)

app.static_folder = 'static'

@app.route("/")
def home():
# 사용자가 입력한 슬롯을 저장할 슬롯 사전
    app.slot_dict = {
        "당도": "",
        "바디감": "",
        "산미": "",
        "종류": "",
        "금액": ""
    }
    # 슬롯으로 인식할 점수 설정하기
    app.score_limit = 0.7
    # 대화에 필요한 변수 설정
    app.confirm = 0
    return render_template("index.html")
 

# 챗봇 사용자가 메세지를 입력했을 때 실행   
@app.route("/get")
def get_bot_response():
    # 사용자가 입력한 문장
    userText = request.args.get('msg').strip() 
    # 명령어 인식 - 사용자가 입력한 문장이 느낌표(!)로 시작할 때
    if userText[0] == "!":
        try:
          if userText == '!명령어':
            li = cmds[userText[1:]]
            message = "<br />\n".join(li)
          else:
            li = cmds[userText[1:]]
            message = ''.join(li)
        except:
            message = "입력한 명령어가 존재하지 않습니다."
        return message


    # 사용자가 입력한 문장을 토큰화
    text_arr = tokenizer.tokenize(userText)
    input_ids, input_mask, segment_ids = bert_to_array.transform([" ".join(text_arr)])

    # 훈련한 슬롯태깅 모델을 사용하여 슬롯 예측
    with graph.as_default():
        with sess.as_default():
            inferred_tags, slots_score = model.predict_slots(
                [input_ids, input_mask, segment_ids], tags_to_array
            )

    # inference 결과로 나온 토큰을 저장하는 사전
    slot_text = {k: "" for k in app.slot_dict}
    # 슬롯태깅 실시
    for i in range(0, len(inferred_tags[0])):
        if slots_score[0][i] >= app.score_limit: # 설정한 점수보다 슬롯 점수가 높을 시
            catch_slot(i, inferred_tags, text_arr, slot_text) # 슬롯을 저장하는 함수 실행
        else:
            print("슬롯 인식 중 에러가 발생했습니다.")
    # 슬롯 사전에 있는 단어와 일치하는지 검증 후 슬롯 사전에 최종으로 저장
    for slot in slot_text:
        if slot_text[slot] in dic[slot]:
            app.slot_dict[slot] = slot_text[slot]

    # 채워지지 않은 슬롯들을 슬롯 이름으로 리스트화
    empty_slot = [slot for slot in app.slot_dict if app.slot_dict[slot] == "" ]

    # 상관으로 시작하는 단어를 입력 시 상관없음으로 빈 슬롯을 채움
    if userText.startswith('상관'):
      for i in empty_slot:
        app.slot_dict[i] = '상관없음'
        print(app.slot_dict)
      empty_slot = None

# 추출된 슬롯 정보를 가지고 추천 와인까지 출력(recommend 함수 적용)
    if empty_slot:
      if '금액' in empty_slot:
        message = ", ".join(empty_slot) + "이 아직 선택되지 않았습니다.<br>이대로 추천을 원하신다면 상관없음을 입력해주세요."
      elif ('산미' not in empty_slot)&('종류' not in empty_slot)&('바디감' in empty_slot):
        message = ", ".join(empty_slot) + "이 아직 선택되지 않았습니다.<br>이대로 추천을 원하신다면 상관없음을 입력해주세요."
      else:
        message = ", ".join(empty_slot) + "가 아직 선택되지 않았습니다.<br>이대로 추천을 원하신다면 상관없음을 입력해주세요."
    elif app.confirm == 0:
        message = check_order_msg(app)
        app.confirm += 1
    else:
        if userText.startswith("예"):
            wine_list = recommend(app.slot_dict)
            try:
              if len(wine_list) > 10:
                rec_num = random.choice(range(10))
              else:
                rec_num = random.choice(range(len(wine_list)))
              name = wine_list.iloc[rec_num]['이름']
              img = wine_list.iloc[rec_num]['이미지주소']
              cate = wine_list.iloc[rec_num]['종류']
              page = wine_list.iloc[rec_num]['주소']
              price = wine_list.iloc[rec_num]['금액']
              sweet_wine =  wine_list.iloc[rec_num]['당도']
              sour_wine = wine_list.iloc[rec_num]['산미']
              body_wine = wine_list.iloc[rec_num]['바디감']
              pairing = wine_list.iloc[rec_num]['페어링']
              text = f'''추천드릴 와인은
              {name} 입니다.<br>
              이 와인은 {cate}와인으로<br>
              {sour_wine}의 산미,{body_wine}의 바디감,{sweet_wine}의 당도를 가지고 있고<br>
              가격은 {price}원입니다.<br>
              자세한 정보확인하시거나 구매를 위해서는<br>
              이동하기로 방문해보세요.'''
              message = text+'%$,'+img+'%$,'+page
              return message
            except:
              message = '해당하는 와인이 없습니다. 다시 입력해주세요'
              return message
            finally:
              init_app(app)
              return message
        elif userText.startswith("아니오"):
            message = "다시 주문해주세요."
            init_app(app)
    return message