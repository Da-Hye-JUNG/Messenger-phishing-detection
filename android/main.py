import firebase_admin
from firebase_admin import credentials
from firebase_admin import db
from flask import Flask, make_response, jsonify, request
import os

import numpy as np
import pandas as pd
import csv
import re

# 모델
from konlpy.tag import Okt
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.layers import Embedding, Dense, LSTM
from tensorflow.keras.models import Sequential
from tensorflow.keras.models import load_model
from keras.preprocessing import sequence
from keras.utils import np_utils

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


def listener(event):
    data = event.data
    print(data)
    chat_list = [*data]
    chat_list.remove('predict_result')
    print(chat_list)
    return data, chat_list


# 파이어베이스 연동
cred = credentials.Certificate("cred파일.json")
firebase_admin.initialize_app(cred, {'databaseURL': 'databaseURL 입력'})
dir = db.reference("/")
data = dir.get()


# 데이터 전처리
stop_words = set(
    ['은', '는', '이', '가', '하', '아', '것', '들', '의', '있', '되', '수', '보', '주', '등', '한', '안', '없', '때', '지', '두'])


# 불용어 처리, 어간 추출
def preprocessing(text, okt, remove_stopwords=False, stop_words=[]):
    text1 = re.sub("[^가-힣ㄱ-ㅎㅏ-ㅣ\\s]", "", text)
    word = okt.morphs(text1, stem=True)
    if remove_stopwords:
        word = [token for token in word if not token in stop_words]

    return word


# 토크나이저 피팅에 필요한 텍스트 다운로드
clean_text = []
with open('../data/clean_text.csv', 'r', encoding='utf-8') as f:
    rdr = csv.reader(f)
    for line in rdr:
        clean_text.append(line)


# 줄바꿈 기호 제거
def delete_enter(text):
    text3 = ' '.join(text.split('\n'))
    return text3


# 한 문장으로 병합
def make_data(data):
    new = {}
    for name in data.keys():
        if type(data[name]) == dict:
            full_text = list(data[name].values())
            new_text = [delete_enter(x) if x.find('\n') != -1 else x for x in full_text]
            new[name] = ' '.join(new_text)
        else:
            new[name] = delete_enter(data[name])
    return new


# 불용어, 형태소, 정수 인코딩
def make_test(text):
    okt = Okt()
    test = preprocessing(text, okt, remove_stopwords=True, stop_words=stop_words)
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(clean_text)
    test_s = tokenizer.texts_to_sequences(test)
    test_s = [x[0] for x in test_s if x != []]
    test_input = pad_sequences([test_s], maxlen=100, padding='post')
    return test_input


# 피싱 확률 출력 함수
def pred_prob2(text, k):
    preds = []
    test = make_test(text)
    for i in range(1, k + 1):
        model = load_model('../model/kfold_' + str(i) + '_model.h5')
        prediction = model.predict(test)[0][1]
        preds.append(prediction)
    pred_mean = np.mean(preds)
    return pred_mean


app = Flask(__name__)


# 메인페이지 라우팅
@app.route('/')
def hello():
    data_temp = request.get_json()
    print(data_temp)
    return jsonify({'data': data_temp})


if __name__ == "__main__":
    chat_list = [*data]
    chat_list.remove("predict_result")

    new_data = make_data(data)
    for name in chat_list:
        text = new_data[name]
        prob = pred_prob2(text, 5)
        print(name, ':', '{:.2%}'.format(prob))

        result = str(prob)
        result_dir = db.reference('predict_result')
        result_dir.update({name: result})

    app.run()
