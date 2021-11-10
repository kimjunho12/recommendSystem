# %%
from os import name
from firebase_admin import db
from firebase_admin import credentials
import firebase_admin
import numpy as np
from numpy.lib.function_base import vectorize
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers.experimental import preprocessing
from tensorflow.keras.utils import to_categorical
# %%
subject_df = pd.read_csv('./data/target-subject.csv', index_col=0)
subject_text_df = pd.read_csv('./data/target-subject.csv', index_col=0)
subject_text_df['target'] = subject_text_df.index

label = subject_df.index
subject_df['target'] = label

for c in subject_df.columns:
    subject_df[c] = pd.Categorical(subject_df[c])
    subject_df[c] = subject_df[c].cat.codes

target = subject_df['target']

target_tt = to_categorical(target)

# %%

inputs = keras.Input(shape=(None,), dtype=tf.string, name='Input_Layer')
indexer = preprocessing.StringLookup(
    output_mode='binary', name='One_Hot_Layer')
indexer.adapt(subject_text_df)
v_input = indexer(inputs)
clayer = layers.Dense(len(target)*40, activation='relu', name='Classify_Layer')
c_data = clayer(v_input)
classer = layers.Dense(len(target), activation='softmax', name='Output_Layer')
outputs = classer(c_data)
model = keras.Model(inputs=inputs, outputs=outputs)

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy']
              )
# %%
model.fit(subject_text_df, target_tt, epochs=20)

# %%
# 결과값 이름으로 도출 [:n] - n 개 만큼
def return_target(intclasses):
    reco_target = {}
    for intclass in intclasses:
        print('-'*10)
        for x in intclass[::-1][:10]:
            print(target.index[x])
            reco_target[target.index[x]] = True
    return reco_target


def predict(liked_target):
    return np.argsort(model.predict([list(liked_target)]))
# %%
return_target(np.argsort(model.predict([['K-pop'], ['하키']])))
return_target(np.argsort(model.predict([['K-pop', '서울과학기술대학교']])))

# %%

# # %%
# model.save('./model/keras_model')
# converter = tf.lite.TFLiteConverter.from_saved_model('./model/keras_model/')
# converter.optimizations = [tf.lite.Optimize.DEFAULT]
# converter.inference_input_type = tf.string
# converter.inference_input_type = tf.float32
# converter.target_spec.supported_types = [tf.string, tf.resource]
# converter.target_spec.supported_ops = []
# converter.target_spec.supported_ops.append(tf.lite.OpsSet.TFLITE_BUILTINS)
# converter.target_spec.supported_ops.append(tf.lite.OpsSet.SELECT_TF_OPS)
# tflitemodel = converter.convert()

# with open('./model/contents_base_model.tflite', 'wb') as f:
#   f.write(tflitemodel)

# %% 사용자 설문 정보 받아오기
# %% # Firebase database 인증 및 앱 초기화

cred = credentials.Certificate(
    './andr/supportapp-f34a1-firebase-adminsdk-gzyie-78c718f2be.json')

firebase_admin.initialize_app(
    cred, {'databaseURL': 'https://supportapp-f34a1-default-rtdb.firebaseio.com'})
# %% 사용자 찜목록 기반 추천 후 DB 저장
ref = db.reference('Users')

snapshot = ref.order_by_key().get()
for key, val in snapshot.items():
    try :
        ref.child(key).child('reco').set(return_target(predict(val['like'].keys())))
    except Exception as E :
        print(E)

# # %%
# user = ref.get()
# user_df = pd.DataFrame(user).T.fillna('')
# # %% [n] 번째 유저의 선호도 조사 결과 추출
# n = 0
# user_df.iloc[n]['like'].keys()
# # %% [n] 번째 유저의 맞춤 추천 결과
# return_target(np.argsort(model.predict(
#     [list(user_df.iloc[n]['like'].keys())])))

# # %%
# return_target(predict(user_df.iloc[n]['like'].keys()))
# %% 결과에서 기존에 좋아하는 사람으로 되어있을때 처리 필요