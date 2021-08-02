# %%
import numpy as np
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
x = indexer(inputs)
classer = layers.Dense(len(target), activation='softmax', name='Output_Layer')
outputs = classer(x)
model = keras.Model(inputs=inputs, outputs=outputs)

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy']
              )
# %%
model.fit(subject_text_df, target_tt, epochs=400)

# %%
# 결과값 이름으로 도출 [:n] - n 개 만큼
def return_target(intclasses):
    for intclass in intclasses:
        print('-'*10)
        for x in intclass[::-1][:10]:
            print(target.index[x])
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
