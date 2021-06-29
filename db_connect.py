# %%
import pandas as pd
import firebase_admin
from firebase_admin import credentials
from firebase_admin import db
# %% # Firebase database 인증 및 앱 초기화

cred = credentials.Certificate(
    './andr/firebaseCert.json')

firebase_admin.initialize_app(
    cred, {'databaseURL': 'firebaseDBURL'})

# %% 기본 위치 지정
ref = db.reference('target')
# %%

target = ref.get()
# %%

target.keys()
target_df = pd.DataFrame(target).T.fillna('')
# %%
# %%
tmp = []
for key, s in target_df.subject.iteritems():
    tmp.append(pd.Series(data={'lCategory': s[0]['lCategory'],
                               'mCategory': s[0]['mCategory'],
                               'sCategory': s[0]['sCategory']},
                         name=key))
    print(key, s[0]['lCategory'] + ' ' + s[0]
          ['mCategory'] + ' ' + s[0]['sCategory'])
subject_df = pd.DataFrame(tmp)
subject_df.to_csv('./data/target-subject.csv')
# %%
