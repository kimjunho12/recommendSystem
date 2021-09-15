# %%
import pandas as pd
import firebase_admin
from firebase_admin import credentials
from firebase_admin import db
# %% # Firebase database 인증 및 앱 초기화

cred = credentials.Certificate(
    './andr/supportapp-f34a1-firebase-adminsdk-gzyie-78c718f2be.json')

firebase_admin.initialize_app(
    cred, {'databaseURL': 'https://supportapp-f34a1-default-rtdb.firebaseio.com'})

# %% 기본 위치 지정
ref = db.reference('target')
target = ref.get()
target.keys()
target_df = pd.DataFrame(target).T.fillna('')
# %%
tmp = []
for key, s in target_df.subject.iteritems():
    tmp.append(pd.Series(data={'lCategory': s['lCategory'],
                               'mCategory': s['mCategory'],
                               'sCategory': s['sCategory']},
                         name=key))
    print(key, s['lCategory'] + ' ' + s['mCategory'] + ' ' + s['sCategory'])
subject_df = pd.DataFrame(tmp)
subject_df.to_csv('./data/target-subject.csv')
# %%
