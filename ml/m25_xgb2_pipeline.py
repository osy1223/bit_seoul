parameters =[
    {'n_estimators':[100,200,300], 
    'learning_rate':[0.1,0.3, 0.001, 0.01],
    'max_depth':[4,5,6]
    },
    {'n_estimators':[90,100,110], 
    'learning_rate':[0.1,0.001,0.01],
    'max_depth':[4,5,6],
    'colsample_bytree':[0.6,0.9,1],
    },
    {'n_estimators':[90,110], 
    'learning_rate':[0.1,0.001,0.5],
    'max_depth':[4,5,6],
    'colsample_bytree':[0.6,0.9,1],
    'colsample_bylevel':[0.6,0.,0.9]
    }
]

n_jobs = -1
# 가장 중요한 건 learning_rate!

# 파일을 완성하시오!