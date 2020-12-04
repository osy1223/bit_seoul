# 실습 ( embedding 빼고, lstm과 conv1d로)

from tensorflow.keras.preprocessing.text import Tokenizer
import numpy as np

docs = ['너무 재밌어요',
        '참 최고에요',
        '참 잘 만든 영화에요',
        '추천하고 싶은 영화입니다',
        '한 번 더 보고 싶네요',
        '글쎄요',
        '별로에요',
        '생각보다 지루해요',
        '연기가 어색해요',
        '재미없어요',
        '너무 재미없다',
        '참 재밌네요'
]

# 긍정=1, 부정=0
labels = np.array([1,1,1,1,1,0,0,0,0,0,0,1])

# docs = x , labels = y

token = Tokenizer()
token.fit_on_texts(docs)
print(token.word_index)
'''
{'참': 1, '너무': 2, '재밌어요': 3, '최고에요': 4, '잘': 5, '만든': 6, '영화에요': 7, '추천하고': 8, '싶은': 9, '영화입니다': 10, '한': 11, '번': 12, '더': 13, '보고': 14, '싶네요': 15, '글쎄요': 16, '별로에요': 17, '생각보다': 18, '지루해요': 19, '연기가': 20, '어색해요': 21, '재미없어요': 22, '재미없다': 23, '재밌네요': 24}  
'''

x = token.texts_to_sequences(docs)
print(x)
'''
[[2, 3], [1, 4], [1, 5, 6, 7], [8, 9, 10], [11, 12, 13, 14, 15], [16], [17], [18, 19], [20, 21], [22], [2, 23], [1, 24]]
'''

# 0으로 (빈 자리) 앞쪽 시퀀스 채우겠다
from tensorflow.keras.preprocessing.sequence import pad_sequences
pad_x = pad_sequences(x, padding='pre') #뒤:post 
# 앞쪽으로 채우는 이유?
# 시리얼 타입이니깐 LSTM 같은 걸 사용하다보면, 뒤에다 넣으면 0으로 수렴
print(pad_x)
# [[ 0  0  0  2  3]
print(pad_x.shape) #(12, 5)

word_size = len(token.word_index)+1 #+1은 0
print('전체 토큰 사이즈:', word_size) #전체 토큰 사이즈: 25


# reshape 2차원 -> 3차원
pad_x = pad_x.reshape(12,5,1)

# 2. 모델
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Embedding, LSTM, Conv1D

model = Sequential()
model.add(Conv1D(32, 2, input_shape=(5,1), padding='same'))
model.add(LSTM(32))
model.add(Dense(1, activation='sigmoid')) #0또는 1이므로 sigmoid

model.summary()

'''
Embedding에 바로 Dense 붙이면 가능하지만,
input_length=3 주거나 input_length 값 안주면, Embedding에 Flatten 붙이고 Dense 하면 Error
 ValueError: Input 0 of layer dense is incompatible with the layer: expected axis -1 of input shape to have value 30 but received input with shape [None, 50]  
 ValueError: The last dimension of the inputs to `Dense` should be defined. Found `None`.
'''

'''

input_length = 컬럼의 개수 (달라도 돌아가지만, 맞춰주는게 좋다)
WARNING:tensorflow:Model was constructed with shape (None, 6) for input Tensor("embedding_input:0", shape=(None, 6), dtype=float32), but it was called on an input with incompatible shape (None, 5).
WARNING:tensorflow:Model was constructed with shape (None, 3) for input Tensor("embedding_input:0", shape=(None, 3), dtype=float32), but it was called on an input with incompatible shape (None, 5).
acc: 0.9166666865348816
acc: 1.0
-> 데이터가 적어서 acc 신뢰 NoNoNo

(Embedding(26,10)
acc: 0.9166666865348816

(Embedding(24,10) -> ERROR!

단어사전의 개수는, 더 크게는 가능하나, 더 작게는 불가능 !!
'''

# input_length 명시 차이

# 3. 컴파일 , 훈련
model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['acc']
)

model.fit(pad_x, labels, epochs=30)

# 4. 평가
acc = model.evaluate(pad_x, labels)[1]
print('acc:',acc)

# acc: 1.0
# acc: [0.6386476755142212, 0.8333333134651184]