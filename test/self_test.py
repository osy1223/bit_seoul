from tensorflow.keras.preprocessing.text import Tokenizer, one_hot
from tensorflow.keras.utils import to_categorical

text = "나는 자연어 처리를 배운다"

t=Tokenizer()
t.fit_on_texts([text])
print(t.word_index)

# {'나는': 1, '자연어': 2, '처리를': 3, '배운다': 4}   

sub_text="자연어 처리룰 배운다 나는"
encoded= t.texts_to_sequences([sub_text])[0]
print(encoded)
#[2, 4, 1]

one_hot = to_categorical(encoded)
print(one_hot)
'''
[[0. 0. 1. 0. 0.]
 [0. 0. 0. 0. 1.]
 [0. 1. 0. 0. 0.]]
'''
