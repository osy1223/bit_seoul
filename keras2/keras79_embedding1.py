from tensorflow.keras.preprocessing.text import Tokenizer

text = '나는 진짜 맛있는 밥을 진짜 먹었다.'

token = Tokenizer()
token.fit_on_texts([text])

print(token.word_index)
print(len(token.word_index))

x = token.texts_to_sequences([text])
print(x)

'''
{'진짜': 1, '나는': 2, '맛있는': 3, '밥을': 4, '먹었다': 5}
[[2, 1, 3, 4, 1, 5]]
'''

from tensorflow.keras.utils import to_categorical
word_size = len(token.word_index)
x = to_categorical(x, num_classes=word_size+1)
'''
    categorical[np.arange(n), y] = 1
IndexError: index 5 is out of bounds for axis 1 with size 5
0부터 세니깐 +1
'''
print(x)