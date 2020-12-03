from tensorflow.keras.preprocessing.text import Tokenizer, one_hot
from tensorflow.keras.utils import to_categorical

text = "나는 자연어 처리를 배운다"

t=Tokenizer()
t.fit_on_texts([text])
# print(t.word_index)

# {'나는': 1, '자연어': 2, '처리를': 3, '배운다': 4}   

sub_text="자연어 처리룰 배운다 나는"
encoded= t.texts_to_sequences([sub_text])[0]
# print(encoded)
#[2, 4, 1]

one_hot = to_categorical(encoded)
# print(one_hot)
'''
[[0. 0. 1. 0. 0.]
 [0. 0. 0. 0. 1.]
 [0. 1. 0. 0. 0.]]
'''

from tensorflow.keras.preprocessing.text import text_to_word_sequence
# print(text_to_word_sequence("Don't be fooled by the dark sounding name, Mr.Jone's Orphanage is as cheery as cheery goes for a pastry shop"))
'''
["don't", 'be', 'fooled', 'by', 'the', 'dark', 'sounding', 'name', 'mr', "jone's", 'orphanage', 'is', 'as', 'cheery', 'as', 'cheery', 'goes', 'for', 'a', 'pastry', 
'shop']
'''

# 엔엘티케이(NLTK)는 자연어 처리를 위한 파이썬 패키지
# 표준 토큰화
from nltk.tokenize import TreebankWordTokenizer
tokenizer = TreebankWordTokenizer()
text = "Starting a home-based restaurant may be an ideal. it doesn't have a food chain or restaurant of their own."
# print(tokenizer.tokenize(text))
'''
['Starting', 'a', 'home-based', 'restaurant', 'may', 'be', 'an', 'ideal.', 'it', 'does', "n't", 'have', 'a', 'food', 'chain', 'or', 'restaurant', 'of', 'their', 'own', '.']
'''

# 문장 토큰화(Sentence Tokenization)
from nltk.tokenize import sent_tokenize
text = "His barber kept his word. But keeping such a huge secret to himself was driving him crazy. Finally, the barber went up a mountain and almost to the edge of a cliff. He dug a hole in the midst of some reeds. He looked about, to mae sure no one was near."
# print(sent_tokenize(text))
'''
['His barber kept his word.', 'But keeping such a huge secret to himself was driving him crazy.', 'Finally, the barber went up a mountain and almost to the edge of 
a cliff.', 'He dug a hole in the midst of some reeds.', 'He looked about, to mae sure no one was near.']
'''

text="I am actively looking for Ph.D. students. and you are a Ph.D student."
# print(sent_tokenize(text))
'''
['I am actively looking for Ph.D. students.', 'and you are a Ph.D student.']   
'''

# KSS(Korean Sentence Splitter)
# import kss
# text='딥 러닝 자연어 처리가 재미있기는 합니다. 그런데 문제는 영어보다 한국어로 할 때 너무 어려워요. 농담아니에요. 이제 해보면 알걸요?'
# print(kss.split_sentences(text))

from nltk.tokenize import word_tokenize
text = 'I am actively looking for Ph.D. students. and you are a Ph.D. student.'
# print(word_tokenize(text))
'''
['I', 'am', 'actively', 'looking', 'for', 'Ph.D.', 'students', '.', 'and', 'you', 'are', 'a', 'Ph.D.', 'student', '.']
'''
# 영어 코퍼스에 품사 태깅
from nltk.tag import pos_tag
x = word_tokenize(text)
pos_tag(x)
[('I', 'PRP'), ('am', 'VBP'), ('actively', 'RB'), ('looking', 'VBG'), ('for', 'IN'), ('Ph.D.', 'NNP'), ('students', 'NNS'), ('.', '.'), ('and', 'CC'), ('you', 'PRP'), ('are', 'VBP'), ('a', 'DT'), ('Ph.D.', 'NNP'), ('student', 'NN'), ('.', '.')]
'''
영어 문장에 대해서 토큰화를 수행하고, 이어서 품사 태깅을 수행하였습니다. 
Penn Treebank POG Tags에서 PRP는 인칭 대명사, VBP는 동사, RB는 부사, VBG는 현재부사, IN은 전치사, NNP는 고유 명사, NNS는 복수형 명사, CC는 접속사, DT는 관사를 의미
'''

# 길이가 1~2인 단어들을 정규 표현식을 이용하여 삭제
import re
text = "I was wondering if anyone out there could enlighten me on this car."
shortword = re.compile(r'\W*\b\w{1,2}\b')
# print(shortword.sub('', text))
# was wondering anyone out there could enlighten this car.

# 표제어 추출
# WordNetLemmatizer는 입력으로 단어의 품사를 알려줄 수 있다
from nltk.stem import WordNetLemmatizer
n = WordNetLemmatizer()
words=['policy', 'doing', 'organization', 'have', 'going', 'love', 'lives', 'fly', 'dies', 'watched', 'has', 'starting']
# print([n.lemmatize(w) for w in words])
'''
['policy', 'doing', 'organization', 'have', 'going', 'love', 'life', 'fly', 'dy', 'watched', 'ha', 'starting']
'''
# print(n.lemmatize('dies', 'v')) #die 
# print(n.lemmatize('watched', 'v')) #watch
# print(n.lemmatize('has', 'v')) #have

# 어간 추출(Stemming)
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
s = PorterStemmer()
text = "This was not the map we found in Billy Bones's chest, but an accurate copy, complete in all things--names and heights and soundings--with the single exception of the red crosses and the written notes."
words = word_tokenize(text)
# print(words)
'''
['This', 'was', 'not', 'the', 'map', 'we', 'found', 'in', 'Billy', 'Bones', "'s", 'chest', ',', 'but', 'an', 'accurate', 'copy', ',', 'complete', 'in', 'all', 'things', '--', 'names', 'and', 'heights', 'and', 'soundings', '--', 'with', 'the', 'single', 'exception', 'of', 'the', 'red', 'crosses', 'and', 'the', 'written', 'notes', '.']
'''

# print([s.stem(w) for w in words])
'''
['thi', 'wa', 'not', 'the', 'map', 'we', 'found', 'in', 'billi', 'bone', "'s", 'chest', ',', 'but', 'an', 'accur', 'copi', ',', 'complet', 'in', 'all', 'thing', '--', 'name', 'and', 'height', 'and', 'sound', '--', 'with', 'the', 'singl', 'except', 'of', 'the', 'red', 'cross', 'and', 'the', 'written', 'note', '.']
'''
'''
※ Porter 알고리즘의 상세 규칙은 마틴 포터의 홈페이지에서 확인
포터 알고리즘의 어간 추출은 이러한 규칙들을 가집니다.
ALIZE → AL
ANCE → 제거
ICAL → IC
'''

words=['formalize', 'allowance', 'electricical']
# print([s.stem(w) for w in words])
# ['formal', 'allow', 'electric']

words=['policy', 'doing', 'organization', 'have', 'going', 'love', 'lives', 'fly', 'dies', 'watched', 'has', 'starting']
# print([s.stem(w) for w in words])
# ['polici', 'do', 'organ', 'have', 'go', 'love', 'live', 'fli', 'die', 'watch', 'ha', 'start']

from nltk.stem import LancasterStemmer
l = LancasterStemmer()
words = ['polici', 'do', 'organ', 'have', 'go', 'love', 'live', 'fli', 'die', 'watch', 'ha', 'start']
# print([l.stem(w) for w in words])
# ['polic', 'do', 'org', 'hav', 'go', 'lov', 'liv', 'fli', 'die', 'watch', 'ha', 'start']

'''
Stemming
am → am
the going → the go
having → hav

Lemmatization
am → be
the going → the going
having → have

알고리즘 때문에 자르는 방식이 다름
'''
# 불용어 확인
# stopwords.words("english")는 NLTK가 정의한 영어 불용어 리스트를 리턴
from nltk.corpus import stopwords
# print(stopwords.words('english')[:10])
# ['i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you're"]

# 불용어 제거
example = "Family is not an important thing. It's everything."
stop_words = set(stopwords.words('english'))

word_tokens = word_tokenize(example)

result = []
for w in word_tokens:
    if w not in stop_words:
        result.append(w)

# print(word_tokens)
# print(result)

'''
['Family', 'is', 'not', 'an', 'important', 'thing', '.', 'It', "'s", 'everything', '.']
['Family', 'important', 'thing', '.', 'It', "'s", 'everything', '.']
'''

# 기호
import re
r = re.compile('a.c')
# print(r.search('kkk'))
# print(r.search('abc'))
'''
None
<re.Match object; span=(0, 3), match='abc'>
'''

r = re.compile('ab?c')
# print(r.search('abbc'))
# print(r.search('abc'))
# print(r.search('ac'))
'''
None
<re.Match object; span=(0, 3), match='abc'>
<re.Match object; span=(0, 2), match='ac'>
'''

r = re.compile('ab*c')
# print(r.search('a'))
# print(r.search('ac'))
# print(r.search('abc'))
# print(r.search('aabbc'))
# print(r.search('abbbbc'))
# print(r.search('aabbc'))
'''
None
<re.Match object; span=(0, 2), match='ac'>
<re.Match object; span=(0, 3), match='abc'>
<re.Match object; span=(1, 5), match='abbc'>
<re.Match object; span=(0, 6), match='abbbbc'>
'''

r = re.compile('ab+c')
# print(r.search('ac'))
# print(r.search('abc'))
# print(r.search('abbbbc'))
'''
None
<re.Match object; span=(0, 3), match='abc'>
<re.Match object; span=(0, 6), match='abbbbc'>
'''

r = re.compile('^a')
# print(r.search('bbc'))
# print(r.search('ab'))
'''
None
<re.Match object; span=(0, 1), match='a'>
'''

r = re.compile('ab{2}c')
# print(r.search('ac'))
# print(r.search('abc'))
# print(r.search('abbc'))
'''
None
None
<re.Match object; span=(0, 4), match='abbc'>
'''

r=re.compile("ab{2,8}c")
# print(r.search('ac'))
# print(r.search('abc'))
# print(r.search('abbc'))
# print(r.search('abbbbbbbbc'))
'''
None
None
<re.Match object; span=(0, 4), match='abbc'>
<re.Match object; span=(0, 10), match='abbbbbbbbc'>
'''

r=re.compile("a{2,}bc")
# print(r.search('abbbbbbbbc'))
# print(r.search("aabc"))
'''
None
<re.Match object; span=(0, 4), match='aabc'>
'''

r=re.compile("[abc]")
# print(r.search('abbbbbbbbc'))
# print(r.search("aabc"))
# print(r.search("x"))
'''
<re.Match object; span=(0, 1), match='a'>
<re.Match object; span=(0, 1), match='a'>
None
'''

r=re.compile("[^abc]")
# print(r.search("d"))
# print(r.search("1"))
# print(r.search("abc"))
'''
<re.Match object; span=(0, 1), match='d'>
<re.Match object; span=(0, 1), match='1'>
None
'''

# re.match() 와 re.search()의 차이
# search()가 정규 표현식 전체에 대해서 문자열이 매치하는지를 본다면, 
# match()는 문자열의 첫 부분부터 정규 표현식과 매치하는지를 확인
r=re.compile("ab.")
# print(r.search("kkkabc")  )
# print(r.match('kkkabc'))
'''
<re.Match object; span=(3, 6), match='abc'>
None
'''

# re.split()
text1 = 'apple banana'
text2 = "사과 딸기 수박 메론 바나나"
# print(re.split(',', text1))
# print(re.split(' ', text2))
'''
['apple banana']
['사과', '딸기', '수박', '메론', '바나나']
'''

text3="""사과
딸기
수박
메론
바나나"""
# print(re.split("\n",text3))
# ['사과', '딸기', '수박', '메론', '바나나']

text4="사과+딸기+수박+메론+바나나"
print(re.split("\+",text4))
# ['사과', '딸기', '수박', '메론', '바나나']

