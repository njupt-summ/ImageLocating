from data_reader import text_read
from gensim.models import Word2Vec
import os


dir = 'datas/texts'
# model = Word2Vec('LineSentence(inp)', size=100, window=5, min_count=3, sg=1)
# model.save('models/word2vec/word2vec_mail')

# text_path_list = list()
# for line in open("datas/trains"):
#     text_path = os.path.join(dir,line.strip())
#     text_path_list.append(text_path)
text_path_list = ['datas/texts/e49e172d3ac302b2da907c87b9406c8503e2d4a5']
for path in text_path_list:
    text = text_read(path)
model = Word2Vec(text, size=100, window=5, min_count=1, sg=1)
model.save('models/word2vec/word2vec_mail')