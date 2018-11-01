from nltk.corpus import brown
import re
from gensim import models, corpora
from nltk import word_tokenize
from nltk.corpus import stopwords

data = []
pattern = r"[a-zA-Z\-]{3,}"

for file in brown.fileids():
    document = ' '.join(brown.words(file))
    data.append(document)

# print('\n'.join(data[0:5]))
n = len(data)  # there are 500 rows in brown corpus
n_topics = 10
stop_word = stopwords.words('english')


def clean_text(data):
    tokenized = word_tokenize(data.lower())
    cleaned = [t for t in tokenized if t not in stop_word and
               re.match(pattern, t)]  # take words have 3 characters at least
    return cleaned


tokenized_data = [clean_text(row) for row in data]
# print(tokenized_data[:5])

dict_ = corpora.Dictionary(tokenized_data)
corpus = [dict_.doc2bow(row) for row in tokenized_data]  # (word_id, frequency) for each word
# print(corpus[:5])

print('-'*10, "LDA Model", '-'*10)
lda = models.LdaModel(corpus=corpus, num_topics=n_topics, id2word=dict_)
for i in range(n_topics):
    print("Topic-{}, words:{}".format(i, lda.print_topic(i, 5)))

print('*'*95)
print('-'*10, "LSI Model", '-'*10)
lsi = models.LsiModel(corpus=corpus, num_topics=n_topics, id2word=dict_)
for i in range(n_topics):
    print("Topic-{}, words:{}".format(i, lsi.print_topic(i, 5)))
