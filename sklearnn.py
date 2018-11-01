from sklearn.decomposition import NMF, LatentDirichletAllocation as LDA, TruncatedSVD
from sklearn.feature_extraction.text import CountVectorizer as CV
from nltk.corpus import brown

n_topics = 10
data = []

for file in brown.fileids():
    document = ' '.join(brown.words(file))
    data.append(document)

vectorizer = CV(min_df=5, max_df=0.9, stop_words='english',
                             lowercase=True, token_pattern='[a-zA-Z\-]{3,}')
vectorized_data = vectorizer.fit_transform(data)
# print(vectorized_data[:5])

lda_model = LDA(n_topics=n_topics, max_iter=10, learning_method='online')
lda_data = lda_model.fit_transform(vectorized_data)
# print(lda_data.shape)

nmf_model = NMF(n_components=n_topics)
nmf_data = nmf_model.fit_transform(vectorized_data)

lsi_model = TruncatedSVD(n_components=n_topics)
lsi_data = lsi_model.fit_transform(vectorized_data)

print(lda_data[0])
print(nmf_data[0])
print(lsi_data[0])


def print_topics(model, vectorizer, top_n=10):
    for idx, topic in enumerate(model.components_):
        print("Topic %d:" % (idx))
        print([(vectorizer.get_feature_names()[i], topic[i])
               for i in topic.argsort()[:-top_n - 1:-1]])


print("LDA Model:")
print_topics(lda_model, vectorizer)
print("=" * 20)

print("NMF Model:")
print_topics(nmf_model, vectorizer)
print("=" * 20)

print("LSI Model:")
print_topics(lsi_model, vectorizer)
print("=" * 20)

text = "The economy is working better than ever"
x = lsi_model.transform(vectorizer.transform([text]))[0]
print(x)

x = lda_model.transform(vectorizer.transform([text]))[0]
print(x)

from sklearn.metrics.pairwise import euclidean_distances


def most_similar(x, Z, top_n=5):
    dists = euclidean_distances(x.reshape(1, -1), Z)
    pairs = enumerate(dists[0])
    most_similar = sorted(pairs, key=lambda item: item[1])[:top_n]
    return most_similar


similarities = most_similar(x, nmf_data)
document_id, similarity = similarities[0]
print(data[document_id][:1000])
