import re

docs = [
    "machine learning is fun",
    "learning from data is powerful",
    "deep learning powers modern AI"
]


def my_count_vectorizer(corpus):
    vocab = {}

    string = ""
    for item in corpus:
        string += item + " "

    tokens = re.findall(r"\b\w+\b", string.lower())
    index = 0
    for item in sorted(tokens):
        if item not in vocab:
            vocab[item] = index
            index += 1

    vectors = []
    for doc in corpus:
        vector = [0] * len(vocab)
        for word in doc.split():
            vector[vocab[word.lower()]] += 1
        vectors.append(vector)

    return vocab, vectors


print(my_count_vectorizer(docs))
