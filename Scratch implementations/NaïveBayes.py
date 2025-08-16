import math
from collections import defaultdict


class NaiveBayesClassifier:
    def __init__(self, laplace=1):
        self.laplace = laplace

        self.classes = {}
        self.word_count = defaultdict(lambda: defaultdict(int))
        self.total_words = defaultdict(int)
        self.cls_doc_count = defaultdict(int)
        self.vocab = set()

    def fit(self, data):
        for text, label in data:
            self.cls_doc_count[label] += 1

            words = text.split()

            self.total_words[label] += len(words)

            for word in words:
                self.word_count[label][word] += 1
                self.vocab.add(word)
        total_docs = sum(self.cls_doc_count.values())
        for labels in self.cls_doc_count:
            self.classes[labels] = self.cls_doc_count[labels] / total_docs

    def predict(self, text):
        scores = {}
        for label in self.cls_doc_count:
            prob_class = math.log(self.classes[label])
            for word in text.split():
                word_count = self.word_count[label][word]
                prob_word = (word_count + self.laplace) / (self.total_words[label] + self.laplace * len(self.vocab))
                prob_class += math.log(prob_word)
            scores[label] = prob_class

        return max(scores, key=scores.get)


train_data = [
    ("cheap offer today", "spam"),
    ("limited deal buy now", "spam"),
    ("how are you today", "ham"),
    ("let's catch up tomorrow", "ham")
]
bayes = NaiveBayesClassifier()
bayes.fit(train_data)
print(bayes.predict("le"))
