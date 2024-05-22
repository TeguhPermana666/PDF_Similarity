import math
class TfIdf:
    def __init__(self):
        self.corpus = []
        self.word_idf = {}
        self.vocab = set()

    def fit(self, documents):
        self.corpus = documents

        # Calculate IDF for each word in the vocabulary
        doc_count = len(self.corpus)
        for doc in self.corpus:
            unique_words = set(doc.split())
            self.vocab.update(unique_words)
            for word in unique_words:
                self.word_idf[word] = self.word_idf.get(word, 0) + 1

        for word, count in self.word_idf.items():
            self.word_idf[word] = math.log(doc_count / count)

    def transform(self, document):
        # Calculate TF-IDF for each word in the document
        tfidf_vector = {}
        total_words = len(document.split())
        for word in document.split():
            if word in self.vocab:
                tf = document.count(word) / total_words
                tfidf_vector[word] = tf * self.word_idf[word]
        return tfidf_vector