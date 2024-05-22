import re
import math
from collections import defaultdict

class TfidfVectorizer:
    def __init__(self):
        self.documents = []
        self.vocab = set()
        self.idf = {}
        self.tf_idf_matrix = []

    def fit_transform(self, documents):
        self.documents = documents

        # Tokenize and build vocabulary
        self.vocab = self.build_vocabulary()

        # Calculate IDF
        self.calculate_idf()

        # Calculate TF-IDF matrix
        self.tf_idf_matrix = self.calculate_tf_idf_matrix()

        return self.tf_idf_matrix

    def build_vocabulary(self):
        vocab = set()
        for doc in self.documents:
            words = re.findall(r'\b\w+\b', doc.lower())
            vocab.update(words)
        return vocab

    def calculate_idf(self):
        doc_count = len(self.documents)
        for term in self.vocab:
            doc_freq = sum(1 for doc in self.documents if term in doc.lower())
            self.idf[term] = math.log(doc_count / (1 + doc_freq))

    def calculate_tf_idf_matrix(self):
        tf_idf_matrix = []
        for doc in self.documents:
            tf_vector = self.calculate_tf(doc)
            tf_idf_vector = {term: tf * self.idf[term] for term, tf in tf_vector.items()}
            tf_idf_matrix.append(tf_idf_vector)
        return tf_idf_matrix

    def calculate_tf(self, document):
        tf_vector = defaultdict(float)
        words = re.findall(r'\b\w+\b', document.lower())
        total_words = len(words)
        for word in words:
            tf_vector[word] += 1 / total_words
        return tf_vector
