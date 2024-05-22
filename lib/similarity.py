class Similarity:
    def sqrt(value):
        return value ** 0.5
    
    def dot_product(vector1, vector2):
        return sum(vector1[key] * vector2.get(key, 0) for key in vector1)

    def magnitude(vector):
        return Similarity.sqrt(sum(value ** 2 for value in vector.values()))
    
    def cosine_similarity(tfidf_matrix):
        num_documents = len(tfidf_matrix)
        similarity_matrix = []

        for i in range(num_documents):
            similarities = []
            for j in range(num_documents):
                if i == j:
                    similarities.append(1.0)  # Similarity with itself is 1
                else:
                    dot_prod = Similarity.dot_product(tfidf_matrix[i], tfidf_matrix[j])
                    mag1 = Similarity.magnitude(tfidf_matrix[i])
                    mag2 = Similarity.magnitude(tfidf_matrix[j])
                    similarity = dot_prod / (mag1 * mag2)
                    similarities.append(similarity)
            similarity_matrix.append(similarities)

        return similarity_matrix