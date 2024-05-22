import pdfplumber
from lib.preprocess import preprocess_text
from lib.TfIdf import TfIdf
from lib.TfIdf_Vectorized import TfidfVectorizer
from lib.similarity import Similarity
    
def compare_pdfs(file1, file2):
    with pdfplumber.open(file1) as pdf1, pdfplumber.open(file2) as pdf2:
        # Extract text from all pages
        text1 = '\n'.join(page.extract_text() for page in pdf1.pages)
        text2 = "\n".join(page.extract_text() for page in pdf2.pages)
        
        # Preprocess the text
        text1 = preprocess_text(text1)
        text2 = preprocess_text(text2)

        #TF-IDF => just for me, fow know what the sentence has been extarcted
        # tfidf = TfIdf()
        # tfidf.fit([text1, text2])  # Pass a list of documents
        # text1_tfidf = tfidf.transform(text1)
        
        # Tf-IDF Vectorized calculation
        tfidf_matriks = TfidfVectorizer().fit_transform([text1, text2])
        # Similarity calculation of TF-IDF
        similarity_matriks = Similarity.cosine_similarity(tfidf_matriks)
        similarity = similarity_matriks[0][1]
        if similarity > 0.8:
            print("Similar")
        else:
            print("Not Similar")

compare_pdfs(r"doc\contoh-surat-dinas.pdf", r"doc\AseanEng.pdf")
