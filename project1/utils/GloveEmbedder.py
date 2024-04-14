from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd
import numpy as np
import multiprocessing as mp
import nltk
import joblib

class GloveEmbedder(TransformerMixin, BaseEstimator):
    def __init__ (self, embedding_dict_path, embedding_dim, n_jobs=1):
        self.embeddings_dict = joblib.load(embedding_dict_path)
        self.embedding_dim = embedding_dim
        self.n_jobs = n_jobs

    def fit (self, X, y=None):
        return self

    def transform (self, X, y=None):
        X_copy = X.copy()

        partitions = 1
        cores = mp.cpu_count()
        data = None
        if self.n_jobs == 0:
            data = X_copy.apply(self._embed_document)
        else:
            if self.n_jobs <= -1:
                partitions = cores
            else:
                partitions = min(self.n_jobs, cores)

            data_split = np.array_split(X_copy, partitions)
            pool = mp.Pool(partitions)
            data = pd.concat(pool.map(self._embed_partition, data_split))
            pool.close()
            pool.join()

        return np.stack(data.to_numpy())

    def _embed_partition(self, part):
        return part.apply(self._embed_document)
    
    def _embed_document (self, doc):
        embedding = np.zeros(self.embedding_dim, dtype="float32")
        words = nltk.word_tokenize(doc)
        for word in words:
            if word not in self.embeddings_dict:
                continue
            word_embedding = self.embeddings_dict[word]
            embedding += word_embedding
        
        return embedding/len(words)
