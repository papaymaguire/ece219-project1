from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd
import numpy as np
import re
import string
import multiprocessing as mp
import nltk
from nltk.stem import WordNetLemmatizer
from nltk import pos_tag
from nltk.corpus import wordnet
from nltk.stem import PorterStemmer

class TextPreprocessor(TransformerMixin, BaseEstimator):
  def __init__ (self, type="lemm", n_jobs=1):
    nltk.download("punkt", quiet=True)
    nltk.download('wordnet', quiet=True)
    nltk.download('punkt', quiet=True)
    nltk.download('averaged_perceptron_tagger', quiet=True)
    if type not in ["lemm", "stem"]:
       raise ValueError("type not supported")
    self.type = type
    self.n_jobs = n_jobs

  def fit (self, X, y=None):
    return self

  def transform (self, X, y=None):
    X_copy = X.copy()

    partitions = 1
    cores = mp.cpu_count()
    if self.n_jobs <= -1:
        partitions = cores
    elif self.n_jobs <= 0:
        return X_copy.apply(self._preprocess_text)
    else:
        partitions = min(self.n_jobs, cores)

    data_split = np.array_split(X_copy, partitions)
    pool = mp.Pool(cores)
    data = pd.concat(pool.map(self._preprocess_partition, data_split))
    pool.close()
    pool.join()

    return data
  
  def _nltk_tag_to_wordnet_tag(self, nltk_tag: str):
      if nltk_tag.startswith('J'):
          return wordnet.ADJ
      elif nltk_tag.startswith('V'):
          return wordnet.VERB
      elif nltk_tag.startswith('N'):
          return wordnet.NOUN
      elif nltk_tag.startswith('R'):
          return wordnet.ADV
      else:
          return wordnet.NOUN # use noun as a fall back

  def _preprocess_partition(self, part):
    return part.apply(self._preprocess_text)

  def _preprocess_text(self, text):
    html_clean_text = self._remove_html(text)
    lowercased_text = self._lowercase_text(html_clean_text)
    tagged_tokens = self._tag_and_tokenize(lowercased_text)
    no_punc = self._remove_punctuation(tagged_tokens)
    no_numbers = self._remove_numbers(no_punc)
    stem_or_lemm = None
    if self.type == "lemm":
        stem_or_lemm = self._lemmatize(no_numbers)
    elif self.type == "stem":
        stem_or_lemm = self._stem(no_numbers)
    return " ".join(stem_or_lemm)

  def _remove_html(self, text):
    texter = re.sub(r'^https?:\/\/.*[\r\n]*', '', text, flags=re.MULTILINE)
    texter = re.sub(r"<br />", " ", texter)
    texter = re.sub(r"&quot;", "\"",texter)
    texter = re.sub('&#39;', "\"", texter)
    texter = re.sub('\n', " ", texter)
    texter = re.sub(' u '," you ", texter)
    texter = re.sub('`',"", texter)
    texter = re.sub(' +', ' ', texter)
    texter = re.sub(r"(!)\1+", r"!", texter)
    texter = re.sub(r"(\?)\1+", r"?", texter)
    texter = re.sub('&amp;', 'and', texter)
    texter = re.sub('\r', ' ',texter)
    clean = re.compile('<.*?>')
    texter = texter.encode('ascii', 'ignore').decode('ascii')
    texter = re.sub(clean, '', texter)
    if texter == "":
      texter = ""
    return texter
  
  def _lowercase_text(self, text):
     return str.lower(text)
  
  def _tag_and_tokenize(self, text):
    tokens = []
    sentences = nltk.sent_tokenize(text=text)
    for sentence in sentences:
      words = nltk.word_tokenize(sentence)
      tagged_words = pos_tag(words)
      wordnet_tagged = [(word, self._nltk_tag_to_wordnet_tag(pos_tag)) for word, pos_tag in tagged_words]
      tokens.append(wordnet_tagged)
    return tokens

  def _remove_punctuation(self, tokens):
    return [t for t in tokens if t[0] not in string.punctuation]

  def _remove_numbers (self, tokens):
    def is_number (string):
      try:
        float(string)
        return True
      except:
        return False
    return [t for t in tokens if not is_number(t[0])]

  def _lemmatize (self, tokens):
    lemmatizer = WordNetLemmatizer()
    return [lemmatizer.lemmatize(word, pos_tag) for word, pos_tag in tokens]
  
  def _stem(self, tokens):
     stemmer = PorterStemmer()
     return [stemmer.stem(word) for word, pos_tag in tokens]