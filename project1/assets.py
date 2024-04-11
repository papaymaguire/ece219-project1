import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.decomposition import TruncatedSVD
import random
from dagster import (
    AssetIn,
    AssetOut,
    asset,
    multi_asset,
    file_relative_path
)
from dagstermill import define_dagstermill_asset
from project1.configurations import Project1Config
from project1.utils.TextPreprocessor import TextPreprocessor

np.random.seed(42)
random.seed(42)


@asset(
    group_name="project1"
)
def dataset(config: Project1Config):
    """Pull data from remote CSV into DataFrame"""

    df = pd.read_csv(config.dataset_csv_path)
    return df


question1_notebook = define_dagstermill_asset(
    name="question1",
    group_name="project1",
    notebook_path=file_relative_path(__file__, "./notebooks/question1.ipynb"),
    ins={"dataset": AssetIn("dataset")},
    description="""Notebook used to answer question 1"""
)


@multi_asset(
    group_name="project1",
    outs={
        "train": AssetOut(),
        "test": AssetOut()
    }
)
def train_test_set(dataset):
    """Split dataset into training and test set"""

    train, test = train_test_split(dataset, test_size=0.2)
    return train, test


question2_notebook = define_dagstermill_asset(
    name="question2",
    group_name="project1",
    notebook_path=file_relative_path(__file__, "./notebooks/question2.ipynb"),
    ins={
        "train": AssetIn("train"),
        "test": AssetIn("test")
    },
    description="""Notebook used to answer question 2"""
)


question3_notebook = define_dagstermill_asset(
    name="question3",
    group_name="project1",
    notebook_path=file_relative_path(__file__, "./notebooks/question3.ipynb"),
    ins={
        "train": AssetIn("train"),
        "test": AssetIn("test")
    },
    description="""Notebook used to answer question 3"""
)


@asset(
    group_name="project1"
)
def vocab_model(train):
    """Fit a vocab to the train set"""

    vocab_pipe = Pipeline(steps=[
        ("preprocess", TextPreprocessor("lemm", n_jobs=-1)),
        ("count", CountVectorizer(stop_words="english", min_df=3))
    ])
    vocab_pipe.fit(train['full_text'])
    return vocab_pipe['count']


@asset(
    group_name="project1"
)
def tfidf_model(train, vocab_model):
    """Extract the TF-IDF features from the training dataset using the learned vocab"""
    
    tfidf_pipe = Pipeline(steps=[
        ("preprocess", TextPreprocessor("lemm", n_jobs=-1)),
        ("count", vocab_model),
        ("tfidf", TfidfTransformer())
    ])
    tfidf_pipe.fit(train['full_text'])
    return tfidf_pipe['tfidf']


@asset(
    group_name="project1"
)
def train_feature_matrix(train, vocab_model, tfidf_model):
    """Use the TF-IDF model to transform the train """
    tfidf_pipe = Pipeline(steps=[
        ("preprocess", TextPreprocessor("lemm", n_jobs=-1)),
        ("count", vocab_model),
        ("tfidf", tfidf_model)
    ])
    train_features = tfidf_pipe.transform(train['full_text'])
    return train_features


question4_notebook = define_dagstermill_asset(
    name="question4",
    group_name="project1",
    notebook_path=file_relative_path(__file__, "./notebooks/question4.ipynb"),
    ins={
        "train_feature_matrix": AssetIn("train_feature_matrix")
    },
    description="""Notebook used to answer question 4"""
)


@asset(
    group_name="project1"
)
def dim_reducer_model(train_feature_matrix):
    """Train the dimension reducer found most effective in Notebook 4"""
    svd = TruncatedSVD(n_components=50, random_state=42)
    svd.fit(train_feature_matrix)
    return svd


@asset(
    group_name="project1"
)
def dim_reduced_train_features(train, vocab_model, tfidf_model, dim_reducer_model):
    """Calculate the features of the train set and reduce the dimension"""
    pipeline = Pipeline(steps=[
        ("preprocess", TextPreprocessor("lemm", n_jobs=-1)),
        ("count", vocab_model),
        ("tfidf", tfidf_model),
        ("reduce", dim_reducer_model)
    ])
    return pipeline.transform(train['full_text'])


@asset(
    group_name="project1"
)
def train_root_labels(train):
    """Return the root labels on the training data"""
    return train['root_label']


@asset(
    group_name="project1"
)
def dim_reduced_test_features(test, vocab_model, tfidf_model, dim_reducer_model):
    """Calculate the features of the test set and reduce the dimension"""
    pipeline = Pipeline(steps=[
        ("preprocess", TextPreprocessor("lemm", n_jobs=-1)),
        ("count", vocab_model),
        ("tfidf", tfidf_model),
        ("reduce", dim_reducer_model)
    ])
    return pipeline.transform(test['full_text'])


@asset(
    group_name="project1"
)
def test_root_labels(test):
    """Return the root labels on the testing data"""
    return test['root_label']


question5_notebook = define_dagstermill_asset(
    name="question5",
    group_name="project1",
    notebook_path=file_relative_path(__file__, "./notebooks/question5.ipynb"),
    ins={
        "dim_reduced_train_features": AssetIn("dim_reduced_train_features"),
        "train_root_labels": AssetIn("train_root_labels"),
        "dim_reduced_test_features": AssetIn("dim_reduced_test_features"),
        "test_root_labels": AssetIn("test_root_labels"),
    },
    description="""Notebook used to answer question 5"""
)


question6_notebook = define_dagstermill_asset(
    name="question6",
    group_name="project1",
    notebook_path=file_relative_path(__file__, "./notebooks/question6.ipynb"),
    ins={
        "dim_reduced_train_features": AssetIn("dim_reduced_train_features"),
        "train_root_labels": AssetIn("train_root_labels"),
        "dim_reduced_test_features": AssetIn("dim_reduced_test_features"),
        "test_root_labels": AssetIn("test_root_labels"),
    },
    description="""Notebook used to answer question 6"""
)


question7_notebook = define_dagstermill_asset(
    name="question7",
    group_name="project1",
    notebook_path=file_relative_path(__file__, "./notebooks/question7.ipynb"),
    ins={
        "dim_reduced_train_features": AssetIn("dim_reduced_train_features"),
        "train_root_labels": AssetIn("train_root_labels"),
        "dim_reduced_test_features": AssetIn("dim_reduced_test_features"),
        "test_root_labels": AssetIn("test_root_labels"),
    },
    description="""Notebook used to answer question 7"""
)


question8_notebook = define_dagstermill_asset(
    name="question8",
    group_name="project1",
    notebook_path=file_relative_path(__file__, "./notebooks/question8.ipynb"),
    ins={
        "train": AssetIn("train"),
        "test": AssetIn("test")
    },
    description="""Notebook used to answer question 8"""
)