## How to use
from .load_mnist_embeddings import load_or_create_mnist_resnet18_embeddings
from .load_ag_news_embeddings import load_or_create_ag_news_text_embeddings
from .load_banking77_embeddings import load_or_create_banking77_text_embeddings
from .load_cifar100_embeddings import load_or_create_cifar100_resnet18_embeddings
from .load_cifar10_embeddings import load_or_create_cifar10_resnet18_embeddings
from .load_dbpedia14_embeddings import load_or_create_dbpedia14_text_embeddings
from .load_emotion_embeddings import load_or_create_emotion_text_embeddings
from .load_yelp_review_full_embeddings import (
    load_or_create_yelp_review_full_text_embeddings,
)
from .load_imdb_embeddings import load_or_create_imdb_text_embeddings
from .load_trec_embeddings import load_or_create_trec_text_embeddings

# For vision:

"""
from quantbayes.ball_dp.experiments.load_mnist_embeddings import load_or_create_mnist_resnet18_embeddings

X_train, y_train, X_test, y_test = load_or_create_mnist_resnet18_embeddings(
    data_root="./data",
    require_jax_gpu=False,   # set True if you want to force JAX GPU
)
print(X_train.shape, X_train.dtype)
print(y_train.shape, y_train.dtype)"""

# For text:

"""
from quantbayes.ball_dp.experiments.load_ag_news_embeddings import load_or_create_ag_news_text_embeddings

X_train, y_train, X_test, y_test = load_or_create_ag_news_text_embeddings(
    output_root="./data",
    require_jax_gpu=False,
)
print(X_train.shape, X_train.dtype)"""
