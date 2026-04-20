## How to use

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
