import numpy as np
import jax.numpy as jnp 
import jax 
from fake_data_generator import generate_simple_regression_data, generate_binary_classification_data, generate_multiclass_classification_data
from sklearn.model_selection import train_test_split
from utils import inference, train, visualize_regression, visualize_binary, visualize_multiclass
from sklearn.metrics import log_loss, accuracy_score, mean_absolute_error
import matplotlib.pyplot as plt
from models import regression_model, binary_model, multiclass_model



def test_binary():
    df = generate_binary_classification_data()
    X, y = df.drop(columns=["target"], axis=1), df["target"]
    X, y = jnp.array(X), jnp.array(y)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=24)
    svi, params, loss_progression = train(X_train, y_train, binary_model, hidden_dim=10, num_steps=1000, num_classes=None, track_loss=True)
    predictions = inference(svi, params, X_test, hidden_dim=10, sample_from="y", num_classes=False)
    mean_predictions = predictions.mean(axis=0)
    uncertainty = predictions.std(axis=0)
    lower_bound = mean_predictions - 1.96 * uncertainty
    upper_bound = mean_predictions + 1.96 * uncertainty
    binary_preds = (mean_predictions > 0.5).astype(int)
    accuracy = accuracy_score(np.array(y_test), np.array(binary_preds))
    print(f"Accuracy for binary: {accuracy}")
    visualize_binary(X_test, y_test, mean_predictions, uncertainty, svi, params, (0, 1), 100)
    plt.figure(figsize=(5, 5))
    plt.plot(range(1, len(loss_progression)+1), loss_progression)
    plt.xlabel("Steps")
    plt.ylabel("Loss")
    plt.title("Loss over steps")
    plt.show()

def test_multiclass():
    df = generate_multiclass_classification_data()
    X, y = df.drop(columns=["target"], axis=1), df["target"]
    X, y = jnp.array(X), jnp.array(y)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=24)
    svi, params, loss_progression = train(X_train, y_train, multiclass_model, hidden_dim=10, num_steps=1000, num_classes=3, track_loss=True)
    predictions = inference(svi, params, X_test, hidden_dim=10, sample_from="logits", num_classes=3)
    mean_predictions = predictions.mean(axis=0)
    uncertainty = predictions.std(axis=0)
    lower_bound = mean_predictions - 1.96 * uncertainty
    upper_bound = mean_predictions + 1.96 * uncertainty
    probabilities = jax.nn.softmax(mean_predictions, axis=1)
    loss = log_loss(np.array(y_test), np.array(probabilities))
    class_preds = jnp.argmax(probabilities, axis=1)
    accuracy = accuracy_score(np.array(y_test), np.array(class_preds))
    print(f"Loss: {loss}")
    print(f"Accuracy for Multiclass: {accuracy}")
    visualize_multiclass(X_test, y_test, mean_predictions, uncertainty, svi, params, 3, (0, 1), 100)
    plt.figure(figsize=(5, 5))
    plt.plot(range(1, len(loss_progression)+1), loss_progression)
    plt.xlabel("Steps")
    plt.ylabel("Loss")
    plt.title("Loss over steps")
    plt.show()

def test_regression():
    df = generate_simple_regression_data()
    X, y = df.drop(columns=["target"], axis=1), df["target"]
    X, y = jnp.array(X), jnp.array(y)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=24)
    svi, params, loss_progression = train(X_train, y_train, regression_model, hidden_dim=10, num_steps=1000, num_classes=None, track_loss=True)
    predictions = inference(svi, params, X_test, hidden_dim=10, sample_from="y", num_classes=False)
    mean_predictions = predictions.mean(axis=0)
    uncertainty = predictions.std(axis=0)
    lower_bound = mean_predictions - 1.96 * uncertainty
    upper_bound = mean_predictions + 1.96 * uncertainty
    MAE = mean_absolute_error(np.array(y_test), np.array(mean_predictions))
    print(f"MAE for regressor: {MAE}")
    visualize_regression(X_test, y_test, mean_predictions, lower_bound, upper_bound)
    plt.figure(figsize=(5, 5))
    plt.plot(range(1, len(loss_progression)+1), loss_progression)
    plt.xlabel("Steps")
    plt.ylabel("Loss")
    plt.title("Loss over steps")
    plt.show()


if __name__ == "__main__":
    print("Testing Binary")
    test_binary()
    print("Testing Regressor")
    test_regression()
    print("Testing Multiclass")
    test_multiclass()


