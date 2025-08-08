from utils import (
    load_train_csv,
    load_valid_csv,
    load_public_test_csv,
    load_train_sparse,
)
import numpy as np
import matplotlib.pyplot as plt

def sigmoid(x):
    """Apply sigmoid function."""
    return np.exp(x) / (1 + np.exp(x))


def neg_log_likelihood(data, theta, beta, alpha):
    """Compute the negative log-likelihood.

    You may optionally replace the function arguments to receive a matrix.

    :param data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param theta: Vector
    :param beta: Vector
    :param alpha: Vector
    :return: float
    """
    #####################################################################
    # TODO:                                                             #
    # Implement the function as described in the docstring.             #
    #####################################################################
    log_lklihood = 0.0
    for i in range(len(data["is_correct"])):
        u = data["user_id"][i]
        q = data["question_id"][i]
        x = alpha[q] * (theta[u] - beta[q])
        p = sigmoid(x)
        log_lklihood += data["is_correct"][i] * np.log(p) + (1 - data["is_correct"][i]) * np.log(1 - p)
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################
    return -log_lklihood


def update_parameters(data, lr, theta, beta, alpha):
    """Update theta and beta using gradient descent.

    You are using alternating gradient descent. Your update should look:
    for i in iterations ...
        theta <- new_theta
        beta <- new_beta
        alpha <- new_alpha

    You may optionally replace the function arguments to receive a matrix.

    :param data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param lr: float
    :param theta: Vector
    :param beta: Vector
    :param alpha: Vector
    :return: tuple of vectors
    """
    #####################################################################
    # TODO:                                                             #
    # Implement the function as described in the docstring.             #
    #####################################################################
    d_theta = np.zeros(theta.shape)
    d_beta = np.zeros(beta.shape)
    d_alpha = np.zeros(alpha.shape)

    for i in range(len(data["is_correct"])):
        u = data["user_id"][i]
        q = data["question_id"][i]
        x = alpha[q] * (theta[u] - beta[q])
        p = sigmoid(x)
        diff = data["is_correct"][i] - p

        d_theta[u] += alpha[q] * diff
        d_beta[q] += -alpha[q] * diff
        d_alpha[q] += (theta[u] - beta[q]) * diff

    theta += lr * d_theta
    beta += lr * d_beta
    alpha += lr * d_alpha
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################
    return theta, beta, alpha


def irt(data, val_data, lr, iterations):
    """Train IRT model.

    You may optionally replace the function arguments to receive a matrix.

    :param data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param val_data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param lr: float
    :param iterations: int
    :return: (theta, beta, val_acc_lst)
    """
    # TODO: Initialize theta and beta.
    num_users = len(set(data["user_id"]))
    num_questions = len(set(data["question_id"]))

    theta = np.random.normal(0, 1, num_users)
    beta = np.random.normal(0, 1, num_questions)
    alpha = np.ones(num_questions)

    val_acc_lst = []
    train_nllk_lst = []
    val_nllk_lst = []

    for i in range(iterations):
        # Update parameters
        theta, beta, alpha = update_parameters(data, lr, theta, beta, alpha)

        # Compute metrics
        train_nllk = neg_log_likelihood(data, theta, beta, alpha)
        val_nllk = neg_log_likelihood(val_data, theta, beta, alpha)
        score = evaluate(val_data, theta, beta, alpha)

        train_nllk_lst.append(train_nllk)
        val_nllk_lst.append(val_nllk)
        val_acc_lst.append(score)

        print(f"Iteration {i + 1}: Train NLLK = {train_nllk:.4f}, Val NLLK = {val_nllk:.4f}, Val Acc = {score:.4f}")

    # TODO: You may change the return values to achieve what you want.
    return theta, beta, alpha, train_nllk_lst, val_nllk_lst, val_acc_lst


def evaluate(data, theta, beta, alpha):
    """Evaluate the model given data and return the accuracy.
    :param data: A dictionary {user_id: list, question_id: list,
    is_correct: list}

    :param theta: Vector
    :param beta: Vector
    :param alpha: Vector
    :return: float
    """
    pred = []
    for i, q in enumerate(data["question_id"]):
        u = data["user_id"][i]
        x = alpha[q] * (theta[u] - beta[q])
        p_a = sigmoid(x)
        pred.append(p_a >= 0.5)
    return np.mean((data["is_correct"] == np.array(pred)))


def main():
    train_data = load_train_csv("./data")
    # You may optionally use the sparse matrix.
    # sparse_matrix = load_train_sparse("./data")
    val_data = load_valid_csv("./data")
    test_data = load_public_test_csv("./data")

    best_val_acc = 0
    best_params = {}
    results = []

    learning_rates = [0.001, 0.01, 0.05]
    iterations_list = [100, 300, 500]

    for lr in learning_rates:
        for iters in iterations_list:
            print(f"\nTraining with lr={lr}, iterations={iters}")
            theta, beta, alpha, train_nllk, val_nllk, val_acc = irt(
                train_data, val_data, lr, iters
            )

            test_acc = evaluate(test_data, theta, beta, alpha)
            results.append({
                'lr': lr,
                'iterations': iters,
                'val_acc': val_acc[-1],
                'test_acc': test_acc
            })

            if val_acc[-1] > best_val_acc:
                best_val_acc = val_acc[-1]
                best_params = {
                    'lr': lr,
                    'iterations': iters,
                    'theta': theta,
                    'beta': beta,
                    'alpha': alpha
                }

    print("\n=== Hyperparameter Tuning Results ===")
    for r in results:
        print(f"lr={r['lr']}, iter={r['iterations']}: "
              f"Val Acc={r['val_acc']:.4f}, Test Acc={r['test_acc']:.4f}")

    _, _, _, train_nllk, val_nllk, val_acc = irt(
        train_data, val_data,
        best_params['lr'], best_params['iterations']
    )

    plt.figure(figsize=(15, 5))

    plt.subplot(131)
    plt.plot(train_nllk, label='Train')
    plt.plot(val_nllk, label='Validation')
    plt.xlabel("Iteration")
    plt.ylabel("Negative Log-Likelihood")
    plt.title("Training Curve")
    plt.legend()

    plt.subplot(132)
    plt.plot(val_acc)
    plt.xlabel("Iteration")
    plt.ylabel("Accuracy")
    plt.title("Validation Accuracy")

    plt.subplot(133)
    plt.scatter(best_params['beta'], best_params['alpha'], alpha=0.5)
    plt.xlabel("Difficulty (β)")
    plt.ylabel("Discrimination (α)")
    plt.title("Question Parameter Distribution")

    plt.tight_layout()
    plt.savefig('irt_results.png', dpi=300)
    plt.show()

    final_theta = best_params['theta']
    final_beta = best_params['beta']
    final_alpha = best_params['alpha']

    print("\n=== Final Model Performance ===")
    print(f"Best learning rate: {best_params['lr']}")
    print(f"Best iterations: {best_params['iterations']}")
    print(f"Validation Accuracy: {evaluate(val_data, final_theta, final_beta, final_alpha):.4f}")
    print(f"Test Accuracy: {evaluate(test_data, final_theta, final_beta, final_alpha):.4f}")

    np.savez('irt_params.npz',
             theta=final_theta,
             beta=final_beta,
             alpha=final_alpha)

    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################


if __name__ == "__main__":
    main()
