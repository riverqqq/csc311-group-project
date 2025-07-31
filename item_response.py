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


def neg_log_likelihood(data, theta, beta):
    """Compute the negative log-likelihood.

    You may optionally replace the function arguments to receive a matrix.

    :param data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param theta: Vector
    :param beta: Vector
    :return: float
    """
    #####################################################################
    # TODO:                                                             #
    # Implement the function as described in the docstring.             #
    #####################################################################
    log_lklihood = 0.0
    for u, q, c in zip(data['user_id'], data['question_id'], data['is_correct']):
        x = theta[u] - beta[q]
        log_lklihood += c * x - np.log(1 + np.exp(x))
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################
    return -log_lklihood


def update_theta_beta(data, lr, theta, beta):
    """Update theta and beta using gradient descent.

    You are using alternating gradient descent. Your update should look:
    for i in iterations ...
        theta <- new_theta
        beta <- new_beta

    You may optionally replace the function arguments to receive a matrix.

    :param data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param lr: float
    :param theta: Vector
    :param beta: Vector
    :return: tuple of vectors
    """
    #####################################################################
    # TODO:                                                             #
    # Implement the function as described in the docstring.             #
    #####################################################################
    grad_theta = np.zeros_like(theta)
    grad_beta = np.zeros_like(beta)
    for u, q, c in zip(data['user_id'], data['question_id'], data['is_correct']):
        x = theta[u] - beta[q]
        p = sigmoid(x)
        # dLL/dtheta_u = (c - p)
        grad_theta[u] += (c - p)
        # dLL/dbeta_q   = -(c - p)
        grad_beta[q] += -(c - p)
    theta = theta + lr * grad_theta
    beta = beta + lr * grad_beta
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################
    return theta, beta


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
    num_users     = max(data['user_id']) + 1
    num_questions = max(data['question_id']) + 1
    theta = np.zeros(num_users)
    beta  = np.zeros(num_questions)

    train_ll = []
    val_ll   = []
    val_acc_lst = []

    for i in range(iterations):
        neg_lld = neg_log_likelihood(data, theta=theta, beta=beta)
        nll_val = neg_log_likelihood(val_data, theta, beta)
        score = evaluate(data=val_data, theta=theta, beta=beta)
        train_ll.append(neg_lld) # a
        val_ll.append(nll_val)
        print("NLLK: {} \t Score: {}".format(neg_lld, score))
        theta, beta = update_theta_beta(data, lr, theta, beta)

    # TODO: You may change the return values to achieve what you want.
    return theta, beta, train_ll, val_ll


def evaluate(data, theta, beta):
    """Evaluate the model given data and return the accuracy.
    :param data: A dictionary {user_id: list, question_id: list,
    is_correct: list}

    :param theta: Vector
    :param beta: Vector
    :return: float
    """
    pred = []
    for i, q in enumerate(data["question_id"]):
        u = data["user_id"][i]
        x = (theta[u] - beta[q]).sum()
        p_a = sigmoid(x)
        pred.append(p_a >= 0.5)
    return np.sum((data["is_correct"] == np.array(pred))) / len(data["is_correct"])


def main():
    train_data = load_train_csv("./data")
    # You may optionally use the sparse matrix.
    # sparse_matrix = load_train_sparse("./data")
    val_data = load_valid_csv("./data")
    test_data = load_public_test_csv("./data")

    #####################################################################
    # TODO:                                                             #
    # Tune learning rate and number of iterations. With the implemented #
    # code, report the validation and test accuracy.                    #
    #####################################################################
    # b
    lr = 0.01
    iterations = 50
    print(f"Training with lr={lr}, iterations={iterations}")

    theta, beta, train_ll, val_ll= irt(train_data, val_data, lr, iterations)
    plt.figure()
    plt.plot(range(1, iterations + 1), train_ll, label='Train LL')
    plt.plot(range(1, iterations + 1), val_ll, label='Val LL')
    plt.xlabel('Iteration')
    plt.ylabel('Neg-Log-Likelihood')
    plt.legend()
    plt.title('Training Curve: Neg-Log-Likelihood')
    plt.savefig('logll_q2.png')
    plt.show()

    # c
    val_acc = evaluate(val_data, theta, beta)
    test_acc = evaluate(test_data, theta, beta)
    print(f"Final Validation Acc: {val_acc:.3f}")
    print(f"Test Acc: {test_acc:.3f}")
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################

    #####################################################################
    # TODO:                                                             #
    # Implement part (d)                                                #
    #####################################################################
    sorted_q = np.argsort(beta)
    qs = [sorted_q[0], sorted_q[len(sorted_q) // 2], sorted_q[-1]]
    theta_range = np.linspace(-3, 3, 200)
    plt.figure()
    for q in qs:
        p_curve = sigmoid(theta_range - beta[q])
        plt.plot(theta_range, p_curve, label=f'Q{q} (β={beta[q]:.2f})')
    plt.xlabel('Ability θ')
    plt.ylabel('P(correct)')
    plt.title('Item Characteristic Curves')
    plt.legend()
    plt.savefig('logll_q4.png')
    plt.show()
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################


if __name__ == "__main__":
    main()
