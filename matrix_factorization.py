import numpy as np
np.random.seed(2025)
from scipy.linalg import sqrtm
import matplotlib.pyplot as plt
from utils import (
    load_train_csv,
    load_valid_csv,
    load_public_test_csv,
    load_train_sparse,
    sparse_matrix_evaluate,
)


def svd_reconstruct(matrix, k):
    """Given the matrix, perform singular value decomposition
    to reconstruct the matrix.

    :param matrix: 2D sparse matrix
    :param k: int
    :return: 2D matrix
    """
    # First, you need to fill in the missing values (NaN) to perform SVD.
    # Fill in the missing values using the average on the current item.
    # Note that there are many options to do fill in the
    # missing values (e.g. fill with 0).
    new_matrix = matrix.copy()
    mask = np.isnan(new_matrix)
    masked_matrix = np.ma.masked_array(new_matrix, mask)
    item_means = np.mean(masked_matrix, axis=0)
    new_matrix = masked_matrix.filled(item_means)

    # Next, compute the average and subtract it.
    item_means = np.mean(new_matrix, axis=0)
    mu = np.tile(item_means, (new_matrix.shape[0], 1))
    new_matrix = new_matrix - mu

    # Perform SVD.
    Q, s, Ut = np.linalg.svd(new_matrix, full_matrices=False)
    s = np.diag(s)

    # Choose top k eigenvalues.
    s = s[0:k, 0:k]
    Q = Q[:, 0:k]
    Ut = Ut[0:k, :]
    s_root = sqrtm(s)

    # Reconstruct the matrix.
    reconst_matrix = np.dot(np.dot(Q, s_root), np.dot(s_root, Ut))
    reconst_matrix = reconst_matrix + mu
    return np.array(reconst_matrix)


def squared_error_loss(data, u, z):
    """Return the squared-error-loss given the data.
    :param data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param u: 2D matrix
    :param z: 2D matrix
    :return: float
    """
    loss = 0
    for i, q in enumerate(data["question_id"]):
        loss += (data["is_correct"][i] - np.sum(u[data["user_id"][i]] * z[q])) ** 2.0
    return 0.5 * loss


def update_u_z(train_data, lr, u, z):
    """Return the updated U and Z after applying
    stochastic gradient descent for matrix completion.

    :param train_data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param lr: float
    :param u: 2D matrix
    :param z: 2D matrix
    :return: (u, z)
    """
    #####################################################################
    # TODO:                                                             #
    # Implement the function as described in the docstring.             #
    #####################################################################
    # Randomly select a pair (user_id, question_id).
    i = np.random.choice(len(train_data["question_id"]), 1)[0]

    c = train_data["is_correct"][i]
    n = train_data["user_id"][i]
    q = train_data["question_id"][i]

    pred = np.dot(u[n], z[q])
    error = c - pred

    u_n_old = u[n].copy()
    u[n] += lr * error * z[q]
    z[q] += lr * error * u_n_old

    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################
    return u, z


def als(train_data, k, lr, num_iteration):
    """Performs ALS algorithm, here we use the iterative solution - SGD
    rather than the direct solution.

    :param train_data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param k: int
    :param lr: float
    :param num_iteration: int
    :return: 2D reconstructed Matrix.
    """
    # Initialize u and z
    u = np.random.uniform(
        low=0, high=1 / np.sqrt(k), size=(len(set(train_data["user_id"])), k)
    )
    z = np.random.uniform(
        low=0, high=1 / np.sqrt(k), size=(len(set(train_data["question_id"])), k)
    )

    #####################################################################
    # TODO:                                                             #
    # Implement the function as described in the docstring.             #
    #####################################################################

    for _ in range(num_iteration):
        u, z = update_u_z(train_data, lr, u, z)
    mat = np.dot(u, z.T)
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################
    return mat

def als_with_records(train_data, val_data, k, lr, num_iter, eval_every=1000):
    # new version with record
    num_users = len(set(train_data["user_id"]))
    num_items = len(set(train_data["question_id"]))
    u = np.random.uniform(0, 1/np.sqrt(k), (num_users, k))
    z = np.random.uniform(0, 1/np.sqrt(k), (num_items, k))

    iters, tr_losses, vl_losses = [], [], []
    for it in range(1, num_iter+1):
        u, z = update_u_z(train_data, lr, u, z)
        if it % eval_every == 0 or it == 1:
            iters.append(it)
            tr_losses.append(squared_error_loss(train_data, u, z))
            vl_losses.append(squared_error_loss(val_data, u, z))
    return iters, tr_losses, vl_losses

def main():
    train_matrix = load_train_sparse("./data").toarray()
    train_data = load_train_csv("./data")
    val_data = load_valid_csv("./data")
    test_data = load_public_test_csv("./data")

    #####################################################################
    # TODO:                                                             #
    # (SVD) Try out at least 5 different k and select the best k        #
    # using the validation set.                                         #
    #####################################################################
    k_list = [10, 20, 30, 40, 50]

    best_k, best_val_acc, best_test_acc = None, -1, None
    for k in k_list:
        reconst = svd_reconstruct(train_matrix, k)
        val_acc = sparse_matrix_evaluate(val_data, reconst)
        test_acc = sparse_matrix_evaluate(test_data, reconst)
        print(f"SVD k={k}: val_acc={val_acc:.4f}, test_acc={test_acc:.4f}")
        if val_acc > best_val_acc:
            best_k, best_val_acc, best_test_acc = k, val_acc, test_acc
    print(f"Best SVD -> k={best_k}, val={best_val_acc:.4f}, test={best_test_acc:.4f}\n")
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################

    #####################################################################
    # TODO:                                                             #
    # (ALS) Try out at least 5 different k and select the best k        #
    # using the validation set.                                         #
    #####################################################################
    ks  = [5, 10, 20, 30, 40]
    lrs = [0.005, 0.01, 0.02]
    iters = [10000, 20000, 50000]
    best = {"val_acc": -1}
    for k in ks:
        for lr in lrs:
            for num_iter in iters:
                print(f"ALS k={k}, lr={lr}, iters={num_iter}")
                reconst = als(train_data, k, lr, num_iter)
                val_acc = sparse_matrix_evaluate(val_data, reconst)
                test_acc= sparse_matrix_evaluate(test_data, reconst)
                print(f"  -> val_acc={val_acc:.4f}, test_acc={test_acc:.4f}")
                if val_acc > best["val_acc"]:
                    best.update({"k":k, "lr":lr, "iters":num_iter,
                                 "val_acc":val_acc, "test_acc":test_acc})
    print(f"\nBest ALS -> {best}")

    # (e) Plot training & validation loss curves
    iters, tr_l, vl_l = als_with_records(
        train_data, val_data,
        best["k"], best["lr"], best["iters"],
        eval_every=max(1, best["iters"] // 20)
    )
    plt.plot(iters, tr_l, label="train loss")
    plt.plot(iters, vl_l, label="val loss")
    plt.xlabel("SGD iterations")
    plt.ylabel("0.5 * squared-error loss")
    plt.title(f"ALS Loss (k={best['k']}, lr={best['lr']})")
    plt.legend()
    plt.savefig('als_q5.png')
    plt.show()
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################


if __name__ == "__main__":
    main()
