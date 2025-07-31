import numpy as np
from sklearn.impute import KNNImputer
import matplotlib.pyplot as plt
from utils import (
    load_valid_csv,
    load_public_test_csv,
    load_train_sparse,
    sparse_matrix_evaluate,
)


def knn_impute_by_user(matrix, valid_data, k):
    """Fill in the missing values using k-Nearest Neighbors based on
    student similarity. Return the accuracy on valid_data.

    See https://scikit-learn.org/stable/modules/generated/sklearn.
    impute.KNNImputer.html for details.

    :param matrix: 2D sparse matrix
    :param valid_data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param k: int
    :return: float
    """
    nbrs = KNNImputer(n_neighbors=k)
    # We use NaN-Euclidean distance measure.
    mat = nbrs.fit_transform(matrix)
    acc = sparse_matrix_evaluate(valid_data, mat)
    # print("Validation Accuracy: {}".format(acc))
    print(f"Validation Accuracy (user-based, k={k}): {acc:.4f}")
    return acc


def knn_impute_by_item(matrix, valid_data, k):
    """Fill in the missing values using k-Nearest Neighbors based on
    question similarity. Return the accuracy on valid_data.

    :param matrix: 2D sparse matrix
    :param valid_data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param k: int
    :return: float
    """
    #####################################################################
    # TODO:                                                             #
    # Implement the function as described in the docstring.             #
    #####################################################################
    imputer = KNNImputer(n_neighbors=k)
    filled_transposed = imputer.fit_transform(matrix.T)
    filled = filled_transposed.T
    acc = sparse_matrix_evaluate(valid_data, filled)
    print(f"Validation Accuracy (item-based, k={k}): {acc:.4f}")
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################
    return acc


def main():
    sparse_matrix = load_train_sparse("./data").toarray()
    val_data = load_valid_csv("./data")
    test_data = load_public_test_csv("./data")

    print("Sparse matrix:")
    print(sparse_matrix)
    print("Shape of sparse matrix:")
    print(sparse_matrix.shape)

    #####################################################################
    # TODO:                                                             #
    # Compute the validation accuracy for each k. Then pick k* with     #
    # the best performance and report the test accuracy with the        #
    # chosen k*.                                                        #
    #####################################################################
    k_list = [1, 6, 11, 16, 21, 26]
    user_val_acc = []
    item_val_acc = []

    # Compute validation accuracy
    for k in k_list:
        acc_u = knn_impute_by_user(sparse_matrix, val_data, k)
        acc_i = knn_impute_by_item(sparse_matrix, val_data, k)
        user_val_acc.append(acc_u)
        item_val_acc.append(acc_i)

    # Plot
    plt.figure()
    plt.plot(k_list, user_val_acc, marker='o', label='user-based')
    plt.plot(k_list, item_val_acc, marker='o', label='item-based')
    plt.xlabel('k (number of neighbors)')
    plt.ylabel('Validation Accuracy')
    plt.title('KNN Performance')
    plt.legend()
    plt.grid(True)
    # plt.savefig('knn_q1.png')
    plt.savefig('knn_q3.png')
    plt.show()

    # Select best k*
    best_u_idx = int(np.argmax(user_val_acc))
    best_i_idx = int(np.argmax(item_val_acc))
    best_k_user = k_list[best_u_idx]
    best_k_item = k_list[best_i_idx]
    print(f"Chosen k* for user-based: {best_k_user} (val acc = {user_val_acc[best_u_idx]:.4f})")
    print(f"Chosen k* for item-based: {best_k_item} (val acc = {item_val_acc[best_i_idx]:.4f})")

    # Evaluate
    test_acc_user = knn_impute_by_user(sparse_matrix, test_data, best_k_user)
    test_acc_item = knn_impute_by_item(sparse_matrix, test_data, best_k_item)
    print(f"Test accuracy (user-based, k={best_k_user}): {test_acc_user:.4f}")
    print(f"Test accuracy (item-based, k={best_k_item}): {test_acc_item:.4f}")

    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################


if __name__ == "__main__":
    main()
