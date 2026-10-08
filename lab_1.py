import numpy as np


def cosine_sim(a, b):
    dot = np.sum(a * b)
    norm_a = np.sqrt(np.sum(a ** 2))
    norm_b = np.sqrt(np.sum(b ** 2))
    return dot / (norm_a * norm_b)


def similarity_matrix(matrix):
    n = matrix.shape[0]
    sim = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            sim[i, j] = cosine_sim(matrix[i], matrix[j])
    return sim


def most_similar_pair(matrix):
    sim = similarity_matrix(matrix)
    np.fill_diagonal(sim, 0)
    return np.unravel_index(np.argmax(sim), sim.shape)


products = np.loadtxt("similarity.csv", delimiter=",")
users_matrix = products.T

names = ['P1', 'P2', 'P3', 'P4', 'P5', 'P6']
users = ['U1', 'U2', 'U3', 'U4', 'U5']

ind = most_similar_pair(products)
print(names[ind[0]], "-", names[ind[1]])

ind = most_similar_pair(users_matrix)
print(users[ind[0]], "-", users[ind[1]])