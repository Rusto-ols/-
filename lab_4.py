import numpy as np

EXCLUDE = ['U4']

matrix = np.loadtxt("similarity4.csv", delimiter=",")
products = ['P1', 'P2', 'P3', 'P4', 'P5']
users = ['U1', 'U2', 'U3', 'U4']

keep = [i for i, u in enumerate(users) if u not in EXCLUDE]
users = [users[i] for i in keep]
R = matrix[:, keep].T


def cosine_sim(a, b):
    return np.sum(a * b) / (np.sqrt(np.sum(a ** 2)) * np.sqrt(np.sum(b ** 2)))


def mean_rating(vec):
    rated = vec[vec > 0]
    return rated.mean() if len(rated) > 0 else 0


means = np.array([mean_rating(r) for r in R])
n = len(users)
sim = np.array([[cosine_sim(R[i], R[j]) if means[i] > 0 and means[j] > 0 else 0
                 for j in range(n)] for i in range(n)])
np.fill_diagonal(sim, 0)

print("Сходство пользователей:")
print("    " + "  ".join(users))
for u, row in zip(users, sim):
    print(u.ljust(3), "  ".join(f"{v:.2f}" for v in row))
print("Средние рейтинги:", {u: round(float(m), 2) for u, m in zip(users, means)})


def predict(a, i):
    num, den = 0.0, 0.0
    for u in range(n):
        if u != a and R[u, i] > 0:
            num += (R[u, i] - means[u]) * sim[a, u]
            den += abs(sim[a, u])
    if den == 0:
        return None
    return means[a] + num / den


def recommend(a):
    if means[a] == 0:
        avg = [mean_rating(R[:, i]) for i in range(R.shape[1])]
        return products[int(np.argmax(avg))], None

    best, best_val = None, -1
    for i in range(R.shape[1]):
        if R[a, i] == 0:
            val = predict(a, i)
            print(f"  Pr({users[a]}, {products[i]}) = {val:.2f}")
            if val is not None and val > best_val:
                best, best_val = products[i], val
    return best, best_val


print()
for a in range(n):
    print(users[a])
    prod, val = recommend(a)
    if prod is None:
        print("  рекомендовать нечего: всё уже оценено")
    else:
        print(f"  рекомендовать {prod}")
