import numpy as np

R = 0.85

sim = np.loadtxt("similarity2.csv", delimiter=",")
names = ['U1', 'U2', 'U3', 'U4', 'U5']
clusters = [[n] for n in names]


def label(cluster):
    return ", ".join(cluster)


def print_matrix(sim, clusters, title):
    print(title)
    labels = [label(c) for c in clusters]
    width = max(max(len(l) for l in labels), 4) + 2
    print(" " * width + "".join(l.ljust(width) for l in labels))
    for l, row in zip(labels, sim):
        print(l.ljust(width) + "".join(f"{v:<{width}.2f}" for v in row))
    print()


print_matrix(sim, clusters, "Исходная матрица")

step = 1
while len(clusters) > 1:
    work = sim.copy()
    np.fill_diagonal(work, -1)
    i, j = np.unravel_index(np.argmax(work), work.shape)
    i, j = min(i, j), max(i, j)
    best = work[i, j]

    if best < R:
        print(f"Максимум {best} < R = {R} -> объединение останавливается\n")
        break

    print(f"Шаг {step}: ({label(clusters[i])}) и ({label(clusters[j])}) "
          f"объединены по сходству {best} >= R")

    new_row = np.maximum(sim[i], sim[j])
    sim[i, :] = new_row
    sim[:, i] = new_row
    sim[i, i] = 0

    sim = np.delete(np.delete(sim, j, axis=0), j, axis=1)
    clusters[i] = clusters[i] + clusters[j]
    del clusters[j]

    print_matrix(sim, clusters, f"{step}. Перестройка матрицы")
    step += 1

print("Итоговые кластеры:")
for c in clusters:
    print("  {" + label(c) + "}")