import numpy as np

MIN_RATING = 4

matrix = np.loadtxt("similarity3.csv", delimiter=",", dtype=int)
products = ['P1', 'P2', 'P3', 'P4', 'P5', 'P6']
users = ['U1', 'U2', 'U3', 'U4', 'U5']


def mid(row):
    rated = row[row > 0]
    return rated.mean() if len(rated) > 0 else 0


mids = np.array([mid(row) for row in matrix])
for p, m in zip(products, mids):
    print(f"{p}: MID = {m:.2f}")

keep_rows = mids >= MIN_RATING
keep_cols = np.any(matrix > 0, axis=0)

filtered = matrix[keep_rows][:, keep_cols]
f_products = [p for p, k in zip(products, keep_rows) if k]
f_users = [u for u, k in zip(users, keep_cols) if k]

print("\nМатрица предпочтений 1:")
print("    " + "  ".join(f_users))
for p, row in zip(f_products, filtered):
    print(p.ljust(3), "  ".join(str(v) for v in row))


def to_coo(a):
    value, row, col = [], [], []
    for i in range(a.shape[0]):
        for j in range(a.shape[1]):
            if a[i, j] != 0:
                value.append(a[i, j])
                row.append(i)
                col.append(j)
    return value, row, col


def to_csr(a):
    value, col, row_index = [], [], [0]
    for i in range(a.shape[0]):
        for j in range(a.shape[1]):
            if a[i, j] != 0:
                value.append(a[i, j])
                col.append(j)
        row_index.append(len(value))
    return value, col, row_index


def to_ellpack(a):
    rows = []
    for i in range(a.shape[0]):
        rows.append([(a[i, j], j) for j in range(a.shape[1]) if a[i, j] != 0])
    width = max(len(r) for r in rows)  
    value = np.zeros((a.shape[0], width), dtype=int)
    column = np.zeros((a.shape[0], width), dtype=int) 
    for i, r in enumerate(rows):
        for k, (v, j) in enumerate(r):
            value[i, k] = v
            column[i, k] = j
    return value, column


def show(a, title):
    print(f"\n===== {title} =====")
    v, r, c = to_coo(a)
    print("COO")
    print("Value:", [int(x) for x in v])
    print("Row:  ", r)
    print("Col:  ", c)

    v, c, ri = to_csr(a)
    print("CSR")
    print("Value:   ", [int(x) for x in v])
    print("Col:     ", c)
    print("RowIndex:", ri)

    v, c = to_ellpack(a)
    print("ELLPACK")
    print("Value:\n", v)
    print("Column:\n", c)


show(matrix, "Матрица предпочтений 0 (исходная)")
show(filtered, "Матрица предпочтений 1 (после фильтрации)")
