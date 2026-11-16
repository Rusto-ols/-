
import numpy as np
from pathlib import Path
import csv

BASE = Path(__file__).resolve().parent

with (BASE / 'ratings5.csv').open(encoding='utf-8-sig', newline='') as f:
    rows = list(csv.reader(f))

users = rows[0][1:]
products = [row[0] for row in rows[1:]]
R = np.array([[float(x) for x in row[1:]] for row in rows[1:]])

with (BASE / 'similarity5.csv').open(encoding='utf-8-sig', newline='') as f:
    similarity_rows = list(csv.reader(f))

assert similarity_rows[0][1:] == products
assert [row[0] for row in similarity_rows[1:]] == products

sim = np.array([[float(x) for x in row[1:]] for row in similarity_rows[1:]])

THRESHOLD = 0.85


def mean_positive(values):
    nonzero = values[values > 0]
    return float(nonzero.mean()) if nonzero.size else 0.0


def recommend(user_index):
    rated = R[:, user_index]

    if not np.any(rated > 0):
        averages = np.array([mean_positive(row) for row in R])

        if not np.any(averages > 0):
            return None, None, 'Нет оценок ни одного товара'

        i = int(np.argmax(averages))
        return (
            products[i],
            float(averages[i]),
            'Новый пользователь: средний рейтинг товара'
        )

    predictions = []

    for i in range(len(products)):
        if rated[i] > 0:
            continue

        neighbors = [
            j for j in range(len(products))
            if rated[j] > 0 and sim[i, j] >= THRESHOLD
        ]

        if not neighbors:
            print(
                f'  Pr({users[user_index]}, {products[i]}) = '
                'нет похожих оценённых товаров'
            )
            continue

        score = (
            sum(sim[i, j] * rated[j] for j in neighbors)
            / sum(sim[i, j] for j in neighbors)
        )

        print(f'  Pr({users[user_index]}, {products[i]}) = {score:.2f}')
        predictions.append((score, products[i]))

    if not predictions:
        return None, None, 'Нет подходящих неоценённых товаров'

    score, product = max(predictions, key=lambda item: item[0])

    return product, score, 'Рекомендация по сходству товаров'


def main():
    print(f'Порог сходства товаров: {THRESHOLD}')

    for index, user in enumerate(users):
        print(f'\n{user}:')

        product, score, reason = recommend(index)

        print(f'  {reason}')
        print(
            f'  Рекомендация: {product}'
            + (f' (оценка {score:.2f})' if score is not None else '')
        )


if __name__ == '__main__':
    main()

