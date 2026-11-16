from pathlib import Path
import numpy as np

BASE = Path(__file__).resolve().parent
EXCLUDE = {'U4'}
products = ['P1', 'P2', 'P3', 'P4', 'P5']
all_users = ['U1', 'U2', 'U3', 'U4']
source = np.loadtxt(BASE / 'similarity4.csv', delimiter=',')
if source.shape != (len(products), len(all_users)):
    raise ValueError('Размер матрицы не совпадает со списками товаров и пользователей')
keep = [i for i, user in enumerate(all_users) if user not in EXCLUDE]
users = [all_users[i] for i in keep] + ['U_new']
R = np.column_stack((source[:, keep], np.zeros(len(products)))).T


def mean_rating(values):
    rated = values[values > 0]
    return float(rated.mean()) if rated.size else 0.0


def cosine_sim(a, b):
    norm = np.linalg.norm(a) * np.linalg.norm(b)
    return float(np.dot(a, b) / norm) if norm else 0.0


def recommend(a, sim, means):
    if not np.any(R[a] > 0):
        item_means = np.array([mean_rating(R[:, i]) for i in range(len(products))])
        if not np.any(item_means > 0):
            return None, None, 'Нет оценок товаров для формирования рекомендации'
        best = int(np.argmax(item_means))
        return products[best], float(item_means[best]), 'Новый пользователь: максимальный средний рейтинг товара'

    predictions = []
    for i, product in enumerate(products):
        if R[a, i] > 0:
            continue
        neighbors = [u for u in range(len(users)) if u != a and R[u, i] > 0 and sim[a, u] > 0]
        if not neighbors:
            print(f'  Pr({users[a]}, {product}) = нет подходящих соседей')
            continue
        numerator = sum((R[u, i] - means[u]) * sim[a, u] for u in neighbors)
        denominator = sum(abs(sim[a, u]) for u in neighbors)
        prediction = means[a] + numerator / denominator
        print(f'  Pr({users[a]}, {product}) = {prediction:.2f}')
        predictions.append((prediction, product))
    if not predictions:
        return None, None, 'Нет неоценённых товаров с подходящими соседями'
    value, product = max(predictions, key=lambda pair: pair[0])
    return product, value, 'Прогноз на основе похожих пользователей'


def main():
    means = np.array([mean_rating(row) for row in R])
    sim = np.array([[cosine_sim(R[i], R[j]) if i != j else 0.0
                     for j in range(len(users))] for i in range(len(users))])
    print('Матрица сходства пользователей:')
    print('     ' + ' '.join(f'{u:>7}' for u in users))
    for u, row in zip(users, sim):
        print(f'{u:>5}' + ''.join(f'{x:7.2f}' for x in row))
    print('\nСредние оценки:', dict(zip(users, np.round(means, 2))))
    for a, user in enumerate(users):
        print(f'\n{user}:')
        product, value, explanation = recommend(a, sim, means)
        print(f'  {explanation}')
        print(f'  Рекомендация: {product}' + (f' (оценка {value:.2f})' if value is not None else ''))


if __name__ == '__main__':
    main()
