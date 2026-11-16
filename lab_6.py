
def fuzzy_and(a, b):
    return min(a, b)


def fuzzy_or(a, b):
    return max(a, b)


def combine(a, b):
    return a + b * (1 - a)


def main():
    from pathlib import Path
    import csv

    with (
        Path(__file__).resolve().parent / 'fuzzy6.csv'
    ).open(encoding='utf-8-sig', newline='') as f:
        rows = list(csv.DictReader(f))

    groups = {
        rule: [r for r in rows if r['rule'] == rule]
        for rule in ('E1', 'E2')
    }

    assert len(groups['E1']) == 2 and len(groups['E2']) == 2

    md1 = fuzzy_and(*(float(r['MD']) for r in groups['E1']))
    mnd1 = fuzzy_and(*(float(r['MND']) for r in groups['E1']))

    md2 = fuzzy_or(*(float(r['MD']) for r in groups['E2']))
    mnd2 = fuzzy_or(*(float(r['MND']) for r in groups['E2']))

    md = combine(md1, md2)
    mnd = combine(mnd1, mnd2)

    ku = md - mnd

    print(f'E1 (И): MD = {md1:.4f}; MND = {mnd1:.4f}')
    print(f'E2 (ИЛИ): MD = {md2:.4f}; MND = {mnd2:.4f}')

    print(
        f'Итоговая MD = {md1:.2f} + {md2:.2f} '
        f'× (1 − {md1:.2f}) = {md:.4f}'
    )

    print(
        f'Итоговая MND = {mnd1:.2f} + {mnd2:.2f} '
        f'× (1 − {mnd1:.2f}) = {mnd:.4f}'
    )

    print(
        f'Коэффициент уверенности KU = '
        f'{md:.4f} − {mnd:.4f} = {ku:.4f}'
    )


if __name__ == '__main__':
    main()
