# -*- coding: utf-8 -*-
"""
clustering.py
=============
Ядро программы разделительной кластеризации (задание «Разделительная кластеризация»).

Реализованы «с нуля» (только numpy):
    * k-Means      — инициализация k-means++, итерации Ллойда, ошибка = SSE;
    * k-Medoids    — инициализация жадным BUILD, затем чередование
                     «назначить точки → выбрать медоид»; ошибка = сумма расстояний;
    * PCA (2 компоненты) — для двумерной визуализации многомерных данных;
    * внесение шума в заданную долю объектов (пункт 4 задания).

Соответствие пунктам задания:
    п.1 — параметры программы: набор данных (--data), число кластеров (--k);
          на выходе — координаты точек, назначенные кластеры и значение ошибки;
    п.2 — набор данных customers (банковские клиенты);
    п.4 — параметр «доля зашумлённых объектов» (--noise) и функция add_noise().

Запуск (пример, пункт 1 задания):
    python clustering.py --data data/customers.csv --k 5 --algo both
    python clustering.py --k 5 --algo kmeans   --noise 0.05
"""

import argparse
import os

import numpy as np
import pandas as pd

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
RESULT_DIR = os.path.join(BASE_DIR, "results")
os.makedirs(RESULT_DIR, exist_ok=True)

# Признаки для кластеризации:
#   Row, CustomerId  — идентификаторы, не признаки;
#   Defaulted        — исключён: 150 пропусков из 850 (17.6%).
FEATURES = ["Age", "Education", "YearsEmployed", "Income",
            "CardDebt", "OtherDebt", "DebtIncomeRatio"]


# =============================================================================
#  Данные
# =============================================================================
def load_customers(path=None, features=None):
    """Читает customers.csv и возвращает (X — матрица признаков, df — исходный DataFrame)."""
    path = path or os.path.join(DATA_DIR, "customers.csv")
    features = features or FEATURES
    df = pd.read_csv(path)
    missing = [f for f in features if f not in df.columns]
    if missing:
        raise ValueError(f"В наборе нет признаков: {missing}")
    X = df[features].to_numpy(dtype=float)
    if np.isnan(X).any():
        raise ValueError("В выбранных признаках есть пропуски — выберите другие признаки")
    return X, df


def standardize(X):
    """Z-стандартизация: (x - mean) / std. Возвращает (Xz, mean, std)."""
    mean = X.mean(axis=0)
    std = X.std(axis=0)
    std[std == 0] = 1.0
    return (X - mean) / std, mean, std


# =============================================================================
#  Пункт 4: внесение шума
# =============================================================================
def add_noise(X, fraction, seed=42, n_coords=(1, 2), scale=1.5):
    """
    Вносит шум в заданную долю объектов (пункт 4 задания).

    Для каждого выбранного объекта:
      * случайно выбирается 1..n_coords координат (по умолчанию от 1 до 2);
      * к каждой из них прибавляется или вычитается случайное число
        из диапазона [0.5*scale, 1.0*scale] (в единицах СКО признака,
        т.к. данные уже стандартизованы).

    :param X:        матрица (n, d), желательно стандартизованная
    :param fraction: доля зашумлённых объектов (0..1), например 0.01 = 1%
    :param seed:     фиксация случайности
    :param n_coords: (мин, макс) число изменяемых координат у объекта
    :param scale:    масштаб шума
    :return: (X_noisy, noisy_mask)
    """
    rng = np.random.RandomState(seed)
    X_noisy = X.copy()
    n, d = X.shape
    n_noisy = int(round(n * fraction))
    if n_noisy <= 0:
        return X_noisy, np.zeros(n, dtype=bool)

    idx = rng.choice(n, size=n_noisy, replace=False)
    noisy_mask = np.zeros(n, dtype=bool)
    noisy_mask[idx] = True

    lo, hi = n_coords
    for i in idx:
        m = rng.randint(lo, hi + 1)
        m = min(m, d)
        coords = rng.choice(d, size=m, replace=False)
        for c in coords:
            magnitude = rng.uniform(0.5, 1.0) * scale
            sign = 1.0 if rng.rand() < 0.5 else -1.0
            X_noisy[i, c] += sign * magnitude
    return X_noisy, noisy_mask


# =============================================================================
#  k-Means
# =============================================================================
def _sq_distances(X, C):
    """Матрица квадратов евклидовых расстояний (n, k)."""
    return ((X[:, None, :] - C[None, :, :]) ** 2).sum(axis=2)


def kmeans(X, k, seed=42, max_iter=300, tol=1e-8):
    """
    k-Means (итерации Ллойда) с инициализацией k-means++.

    :return: dict(labels, centers, error, n_iter)
             error = SSE (сумма квадратов расстояний до центров) — «ошибка кластеризации».
    """
    rng = np.random.RandomState(seed)
    n = X.shape[0]

    # ---- инициализация k-means++ ----
    centers = [X[rng.randint(n)]]
    for _ in range(1, k):
        d2 = np.min(((X[:, None, :] - np.array(centers)[None, :, :]) ** 2).sum(axis=2), axis=1)
        total = d2.sum()
        if total <= 0:
            centers.append(X[rng.randint(n)])
            continue
        probs = d2 / total
        centers.append(X[rng.choice(n, p=probs)])
    C = np.array(centers, dtype=float)

    labels = np.zeros(n, dtype=int)
    for it in range(1, max_iter + 1):
        D = _sq_distances(X, C)
        new_labels = D.argmin(axis=1)

        # ---- пересчёт центров ----
        new_C = C.copy()
        for j in range(k):
            members = X[new_labels == j]
            if len(members) > 0:
                new_C[j] = members.mean(axis=0)
            else:
                # пустой кластер: переносим центр в самую дальнюю точку
                far = D.min(axis=1).argmax()
                new_C[j] = X[far]

        shift = np.abs(new_C - C).max()
        C = new_C
        if np.array_equal(new_labels, labels) or shift < tol:
            labels = new_labels
            break
        labels = new_labels

    D = _sq_distances(X, C)
    labels = D.argmin(axis=1)
    error = float(D.min(axis=1).sum())     # SSE
    return {"labels": labels, "centers": C, "error": error, "n_iter": it}


# =============================================================================
#  k-Medoids
# =============================================================================
def _distances(X, Y):
    """Матрица евклидовых расстояний (n, m)."""
    return np.sqrt(((X[:, None, :] - Y[None, :, :]) ** 2).sum(axis=2))


def kmedoids(X, k, seed=42, max_iter=200, refine=3):
    """
    k-Medoids (PAM): инициализация жадным BUILD + чередование
    «назначить точки ближайшему медоиду → выбрать медоид, минимизирующий
    суммарное расстояние внутри кластера» + SWAP-уточнение.

    Ошибка = сумма расстояний от точек до их медоидов.

    :return: dict(labels, medoid_indices, medoids, error, n_iter)
    """
    rng = np.random.RandomState(seed)
    n = X.shape[0]
    if k >= n:
        raise ValueError("k должно быть меньше числа объектов")

    D = _distances(X, X)                      # матрица расстояний n x n

    # ---------- BUILD: жадно минимизируем суммарную стоимость ----------
    # первый медоид — самый «центральный» объект
    medoids = [int(D.sum(axis=1).argmin())]
    for _ in range(1, k):
        cur_cost = D[:, medoids].min(axis=1)          # текущая стоимость каждой точки
        best_c, best_total = -1, np.inf
        for c in range(n):
            if c in medoids:
                continue
            total = np.minimum(cur_cost, D[:, c]).sum()   # стоимость при добавлении c
            if total < best_total:
                best_total, best_c = total, c
        medoids.append(best_c)

    def total_cost(med):
        return float(D[:, med].min(axis=1).sum())

    cost = total_cost(medoids)
    n_iter = 0

    # ---------- чередование: назначение ↔ выбор медоида ----------
    for n_iter in range(1, max_iter + 1):
        labels = D[:, medoids].argmin(axis=1)

        new_medoids = list(medoids)
        for j in range(k):
            members = np.where(labels == j)[0]
            if len(members) == 0:
                continue
            sub = D[np.ix_(members, members)]
            new_medoids[j] = int(members[sub.sum(axis=1).argmin()])

        if new_medoids == medoids:
            break
        medoids = new_medoids

    # ---------- SWAP-уточнение (как в PAM): пробуем заменить медоид ----------
    for _ in range(refine):
        improved = False
        for j in range(k):
            best_swap, best_cost = None, cost
            others = [m for t, m in enumerate(medoids) if t != j]
            for c in range(n):
                if c in medoids:
                    continue
                cand = others + [c]
                new_cost = float(D[:, cand].min(axis=1).sum())
                if new_cost < best_cost - 1e-9:
                    best_cost, best_swap = new_cost, c
            if best_swap is not None:
                medoids[j] = best_swap
                cost = best_cost
                improved = True
        if not improved:
            break

    labels = D[:, medoids].argmin(axis=1)
    error = float(D[:, medoids].min(axis=1).sum())
    return {"labels": labels, "medoid_indices": np.array(medoids),
            "medoids": X[medoids], "error": error, "n_iter": n_iter}


# =============================================================================
#  PCA (2 компоненты) — для визуализации
# =============================================================================
def pca_2d(X):
    """Проекция на первые две главные компоненты (для точечных графиков)."""
    Xc = X - X.mean(axis=0)
    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    return Xc @ Vt[:2].T, (S ** 2) / np.sum(S ** 2)


# =============================================================================
#  Обёртка: кластеризация одним вызовом
# =============================================================================
def run_clustering(X, k, algo="both", seed=42):
    """Запускает выбранный алгоритм(ы). Возвращает dict имя -> результат."""
    out = {}
    if algo in ("kmeans", "both"):
        out["kmeans"] = kmeans(X, k, seed=seed)
    if algo in ("kmedoids", "both"):
        out["kmedoids"] = kmedoids(X, k, seed=seed)
    return out


def save_clusters(df, labels, path, extra_cols=("Row", "CustomerId")):
    """Пункт 1: сохраняет координаты точек и назначенные им кластеры."""
    out = df.copy()
    out["cluster"] = labels
    cols = [c for c in extra_cols if c in out.columns] + FEATURES + ["cluster"]
    out[["cluster"] + [c for c in cols if c != "cluster"]]
    out.to_csv(path, index=False)
    return path


# =============================================================================
#  CLI (пункт 1 задания)
# =============================================================================
def main():
    p = argparse.ArgumentParser(
        description="Разделительная кластеризация: k-Means и k-Medoids")
    p.add_argument("--data", default=None, help="путь к CSV (по умолчанию data/customers.csv)")
    p.add_argument("--k", type=int, default=5, help="число кластеров")
    p.add_argument("--algo", default="both", choices=["kmeans", "kmedoids", "both"])
    p.add_argument("--noise", type=float, default=0.0,
                   help="доля зашумлённых объектов, 0..1 (например 0.05 = 5%%)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", default=None, help="файл для сохранения координат и кластеров")
    args = p.parse_args()

    X_raw, df = load_customers(args.data)
    Xz, mean, std = standardize(X_raw)

    noisy_mask = np.zeros(len(Xz), dtype=bool)
    if args.noise > 0:
        Xz, noisy_mask = add_noise(Xz, args.noise, seed=args.seed)

    results = run_clustering(Xz, args.k, args.algo, seed=args.seed)

    print("=" * 70)
    print("РАЗДЕЛИТЕЛЬНАЯ КЛАСТЕРИЗАЦИЯ — пункт 1 задания")
    print("=" * 70)
    print(f"Набор данных:      {os.path.basename(args.data or 'customers.csv')}")
    print(f"Объектов:          {len(Xz)}")
    print(f"Признаки ({len(FEATURES)}):     {', '.join(FEATURES)}")
    print(f"Число кластеров:   {args.k}")
    if args.noise > 0:
        print(f"Зашумлено:         {args.noise:.0%} ({noisy_mask.sum()} объектов)")
    print("-" * 70)

    for name, res in results.items():
        title = "k-Means" if name == "kmeans" else "k-Medoids"
        print(f"\n[{title}]  итераций: {res['n_iter']}")
        print(f"  ошибка кластеризации = {res['error']:.4f}"
              + ("  (SSE)" if name == "kmeans" else "  (сумма расстояний)"))

        labels = res["labels"]
        sizes = np.bincount(labels, minlength=args.k)
        print(f"  размеры кластеров:  {list(sizes)}")

        # координаты точек и назначенные кластеры (первые 10 строк)
        head = pd.DataFrame({
            "Row": df["Row"].to_numpy()[:10],
            "cluster": labels[:10],
        })
        for j, f in enumerate(FEATURES):
            head[f] = X_raw[:10, j]
        print("  первые 10 объектов:")
        print(head.to_string(index=False))

        out_path = args.out or os.path.join(RESULT_DIR, f"clusters_{name}_k{args.k}.csv")
        save_clusters(df, labels, out_path)
        print(f"  координаты и кластеры сохранены: {os.path.relpath(out_path, BASE_DIR)}")

    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()
