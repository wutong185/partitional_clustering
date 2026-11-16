# -*- coding: utf-8 -*-
"""
step2_3_experiments.py
======================
ПУНКТ 2 задания — эксперименты на наборе данных customers.
ПУНКТ 3 задания — визуализация результатов:
    (а) точечный график, где цвет точки отражает принадлежность кластеру;
    (б) зависимость ошибки кластеризации от параметра k.

Что делает:
    * перебирает k = 3…9 для k-Means и k-Medoids на наборе customers;
    * сохраняет таблицу results/base_error_vs_k.csv;
    * строит:
        results/fig3_1_scatter_kmeans.png     — точечный график k-Means (k=5)
        results/fig3_2_scatter_kmedoids.png   — точечный график k-Medoids (k=5)
        results/fig3_3_error_vs_k.png         — ошибка кластеризации от k

Запуск:
    python step2_3_experiments.py
"""

import os

import pandas as pd

from clustering import (BASE_DIR, FEATURES, RESULT_DIR, load_customers,
                        kmeans, kmedoids, save_clusters, standardize)
from viz import plot_error_vs_k, scatter_clusters

K_RANGE = list(range(3, 10))       # 3, 4, 5, 6, 7, 8, 9
SCATTER_K = 5                      # для точечных графиков
SEED = 42


def main():
    print("=" * 78)
    print("ПУНКТ 2 — ЭКСПЕРИМЕНТЫ НА НАБОРЕ customers")
    print("ПУНКТ 3 — ВИЗУАЛИЗАЦИЯ РЕЗУЛЬТАТОВ")
    print("=" * 78)

    X_raw, df = load_customers()
    Xz, mean, std = standardize(X_raw)
    print(f"Объектов: {len(Xz)}   Признаков: {len(FEATURES)}   "
          f"k = {K_RANGE[0]}…{K_RANGE[-1]}")
    print("-" * 78)
    print(f"{'k':>3} | {'k-Means: ошибка (SSE)':>24} | {'k-Medoids: ошибка (сумма расст.)':>34}")
    print("-" * 78)

    rows = []
    for k in K_RANGE:
        rk = kmeans(Xz, k, seed=SEED)
        rm = kmedoids(Xz, k, seed=SEED)
        rows.append({
            "k": k,
            "kmeans_error": rk["error"],
            "kmeans_iter": rk["n_iter"],
            "kmedoids_error": rm["error"],
            "kmedoids_iter": rm["n_iter"],
        })
        print(f"{k:>3} | {rk['error']:>24.4f} | {rm['error']:>34.4f}")

    df_res = pd.DataFrame(rows)
    csv_path = os.path.join(RESULT_DIR, "base_error_vs_k.csv")
    df_res.to_csv(csv_path, index=False, float_format="%.4f")
    print("-" * 78)
    print(f"Таблица сохранена: {os.path.relpath(csv_path, BASE_DIR)}")

    # ---------- ПУНКТ 3 (а): точечные графики ----------
    print("\nПостроение точечных графиков (k = %d) ..." % SCATTER_K)
    rk5 = kmeans(Xz, SCATTER_K, seed=SEED)
    rm5 = kmedoids(Xz, SCATTER_K, seed=SEED)

    f1 = scatter_clusters(
        Xz, rk5["labels"],
        f"k-Means: кластеры при k = {SCATTER_K}\n"
        f"(проекция на 2 главные компоненты; ошибка SSE = {rk5['error']:.2f})",
        "fig3_1_scatter_kmeans.png",
        centers=rk5["centers"], center_kind="centroid", k=SCATTER_K)

    f2 = scatter_clusters(
        Xz, rm5["labels"],
        f"k-Medoids: кластеры при k = {SCATTER_K}\n"
        f"(проекция на 2 главные компоненты; ошибка = {rm5['error']:.2f})",
        "fig3_2_scatter_kmedoids.png",
        centers=rm5["medoids"], center_kind="medoid", k=SCATTER_K)

    # ---------- ПУНКТ 3 (б): ошибка от k ----------
    f3 = plot_error_vs_k(
        [
            {"label": "k-Means (SSE)", "ks": df_res["k"], "errors": df_res["kmeans_error"],
             "color": "#e74c3c", "marker": "o"},
            {"label": "k-Medoids (сумма расстояний)", "ks": df_res["k"],
             "errors": df_res["kmedoids_error"], "color": "#2980b9", "marker": "s"},
        ],
        "Зависимость ошибки кластеризации от числа кластеров k\n"
        "(набор данных customers, без шума)",
        "fig3_3_error_vs_k.png")

    # ---------- сохранить кластеры для k=5 ----------
    save_clusters(df, rk5["labels"],
                  os.path.join(RESULT_DIR, "clusters_kmeans_k5.csv"))
    save_clusters(df, rm5["labels"],
                  os.path.join(RESULT_DIR, "clusters_kmedoids_k5.csv"))

    print(f"  {os.path.relpath(f1, BASE_DIR)}")
    print(f"  {os.path.relpath(f2, BASE_DIR)}")
    print(f"  {os.path.relpath(f3, BASE_DIR)}")
    print("\n" + "=" * 78)
    print("Пункты 2 и 3 выполнены.")
    print("=" * 78)


if __name__ == "__main__":
    main()
