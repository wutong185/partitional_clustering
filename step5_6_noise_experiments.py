# -*- coding: utf-8 -*-
"""
step5_6_noise_experiments.py
============================
ПУНКТ 5 задания — эксперименты на наборе customers, варьируя
    * долю зашумлённых объектов: 1%, 3%, 5%, 10%;
    * число кластеров k из интервала 3…9.
ПУНКТ 6 задания — визуализация полученных результатов тем же способом:
    * точечные графики (цвет = кластер, кольцо = зашумлённый объект);
    * зависимость ошибки кластеризации от k для каждого уровня шума.

Что сохраняется:
    results/noise_error_matrix.csv          — полная таблица (шум × k × алгоритм)
    results/fig6_1_scatter_kmeans_noise.png — точечные графики k-Means (4 уровня шума)
    results/fig6_2_scatter_kmedoids_noise.png— точечные графики k-Medoids
    results/fig6_3_error_vs_k_kmeans.png    — ошибка от k при разных долях шума (k-Means)
    results/fig6_4_error_vs_k_kmedoids.png  — то же для k-Medoids
    results/fig6_5_error_vs_noise.png       — ошибка от доли шума при k = 5

Запуск:
    python step5_6_noise_experiments.py
"""

import os

import pandas as pd

from clustering import (BASE_DIR, add_noise, kmeans, kmedoids,
                        load_customers, standardize, RESULT_DIR)
from viz import plot_error_vs_k, plot_noise_panels, scatter_clusters

NOISE_LEVELS = [0.01, 0.03, 0.05, 0.10]     # 1%, 3%, 5%, 10%
K_RANGE = list(range(3, 10))                # 3…9
SCATTER_K = 5                                # k для точечных графиков
SEED = 42

COLORS = ["#e74c3c", "#e67e22", "#27ae60", "#8e44ad"]


def main():
    print("=" * 78)
    print("ПУНКТ 5 — ЭКСПЕРИМЕНТЫ С ШУМОМ (1%, 3%, 5%, 10% × k = 3…9)")
    print("ПУНКТ 6 — ВИЗУАЛИЗАЦИЯ РЕЗУЛЬТАТОВ")
    print("=" * 78)

    X_raw, df = load_customers()
    Xz, _, _ = standardize(X_raw)

    rows = []

    for noise in NOISE_LEVELS:
        Xn, mask = add_noise(Xz, noise, seed=SEED)
        print(f"\nШум {noise:.0%}  ({int(mask.sum())} объектов из {len(Xn)})")
        print(f"  {'k':>3} | {'k-Means (SSE)':>15} | {'k-Medoids (сумма расст.)':>26}")
        print("  " + "-" * 52)
        for k in K_RANGE:
            rk = kmeans(Xn, k, seed=SEED)
            rm = kmedoids(Xn, k, seed=SEED)
            rows.append({
                "noise": noise,
                "noise_pct": int(round(noise * 100)),
                "n_noisy": int(mask.sum()),
                "k": k,
                "kmeans_error": rk["error"],
                "kmedoids_error": rm["error"],
            })
            print(f"  {k:>3} | {rk['error']:>15.4f} | {rm['error']:>26.4f}")

    res = pd.DataFrame(rows)
    csv_path = os.path.join(RESULT_DIR, "noise_error_matrix.csv")
    res.to_csv(csv_path, index=False, float_format="%.4f")
    print(f"\nТаблица сохранена: {os.path.relpath(csv_path, BASE_DIR)}")

    # ================= ПУНКТ 6 (а): точечные графики =================
    print("\nПостроение точечных графиков (k = %d) для всех уровней шума ..." % SCATTER_K)

    for algo, fname, title_ru in [
        ("kmeans", "fig6_1_scatter_kmeans_noise.png", "k-Means"),
        ("kmedoids", "fig6_2_scatter_kmedoids_noise.png", "k-Medoids"),
    ]:
        panels = []
        for noise in NOISE_LEVELS:
            Xn, mask = add_noise(Xz, noise, seed=SEED)
            r = kmeans(Xn, SCATTER_K, seed=SEED) if algo == "kmeans" \
                else kmedoids(Xn, SCATTER_K, seed=SEED)
            panels.append({
                "X": Xn, "labels": r["labels"], "noisy_mask": mask, "k": SCATTER_K,
                "subtitle": f"Шум {noise:.0%}, {title_ru}, k={SCATTER_K}\n"
                            f"ошибка = {r['error']:.2f}",
            })
        p = plot_noise_panels(
            panels,
            f"Кластеры {title_ru} при разной доле зашумлённых объектов (k = {SCATTER_K})\n"
            f"чёрным кольцом обведены зашумлённые объекты",
            fname, ncols=2)
        print(f"  {os.path.relpath(p, BASE_DIR)}")

    # отдельный крупный точечный график для 5% шума (пункт 6, «цвет = кластер»)
    Xn5, mask5 = add_noise(Xz, 0.05, seed=SEED)
    rk5 = kmeans(Xn5, SCATTER_K, seed=SEED)
    big = scatter_clusters(
        Xn5, rk5["labels"],
        f"k-Means, шум 5%, k = {SCATTER_K}: цвет точки — принадлежность кластеру\n"
        f"(ошибка SSE = {rk5['error']:.2f}; кольцо — зашумлённые объекты)",
        "fig6_1b_scatter_noise5_kmeans.png",
        centers=rk5["centers"], center_kind="centroid",
        noisy_mask=mask5, k=SCATTER_K)
    print(f"  {os.path.relpath(big, BASE_DIR)}")

    # ================= ПУНКТ 6 (б): ошибка от k при разных шумах =================
    print("\nПостроение графиков «ошибка от k» для каждого уровня шума ...")

    for algo, col, fname, title_ru in [
        ("kmeans", "kmeans_error", "fig6_3_error_vs_k_kmeans.png", "k-Means (SSE)"),
        ("kmedoids", "kmedoids_error", "fig6_4_error_vs_k_kmedoids.png",
         "k-Medoids (сумма расстояний)"),
    ]:
        series = []
        for i, noise in enumerate(NOISE_LEVELS):
            sub = res[res["noise"] == noise]
            series.append({
                "label": f"шум {noise:.0%}",
                "ks": sub["k"], "errors": sub[col],
                "color": COLORS[i], "marker": ["o", "s", "^", "D"][i],
            })
        p = plot_error_vs_k(
            series,
            f"{title_ru}: зависимость ошибки кластеризации от k\n"
            f"при разной доле зашумлённых объектов (1%, 3%, 5%, 10%)",
            fname)
        print(f"  {os.path.relpath(p, BASE_DIR)}")

    # ================= ошибка от доли шума при k = 5 =================
    series = []
    for algo, col, title_ru, color, marker in [
        ("kmeans", "kmeans_error", "k-Means", "#e74c3c", "o"),
        ("kmedoids", "kmedoids_error", "k-Medoids", "#2980b9", "s"),
    ]:
        sub = res[res["k"] == SCATTER_K]
        series.append({
            "label": title_ru, "ks": sub["noise_pct"], "errors": sub[col],
            "color": color, "marker": marker,
        })
    p = plot_error_vs_k(
        series,
        f"Зависимость ошибки кластеризации от доли зашумлённых объектов (k = {SCATTER_K})",
        "fig6_5_error_vs_noise.png",
        xlabel="Доля зашумлённых объектов, %")
    print(f"  {os.path.relpath(p, BASE_DIR)}")

    print("\n" + "=" * 78)
    print("Пункты 5 и 6 выполнены.")
    print("=" * 78)


if __name__ == "__main__":
    main()
