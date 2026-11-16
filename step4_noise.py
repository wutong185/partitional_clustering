# -*- coding: utf-8 -*-
"""
step4_noise.py
==============
ПУНКТ 4 задания — доработка программы: параметр «доля зашумлённых объектов»
и функция внесения шума в набор данных.

Требование пункта 4:
    «Доработайте программу, добавив в список её параметров долю зашумлённых
     объектов набора. Дополнительно к ранее реализованным функциям программа
     должна вносить шум в набор данных: случайным образом изменить заданную
     долю объектов набора (изменение может заключаться в добавлении/вычитании
     случайного числа к/из одной/нескольких координат объекта).»

Реализация — функция clustering.add_noise(X, fraction, seed, n_coords, scale)
и параметр командной строки --noise (например --noise 0.05 = 5%).

Что делает этот скрипт:
    * показывает, сколько объектов зашумляется при 1%, 3%, 5%, 10%;
    * показывает на конкретном объекте, как именно меняются координаты;
    * строит наглядный рисунок «чистые данные → 1% → 5% → 10%»;
    * сохраняет таблицу с координатами до/после для первых зашумлённых объектов.

Запуск:
    python step4_noise.py
"""

import os

import numpy as np
import pandas as pd

from clustering import (BASE_DIR, FEATURES, RESULT_DIR, add_noise, kmeans,
                        load_customers, save_clusters, standardize)
from viz import plot_noise_panels

NOISE_LEVELS = [0.01, 0.03, 0.05, 0.10]     # 1%, 3%, 5%, 10%
SEED = 42


def main():
    print("=" * 78)
    print("ПУНКТ 4 — ПАРАМЕТР «ДОЛЯ ЗАШУМЛЁННЫХ ОБЪЕКТОВ» И ВНЕСЕНИЕ ШУМА")
    print("=" * 78)

    X_raw, df = load_customers()
    Xz, mean, std = standardize(X_raw)
    n = len(Xz)

    print(f"Объектов в наборе: {n}")
    print(f"Число признаков:   {len(FEATURES)}")
    print("\nСколько объектов зашумляется при разных долях:")
    for p in NOISE_LEVELS:
        print(f"  {p:.0%}  ->  {int(round(n * p))} объектов")

    # ---------- наглядный пример: как меняется один объект ----------
    print("\nПример изменения координат (объект Row = 0), 5% шума:")
    Xn5, mask5 = add_noise(Xz, 0.05, seed=SEED)
    idx = int(np.where(mask5)[0][0])
    before = Xz[idx]
    after = Xn5[idx]
    changed = np.where(np.abs(after - before) > 1e-12)[0]
    print(f"  объект №{idx} (Row = {int(df['Row'].iloc[idx])})")
    for j in changed:
        print(f"    {FEATURES[j]:<16}: {before[j]:+.4f}  ->  {after[j]:+.4f}"
              f"   (изменение {after[j] - before[j]:+.4f})")
    print(f"  изменено координат: {len(changed)} из {len(FEATURES)}")

    # ---------- таблица «до / после» для первых зашумлённых объектов ----------
    rows = []
    for i in np.where(mask5)[0][:10]:
        rec = {"Row": int(df["Row"].iloc[i]), "index": int(i)}
        for j, f in enumerate(FEATURES):
            rec[f + "_before"] = round(float(Xz[i, j]), 4)
            rec[f + "_after"] = round(float(Xn5[i, j]), 4)
        rows.append(rec)
    tbl_path = os.path.join(RESULT_DIR, "noise_example_5pct.csv")
    pd.DataFrame(rows).to_csv(tbl_path, index=False)
    print(f"\nТаблица «до/после» сохранена: {os.path.relpath(tbl_path, BASE_DIR)}")

    # ---------- рисунок: чистые данные и разные доли шума ----------
    print("\nПостроение рисунка «чистые данные → 1% → 5% → 10%» ...")
    panels = []

    clean = kmeans(Xz, 5, seed=SEED)
    panels.append({"X": Xz, "labels": clean["labels"], "noisy_mask": None,
                   "subtitle": "Без шума (0%)", "k": 5})

    for p in (0.01, 0.05, 0.10):
        Xn, mask = add_noise(Xz, p, seed=SEED)
        res = kmeans(Xn, 5, seed=SEED)
        panels.append({"X": Xn, "labels": res["labels"], "noisy_mask": mask,
                       "subtitle": f"Шум {p:.0%} ({int(mask.sum())} объектов)", "k": 5})

    fig_path = plot_noise_panels(
        panels,
        "Влияние доли зашумлённых объектов на кластеризацию (k-Means, k = 5)\n"
        "чёрным кольцом обведены зашумлённые объекты",
        "fig4_noise_examples.png", ncols=2)

    print(f"  {os.path.relpath(fig_path, BASE_DIR)}")
    print("\n" + "=" * 78)
    print("Пункт 4 выполнен: добавлен параметр --noise и функция внесения шума.")
    print("=" * 78)


if __name__ == "__main__":
    main()
