# -*- coding: utf-8 -*-
"""
step1_program.py
================
ПУНКТ 1 задания — демонстрация программы кластеризации.

Требование пункта 1:
    «Разработайте программу, которая выполняет кластеризацию заданного набора
     данных с помощью алгоритмов k-Means и k-Medoids. Параметрами программы
     являются набор данных и число кластеров. Программа должна выдавать
     координаты точек и назначенные им кластеры, а также значение ошибки
     кластеризации.»

Здесь показывается, что программа:
    * принимает параметры: набор данных и число кластеров;
    * реализует оба алгоритма (k-Means и k-Medoids);
    * выводит координаты точек + номер кластера + значение ошибки.

Сама программа — clustering.py (запускается и из командной строки, см. ниже).

Запуск:
    python step1_program.py
    # или напрямую, с произвольными параметрами:
    python clustering.py --data data/customers.csv --k 4 --algo both
"""

import os

import numpy as np
import pandas as pd

from clustering import (BASE_DIR, FEATURES, RESULT_DIR, load_customers,
                        run_clustering, save_clusters, standardize)

pd.set_option("display.width", 200)


def show_result(name, res, df, k):
    title = "k-Means" if name == "kmeans" else "k-Medoids"
    err_unit = "SSE" if name == "kmeans" else "сумма расстояний"
    labels = res["labels"]
    sizes = np.bincount(labels, minlength=k)

    print(f"\n  --- {title} ---")
    print(f"  итераций до сходимости : {res['n_iter']}")
    print(f"  ошибка кластеризации   : {res['error']:.4f}  ({err_unit})")
    print(f"  размеры кластеров      : {list(int(s) for s in sizes)}")
    if name == "kmeans":
        print(f"  координаты центров (в стандартизованном пространстве):")
        for j, c in enumerate(res["centers"]):
            print("    " + "  ".join(f"{v:+.3f}" for v in c))
    else:
        print(f"  номера объектов-медоидов: {list(int(i) for i in res['medoid_indices'])}")


def main():
    print("=" * 78)
    print("ПУНКТ 1 — ПРОГРАММА КЛАСТЕРИЗАЦИИ (k-Means и k-Medoids)")
    print("=" * 78)

    X_raw, df = load_customers()
    Xz, mean, std = standardize(X_raw)

    print(f"Набор данных : data/customers.csv  ({len(Xz)} объектов)")
    print(f"Признаки ({len(FEATURES)}) : {', '.join(FEATURES)}")
    print(f"Пропуски     : нет (признак Defaulted исключён — 150 пропусков)")

    for k in (3, 5):
        print("\n" + "-" * 78)
        print(f"Число кластеров k = {k}")
        print("-" * 78)

        results = run_clustering(Xz, k, algo="both", seed=42)
        for name, res in results.items():
            show_result(name, res, df, k)

        # ---- координаты точек и назначенные кластеры (первые 8 строк) ----
        labels = results["kmeans"]["labels"]
        table = pd.DataFrame({"Row": df["Row"].to_numpy()[:8],
                              "Cluster (k-Means)": labels[:8]})
        for j, f in enumerate(FEATURES):
            table[f] = X_raw[:8, j]
        print("\n  Координаты точек и назначенные кластеры (первые 8):")
        print(table.to_string(index=False))

        # ---- сохранение полного результата ----
        for name, res in results.items():
            p = os.path.join(RESULT_DIR, f"clusters_{name}_k{k}.csv")
            save_clusters(df, res["labels"], p)
            print(f"  сохранено: {os.path.relpath(p, BASE_DIR)}")

    print("\n" + "=" * 78)
    print("Пункт 1 выполнен: параметры (набор данных, число кластеров) задаются,")
    print("оба алгоритма реализованы, на выходе — координаты, кластеры и ошибка.")
    print("=" * 78)


if __name__ == "__main__":
    main()
