# -*- coding: utf-8 -*-
"""
viz.py
======
Визуализация результатов кластеризации (пункты 3 и 6 задания):
    * точечный график, цвет точки = принадлежность кластеру;
    * зависимость ошибки кластеризации от параметра k.

Данные семимерные, поэтому для двумерного графика используется проекция
на первые две главные компоненты (PCA).
"""

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from clustering import pca_2d, RESULT_DIR

plt.rcParams["font.family"] = "Arial"
plt.rcParams["axes.unicode_minus"] = False

PALETTE = ["#e74c3c", "#2980b9", "#27ae60", "#f39c12", "#8e44ad",
           "#16a085", "#d35400", "#2c3e50", "#c0392b"]
MARKERS = ["o", "s", "^", "D", "v", "P", "X", "*", "h"]


def scatter_clusters(X, labels, title, filename,
                     centers=None, center_kind="centroid",
                     noisy_mask=None, k=None):
    """
    Точечный график: цвет точки отражает кластер.
    Данные проецируются на 2 главные компоненты (PCA).

    :param centers:     координаты центроидов (k-Means) или медоидов (k-Medoids)
                        в исходном (стандартизованном) пространстве
    :param center_kind: 'centroid' или 'medoid' — подпись в легенде
    """
    X2, explained = pca_2d(X)
    k = k or int(labels.max() + 1)

    fig, ax = plt.subplots(figsize=(7.6, 5.8), dpi=150)

    for j in range(k):
        m = labels == j
        ax.scatter(X2[m, 0], X2[m, 1], s=26, alpha=0.75,
                   color=PALETTE[j % len(PALETTE)],
                   marker="o", linewidths=0,
                   label=f"Кластер {j + 1} ({int(m.sum())} объектов)")

    # зашумлённые объекты — обводим тёмным кольцом
    if noisy_mask is not None and bool(np.any(noisy_mask)):
        ax.scatter(X2[noisy_mask, 0], X2[noisy_mask, 1], s=70,
                   facecolors="none", edgecolors="black", linewidths=1.3,
                   label=f"Зашумлённые ({int(np.sum(noisy_mask))})")

    # центры / медоиды (проецируются тем же преобразованием, что и данные)
    if centers is not None and len(centers) > 0:
        mu = X.mean(axis=0)
        _, _, Vt = np.linalg.svd(X - mu, full_matrices=False)
        C2 = (np.asarray(centers) - mu) @ Vt[:2].T
        ax.scatter(C2[:, 0], C2[:, 1], s=190, marker="X",
                   color="black", edgecolors="white", linewidths=1.6,
                   zorder=5,
                   label=("Центроиды" if center_kind == "centroid" else "Медоиды"))

    ax.set_title(title, fontsize=11.5)
    ax.set_xlabel(f"Главная компонента 1 ({explained[0]:.1%} дисперсии)")
    ax.set_ylabel(f"Главная компонента 2 ({explained[1]:.1%} дисперсии)")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8, loc="best", framealpha=0.9)
    fig.tight_layout()
    path = os.path.join(RESULT_DIR, filename)
    fig.savefig(path)
    plt.close(fig)
    return path


def plot_error_vs_k(series, title, filename, xlabel="Число кластеров k"):
    """
    Зависимость ошибки кластеризации от k.

    :param series: список dict(label, ks, errors, color, marker, ls)
    """
    fig, ax = plt.subplots(figsize=(7.8, 5.0), dpi=150)
    for s in series:
        ax.plot(s["ks"], s["errors"], marker=s.get("marker", "o"),
                color=s.get("color", None), ls=s.get("ls", "-"),
                lw=2, ms=6, label=s["label"])
    ax.set_title(title, fontsize=11.5)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Ошибка кластеризации")
    ax.set_xticks(sorted({k for s in series for k in s["ks"]}))
    ax.grid(alpha=0.3)
    ax.legend(fontsize=9)
    fig.tight_layout()
    path = os.path.join(RESULT_DIR, filename)
    fig.savefig(path)
    plt.close(fig)
    return path


def plot_noise_panels(panels, title, filename, ncols=2):
    """
    Сетка точечных графиков (например, для разных долей шума).
    :param panels: список dict(X, labels, noisy_mask, subtitle, k)
    """
    nrows = int(np.ceil(len(panels) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(6.2 * ncols, 5.0 * nrows), dpi=150)
    axes = np.atleast_1d(axes).ravel()

    for ax, p in zip(axes, panels):
        X2, _ = pca_2d(p["X"])
        k = p.get("k") or int(p["labels"].max() + 1)
        for j in range(k):
            m = p["labels"] == j
            ax.scatter(X2[m, 0], X2[m, 1], s=16, alpha=0.75,
                       color=PALETTE[j % len(PALETTE)], linewidths=0)
        nm = p.get("noisy_mask")
        if nm is not None and nm.any():
            ax.scatter(X2[nm, 0], X2[nm, 1], s=46, facecolors="none",
                       edgecolors="black", linewidths=1.1)
        ax.set_title(p["subtitle"], fontsize=10)
        ax.grid(alpha=0.2)
        ax.set_xticks([])
        ax.set_yticks([])

    for ax in axes[len(panels):]:
        ax.axis("off")

    fig.suptitle(title, fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    path = os.path.join(RESULT_DIR, filename)
    fig.savefig(path)
    plt.close(fig)
    return path
