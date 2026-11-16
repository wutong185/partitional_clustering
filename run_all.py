# -*- coding: utf-8 -*-
"""
run_all.py
==========
一键执行全部 7 个 пунктов задания «Разделительная кластеризация».

Порядок (соответствует двум этапам работы):
    16.11 — пункты 1–3:  step1_program.py, step2_3_experiments.py
    23.11 — пункты 4–7:  step4_noise.py, step5_6_noise_experiments.py, step7_report.py

Запуск:
    python run_all.py
    python run_all.py --only 1 2 3      # только первые три пункта (сдача 16.11)
"""

import argparse
import os
import subprocess
import sys
import time

BASE = os.path.dirname(os.path.abspath(__file__))

STEPS = [
    (1, "step1_program.py", "ПУНКТ 1 — программа k-Means / k-Medoids"),
    (2, "step2_3_experiments.py", "ПУНКТЫ 2–3 — эксперименты на customers + визуализация"),
    (3, "step4_noise.py", "ПУНКТ 4 — параметр доли шума и внесение шума"),
    (4, "step5_6_noise_experiments.py", "ПУНКТЫ 5–6 — эксперименты с шумом + визуализация"),
    (5, "step7_report.py", "ПУНКТ 7 — формирование PDF-отчёта"),
]

# какой скрипт закрывает какие номера пунктов задания
COVERS = {
    1: "1",
    2: "2, 3",
    3: "4",
    4: "5, 6",
    5: "7",
}


def main():
    ap = argparse.ArgumentParser(description="Прогнать все пункты задания")
    ap.add_argument("--only", nargs="*", type=int, default=None,
                    help="номера ШАГОВ (1..5), например: --only 1 2")
    args = ap.parse_args()

    steps = STEPS
    if args.only:
        steps = [s for s in STEPS if s[0] in args.only]

    t_all = time.time()
    failed = []

    for num, script, title in steps:
        print("\n" + "#" * 78)
        print(f"# ШАГ {num}/{len(STEPS)}: {title}")
        print(f"# закрывает пункты задания: {COVERS[num]}   (файл: {script})")
        print("#" * 78)

        t0 = time.time()
        r = subprocess.run([sys.executable, script], cwd=BASE)
        dt = time.time() - t0

        if r.returncode != 0:
            failed.append((script, r.returncode))
            print(f"!!! ОШИБКА в {script} (код {r.returncode})")
        else:
            print(f"--- готово за {dt:.1f} c")

    # ---------- итог ----------
    print("\n" + "=" * 78)
    print("ИТОГ")
    print("=" * 78)
    if failed:
        print("Завершились с ошибкой:")
        for s, c in failed:
            print(f"  {s} (код {c})")
        sys.exit(1)

    print(f"Все шаги выполнены успешно за {time.time() - t_all:.1f} c.\n")
    print("Результаты лежат в папке results/:")
    res = os.path.join(BASE, "results")
    if os.path.isdir(res):
        for name in sorted(os.listdir(res)):
            p = os.path.join(res, name)
            print(f"  {name:<46} {os.path.getsize(p) / 1024:>8.1f} KB")
    print(f"\nОтчёт: {os.path.join(BASE, 'Отчет_Разделительная_кластеризация.pdf')}")


if __name__ == "__main__":
    main()
