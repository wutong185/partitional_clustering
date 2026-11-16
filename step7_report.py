# -*- coding: utf-8 -*-
"""
step7_report.py
===============
ПУНКТ 7 задания — формирование отчёта в формате PDF.

Обязательные разделы (по заданию):
    1. формулировка задания;
    2. гиперссылка на каталог репозитория с исходными текстами, наборами данных
       и др. сопутствующими материалами;
    3. рисунки с результатами визуализации;
    4. пояснения, раскрывающие смысл полученных результатов.

Отчёт охватывает все 7 пунктов задания и оба этапа:
    16.11 — пункты 1-3 (k-Means на наборе customers);
    23.11 — пункты 4-7 (k-Medoids, шум, визуализация, отчёт).

Запуск (после step1, step2_3, step4, step5_6):
    python step7_report.py
"""

import os

import pandas as pd
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (Image, PageBreak, Paragraph, SimpleDocTemplate,
                                Spacer, Table, TableStyle)

BASE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(BASE, "results")
REPO_URL = "https://github.com/wutong185/partitional_clustering"
OUT_NAME = "Отчет_Разделительная_кластеризация.pdf"

pdfmetrics.registerFont(TTFont("Arial", "C:/Windows/Fonts/arial.ttf"))
pdfmetrics.registerFont(TTFont("Arial-Bold", "C:/Windows/Fonts/arialbd.ttf"))

styles = getSampleStyleSheet()
H1 = ParagraphStyle("H1", parent=styles["Heading1"], fontName="Arial-Bold",
                    fontSize=16, textColor=colors.HexColor("#1a365d"), spaceAfter=6)
H2 = ParagraphStyle("H2", parent=styles["Heading2"], fontName="Arial-Bold",
                    fontSize=13, textColor=colors.HexColor("#2b6cb0"),
                    spaceBefore=12, spaceAfter=6)
H3 = ParagraphStyle("H3", parent=styles["Heading3"], fontName="Arial-Bold",
                    fontSize=11.5, textColor=colors.HexColor("#2b6cb0"),
                    spaceBefore=8, spaceAfter=4)
body = ParagraphStyle("body", parent=styles["BodyText"], fontName="Arial",
                      fontSize=10.5, leading=15, alignment=TA_JUSTIFY, spaceAfter=6)
cap = ParagraphStyle("cap", parent=styles["BodyText"], fontName="Arial",
                     fontSize=9.5, leading=13, alignment=TA_CENTER,
                     spaceAfter=10, textColor=colors.HexColor("#555555"))
lst = ParagraphStyle("lst", parent=styles["BodyText"], fontName="Arial",
                     fontSize=9.5, leading=14, leftIndent=10, spaceAfter=3)


def P(t):
    return Paragraph(t, body)


def L(t):
    return Paragraph(t, lst)


def make_table(data, widths, font_size=9):
    tbl = Table(data, colWidths=widths)
    tbl.setStyle(TableStyle([
        ("FONTNAME", (0, 0), (-1, -1), "Arial"),
        ("FONTNAME", (0, 0), (-1, 0), "Arial-Bold"),
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#2b6cb0")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("BACKGROUND", (0, 1), (-1, -1), colors.HexColor("#ebf4ff")),
        ("FONTSIZE", (0, 0), (-1, -1), font_size),
    ]))
    return tbl


def add_image(story, path, width_mm, height_mm, caption):
    if os.path.exists(path):
        story.append(Paragraph(caption, cap))
        story.append(Image(path, width=width_mm * mm, height=height_mm * mm))
        story.append(Spacer(1, 8))


def page_number(canvas, doc):
    canvas.saveState()
    canvas.setFont("Arial", 9)
    canvas.drawCentredString(A4[0] / 2, 12 * mm, f"— {doc.page} —")
    canvas.restoreState()


def main():
    base_df = pd.read_csv(os.path.join(RES, "base_error_vs_k.csv"))
    noise_df = pd.read_csv(os.path.join(RES, "noise_error_matrix.csv"))
    noise_ex = pd.read_csv(os.path.join(RES, "noise_example_5pct.csv"))

    k5_km = base_df.loc[base_df.k == 5, "kmeans_error"].iloc[0]
    k5_kd = base_df.loc[base_df.k == 5, "kmedoids_error"].iloc[0]

    def noise_err(algo, noise, k):
        q = noise_df[(noise_df.noise == noise) & (noise_df.k == k)]
        col = "kmeans_error" if algo == "kmeans" else "kmedoids_error"
        return float(q[col].iloc[0])

    story = []

    # ================= титул =================
    story.append(Paragraph("Отчет по заданию «Разделительная кластеризация»", H1))
    story.append(Paragraph(
        "Набор данных: customers (сведения о клиентах банка, 850 объектов)", H2))

    story.append(P("Работа выполнялась в два этапа:"))
    story.append(L("&nbsp;&nbsp;<b>16.11</b> — пункты <b>1–3</b>: программа (k-Means и "
                   "k-Medoids), эксперименты на наборе customers, визуализация;"))
    story.append(L("&nbsp;&nbsp;<b>23.11</b> — пункты <b>4–7</b>: зашумление данных, "
                   "эксперименты с шумом, визуализация, настоящий отчёт."))
    story.append(Spacer(1, 4))

    # ================= 1 =================
    story.append(Paragraph("1. Формулировка задания", H2))
    for i, txt in [
        (1, "Разработать программу, выполняющую кластеризацию заданного набора данных "
            "с помощью алгоритмов <b>k-Means</b> и <b>k-Medoids</b>. Параметры программы: "
            "набор данных и число кластеров. Программа должна выдавать координаты точек "
            "и назначенные им кластеры, а также значение ошибки кластеризации."),
        (2, "Провести эксперименты на наборе данных <b>customers</b> (сведения о клиентах "
            "банка: CSV с данными и описанием)."),
        (3, "Выполнить визуализацию результатов: (а) точечный график, цвет точки отражает "
            "принадлежность кластеру; (б) зависимость ошибки кластеризации от параметра k."),
        (4, "Доработать программу, добавив параметр «доля зашумлённых объектов». "
            "Программа должна вносить шум: случайным образом изменять заданную долю "
            "объектов (добавление/вычитание случайного числа к/из одной или нескольких "
            "координат объекта)."),
        (5, "Провести эксперименты, варьируя долю зашумлённых объектов "
            "(<b>1%, 3%, 5%, 10%</b>) и число кластеров <b>k = 3…9</b>."),
        (6, "Выполнить визуализацию результатов указанным выше способом."),
        (7, "Подготовить отчёт в формате PDF с разделами: формулировка задания; "
            "гиперссылка на каталог репозитория; рисунки с результатами визуализации; "
            "пояснения, раскрывающие смысл полученных результатов."),
    ]:
        story.append(L(f"<b>{i}.</b> {txt}"))

    # ================= 2 =================
    story.append(Paragraph("2. Гиперссылка на каталог репозитория", H2))
    story.append(P(
        "Исходные тексты программы, набор данных, таблицы результатов и рисунки "
        "размещены в репозитории GitHub: "
        '<link href="%s" color="#1a4fd6"><u>%s</u></link>' % (REPO_URL, REPO_URL)))
    story.append(P("Клонирование: <b>git clone %s.git</b>" % REPO_URL))
    story.append(P("Состав репозитория и соответствие пунктам задания:"))
    story.append(L("&nbsp;&nbsp;• <b>clustering.py</b> — ядро: k-Means, k-Medoids, PCA, "
                   "внесение шума, CLI (пункты 1 и 4);"))
    story.append(L("&nbsp;&nbsp;• <b>step1_program.py</b> — пункт 1: демонстрация работы "
                   "программы (координаты, кластеры, ошибка);"))
    story.append(L("&nbsp;&nbsp;• <b>step2_3_experiments.py</b> — пункты 2–3: эксперименты "
                   "на customers и визуализация;"))
    story.append(L("&nbsp;&nbsp;• <b>step4_noise.py</b> — пункт 4: параметр доли шума и "
                   "пример зашумления;"))
    story.append(L("&nbsp;&nbsp;• <b>step5_6_noise_experiments.py</b> — пункты 5–6: "
                   "эксперименты с шумом и визуализация;"))
    story.append(L("&nbsp;&nbsp;• <b>step7_report.py</b> — пункт 7: формирование "
                   "настоящего отчёта;"))
    story.append(L("&nbsp;&nbsp;• <b>viz.py</b> — построение точечных графиков и графиков "
                   "ошибки;"))
    story.append(L("&nbsp;&nbsp;• <b>data/</b> — customers.csv, custmers.txt (описание "
                   "признаков);"))
    story.append(L("&nbsp;&nbsp;• <b>results/</b> — таблицы (CSV) и рисунки (PNG)."))

    # ================= 3 =================
    story.append(PageBreak())
    story.append(Paragraph("3. Рисунки с результатами визуализации", H2))

    # --- 3.1 пункт 1 ---
    story.append(Paragraph("3.1. Пункт 1 — что выдаёт программа", H3))
    story.append(P(
        "Программа запускается с параметрами «набор данных» и «число кластеров» "
        "(<b>python clustering.py --data data/customers.csv --k 5</b>) и для каждого "
        "алгоритма выдаёт: число итераций до сходимости, значение ошибки кластеризации, "
        "размеры кластеров, координаты центроидов (или номера объектов-медоидов), "
        "а также сохраняет таблицу «координаты точек + номер кластера» в CSV."))
    t1 = [["k", "k-Means: ошибка (SSE)", "k-Medoids: ошибка (сумма расстояний)"]]
    for _, r in base_df.iterrows():
        t1.append([str(int(r.k)), f"{r.kmeans_error:.2f}", f"{r.kmedoids_error:.2f}"])
    story.append(make_table(t1, [20 * mm, 55 * mm, 75 * mm]))
    story.append(Paragraph("Таблица 1. Значения ошибки кластеризации, которые выдаёт "
                           "программа при разных k (без шума).", cap))

    # --- 3.2 пункты 2-3 ---
    story.append(Paragraph("3.2. Пункты 2–3 — эксперименты на customers и визуализация", H3))
    story.append(P(
        "Набор customers содержит 850 клиентов банка. Для кластеризации использованы "
        "7 числовых признаков: <i>Age, Education, YearsEmployed, Income, CardDebt, "
        "OtherDebt, DebtIncomeRatio</i>. Признак <i>Defaulted</i> исключён: в нём "
        "150 пропусков из 850 (17.6%). Все признаки предварительно стандартизованы "
        "(z-оценки), поскольку их масштабы различаются на порядки (Income — десятки, "
        "CardDebt — единицы)."))
    story.append(P(
        "Так как пространство семимерное, для точечных графиков выполнена проекция на "
        "первые две главные компоненты (PCA). Цвет точки — номер кластера; крестиками "
        "отмечены центроиды (k-Means) и медоиды (k-Medoids)."))

    add_image(story, os.path.join(RES, "fig3_1_scatter_kmeans.png"), 118, 90,
              "Рис. 1. k-Means, k = 5: цвет точки — принадлежность кластеру "
              "(проекция на 2 главные компоненты).")
    add_image(story, os.path.join(RES, "fig3_2_scatter_kmedoids.png"), 118, 90,
              "Рис. 2. k-Medoids, k = 5: цвет точки — принадлежность кластеру; "
              "крестики — медоиды (реальные объекты набора).")

    story.append(PageBreak())
    add_image(story, os.path.join(RES, "fig3_3_error_vs_k.png"), 150, 96,
              "Рис. 3. Зависимость ошибки кластеризации от числа кластеров k "
              "(k = 3…9) для k-Means и k-Medoids, без шума.")

    # --- 3.3 пункт 4 ---
    story.append(Paragraph("3.3. Пункт 4 — параметр доли зашумлённых объектов", H3))
    story.append(P(
        "В программу добавлен параметр <b>--noise</b> (доля зашумлённых объектов) и "
        "функция <b>add_noise()</b>. Для каждого выбранного объекта случайно выбирается "
        "от 1 до 2 координат, и к каждой прибавляется или вычитается случайное число "
        "из диапазона 0.5…1.5 СКО соответствующего признака. Проверка объёма зашумления:"))
    t2 = [["Доля шума", "Зашумляется объектов (из 850)"],
          ["1%", "8"], ["3%", "26"], ["5%", "42"], ["10%", "85"]]
    story.append(make_table(t2, [40 * mm, 70 * mm]))
    story.append(Paragraph("Таблица 2. Количество зашумляемых объектов при разных долях.", cap))

    story.append(P("Пример фактического изменения координат (5% шума):"))
    if not noise_ex.empty:
        r = noise_ex.iloc[0]
        feats = [c[:-7] for c in noise_ex.columns if c.endswith("_before")]
        rows = [["Признак", "До шума", "После шума", "Изменение"]]
        for f in feats:
            b, a = r[f + "_before"], r[f + "_after"]
            if abs(a - b) > 1e-12:
                rows.append([f, f"{b:+.4f}", f"{a:+.4f}", f"{a - b:+.4f}"])
        story.append(make_table(rows, [45 * mm, 30 * mm, 30 * mm, 30 * mm]))
        story.append(Paragraph(
            f"Таблица 3. Объект Row = {int(r['Row'])}: зашумлению подверглась одна "
            f"координата. Полная таблица «до/после» — в results/noise_example_5pct.csv.", cap))

    story.append(PageBreak())
    add_image(story, os.path.join(RES, "fig4_noise_examples.png"), 150, 121,
              "Рис. 4. Влияние доли зашумлённых объектов на кластеризацию (k-Means, k = 5). "
              "Чёрным кольцом обведены зашумлённые объекты.")

    # --- 3.4 пункты 5-6 ---
    story.append(Paragraph("3.4. Пункты 5–6 — эксперименты с шумом и визуализация", H3))
    story.append(P(
        "Эксперименты выполнены для всех сочетаний: доля шума 1%, 3%, 5%, 10% и "
        "число кластеров k = 3…9 (28 конфигураций для каждого алгоритма, всего 56 "
        "запусков). Полная таблица — results/noise_error_matrix.csv."))
    t3 = [["Доля шума", "k-Means, k=5", "k-Medoids, k=5", "k-Means, k=9", "k-Medoids, k=9"]]
    km9 = float(base_df.loc[base_df.k == 9, "kmeans_error"].iloc[0])
    kd9 = float(base_df.loc[base_df.k == 9, "kmedoids_error"].iloc[0])
    t3.append(["0% (без шума)", f"{k5_km:.2f}", f"{k5_kd:.2f}",
               f"{km9:.2f}", f"{kd9:.2f}"])
    for noise, lbl in [(0.01, "1%"), (0.03, "3%"), (0.05, "5%"), (0.10, "10%")]:
        t3.append([lbl,
                   f"{noise_err('kmeans', noise, 5):.2f}",
                   f"{noise_err('kmedoids', noise, 5):.2f}",
                   f"{noise_err('kmeans', noise, 9):.2f}",
                   f"{noise_err('kmedoids', noise, 9):.2f}"])
    story.append(make_table(t3, [30 * mm, 30 * mm, 32 * mm, 30 * mm, 32 * mm], font_size=8.5))
    story.append(Paragraph("Таблица 4. Ошибка кластеризации при разной доле шума "
                           "(k = 5 и k = 9).", cap))

    add_image(story, os.path.join(RES, "fig6_1_scatter_kmeans_noise.png"), 150, 121,
              "Рис. 5. k-Means, k = 5: кластеры при доле шума 1%, 3%, 5%, 10% "
              "(кольцо — зашумлённый объект).")
    story.append(PageBreak())
    add_image(story, os.path.join(RES, "fig6_2_scatter_kmedoids_noise.png"), 150, 121,
              "Рис. 6. k-Medoids, k = 5: кластеры при доле шума 1%, 3%, 5%, 10%.")
    story.append(PageBreak())
    add_image(story, os.path.join(RES, "fig6_3_error_vs_k_kmeans.png"), 145, 93,
              "Рис. 7. k-Means: зависимость ошибки (SSE) от k при разных долях шума.")
    add_image(story, os.path.join(RES, "fig6_4_error_vs_k_kmedoids.png"), 145, 93,
              "Рис. 8. k-Medoids: зависимость ошибки (суммы расстояний) от k "
              "при разных долях шума.")
    add_image(story, os.path.join(RES, "fig6_5_error_vs_noise.png"), 145, 93,
              "Рис. 9. Зависимость ошибки кластеризации от доли зашумлённых объектов "
              "(k = 5).")

    # ================= 4 =================
    story.append(PageBreak())
    story.append(Paragraph("4. Пояснения, раскрывающие смысл полученных результатов", H2))

    story.append(Paragraph("4.1. Условия эксперимента", H3))
    story.append(P(
        "Набор customers: 850 объектов, 7 числовых признаков (возраст, образование, стаж, "
        "доход, задолженность по карте, прочие долги, отношение долга к доходу). "
        "Пропуски отсутствуют (признак Defaulted исключён из-за 150 пропусков). "
        "Данные стандартизованы, поэтому ошибка кластеризации измеряется в единицах "
        "СКО и сопоставима между признаками. Для k-Means ошибка — сумма квадратов "
        "расстояний до центроидов (SSE), для k-Medoids — сумма расстояний до медоидов; "
        "это разные функционалы, поэтому их абсолютные значения сравнивать между "
        "алгоритмами нельзя, а сравнивать поведение (динамику) — можно."))

    story.append(Paragraph("4.2. Почему ошибка падает с ростом k", H3))
    story.append(P(
        "Для обоих алгоритмов ошибка монотонно убывает с увеличением числа кластеров: "
        f"у k-Means SSE падает с {base_df.loc[base_df.k == 3, 'kmeans_error'].iloc[0]:.2f} "
        f"(k = 3) до {base_df.loc[base_df.k == 9, 'kmeans_error'].iloc[0]:.2f} (k = 9), "
        f"у k-Medoids — с {base_df.loc[base_df.k == 3, 'kmedoids_error'].iloc[0]:.2f} "
        f"до {base_df.loc[base_df.k == 9, 'kmedoids_error'].iloc[0]:.2f}. "
        "Это ожидаемо: чем больше центров, тем ближе каждый объект к своему центру. "
        "Именно поэтому ошибка не может служить единственным критерием выбора k — "
        "формально минимум достигается при k, равном числу объектов. Для выбора "
        "разумного k используют «метод локтя»: на рисунках 3, 7 и 8 видно замедление "
        "убывания ошибки примерно после k = 5–6, что и является рекомендуемым диапазоном "
        "для этого набора."))

    story.append(Paragraph("4.3. Сравнение k-Means и k-Medoids", H3))
    story.append(P(
        "Оба алгоритма нашли согласованную структуру: при k = 5 выделяются кластеры "
        "«молодые клиенты с небольшим доходом и низкой задолженностью», «клиенты "
        "среднего возраста со стажем и средним доходом», «клиенты с высоким доходом», "
        "а также немногочисленный кластер клиентов с очень большой задолженностью "
        "и высоким отношением долга к доходу. "
        "Отличие в природе центров: центроид k-Means — «средняя точка», которая может "
        "не совпадать ни с одним реальным клиентом, а медоид k-Medoids — всегда "
        "реальный объект набора, поэтому его проще интерпретировать как «типичного "
        "представителя» кластера."))
    story.append(P(
        "Вычислительная сторона: k-Means сходится за 10–27 итераций, k-Medoids — за 1–2 "
        "итерации после жадной инициализации BUILD и SWAP-уточнения, но каждая итерация "
        "k-Medoids дороже, так как требует матрицы расстояний 850×850. Поэтому на данном "
        "объёме данных оба алгоритма работают за секунды, а на больших наборах "
        "k-Medoids заметно медленнее k-Means."))

    story.append(Paragraph("4.4. Влияние зашумлённых объектов", H3))
    km_rel = (noise_err("kmeans", 0.10, 5) - k5_km) / k5_km * 100
    kd_rel = (noise_err("kmedoids", 0.10, 5) - k5_kd) / k5_kd * 100
    story.append(P(
        f"С ростом доли зашумлённых объектов ошибка растёт у обоих алгоритмов, но "
        f"по-разному. При k = 5 и переходе от 0% к 10% шума ошибка k-Means выросла "
        f"на <b>{km_rel:.1f}%</b>, а ошибка k-Medoids — только на <b>{kd_rel:.1f}%</b>. "
        "Причина в том, что центроид — это среднее арифметическое: одиночный выброс "
        "смещает его тем сильнее, чем меньше точек в кластере, тогда как медоид "
        "выбирается только среди реальных объектов и является «устойчивой» (робастной) "
        "оценкой центра. Таким образом, <b>k-Medoids заметно устойчивее к шуму</b>, "
        "и это видно не только в таблице 4, но и на рисунках 5–6: у k-Means при 10% "
        "шума часть кластеров «размывается», появляются вытянутые и смещённые "
        "образования, тогда как кластеры k-Medoids сохраняют форму."))
    story.append(P(
        "Кроме того, чем больше k, тем чувствительнее результат к шуму: при малом числе "
        "кластеров каждый кластер крупный и одиночный выброс почти не меняет среднее, "
        "а при большом k кластеры маленькие, и один зашумлённый объект может "
        "«перетянуть» на себя целый кластер. На рисунках 7–8 это видно как расхождение "
        "кривых для разных долей шума, которое увеличивается с ростом k."))

    story.append(Paragraph("4.5. Выводы", H3))
    story.append(P(
        "1. Реализованы оба алгоритма разделительной кластеризации (k-Means и k-Medoids) "
        "«с нуля»; программа параметризуется набором данных и числом кластеров и "
        "выдаёт координаты точек, номера кластеров и значение ошибки — пункт 1 выполнен."))
    story.append(P(
        "2. На наборе customers при k = 3…9 ошибка монотонно убывает с ростом k; "
        "«метод локтя» указывает на k ≈ 5–6 как разумный компромисс между качеством "
        "и интерпретируемостью — пункты 2–3 выполнены."))
    story.append(P(
        "3. Добавлен параметр доли зашумлённых объектов и функция внесения шума, "
        "изменяющая от одной до нескольких координат заданной доли объектов — "
        "пункт 4 выполнен."))
    story.append(P(
        "4. При варьировании доли шума (1%, 3%, 5%, 10%) и k = 3…9 показано, что "
        "k-Medoids существенно устойчивее k-Means: при 10% шума относительный рост "
        f"ошибки составил {kd_rel:.1f}% против {km_rel:.1f}% — пункты 5–6 выполнены."))
    story.append(P(
        "5. Оформлен настоящий отчёт с гиперссылкой на репозиторий, рисунками и "
        "пояснениями — пункт 7 выполнен."))

    # ---------- сборка ----------
    out = os.path.join(BASE, OUT_NAME)
    doc = SimpleDocTemplate(out, pagesize=A4,
                            rightMargin=20 * mm, leftMargin=20 * mm,
                            topMargin=18 * mm, bottomMargin=20 * mm,
                            title="Отчет: Разделительная кластеризация",
                            author="wutong185")
    doc.build(story, onFirstPage=page_number, onLaterPages=page_number)
    print("Отчёт сформирован:", out)


if __name__ == "__main__":
    main()
