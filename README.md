# 划分式聚类作业（Разделительная кластеризация）

数据挖掘课程作业：用 **k-Means** 和 **k-Medoids** 对**银行客户数据集 customers** 做聚类，
并研究**噪声**对聚类结果的影响。

> 报告（PDF）和程序输出是俄语（要交给老师）；本说明是中文，给你自己看。

---

## 一、两次课分两次完成（老师黑板/消息）

老师说：「今天和 11 月 23 日学划分式聚类。如果你想今天交作业，可以用 k-均值方法。」

| 阶段 | 日期 | 包含的题号 | 内容 |
|---|---|---|---|
| **第一次** | **16.11（今天）** | **题 1–3** | 写程序（k-Means）、在 customers 上实验、画散点图和误差-k 图 |
| **第二次** | **23.11** | **题 4–7** | 加噪声参数、k-Medoids、噪声实验、可视化、出报告 |

所以**今天只交题 1–3 就成立**（k-Means 部分）。题 4–7 是 23.11 的内容，但我已经全部做完了。

---

## 二、7 个小编号 → 对应文件（一眼看清）

| 题号 | 要求 | 对应脚本 | 产出 |
|---|---|---|---|
| **题 1** | 程序：k-Means + k-Medoids，参数=数据集+簇数，输出坐标/簇号/误差 | `clustering.py`（核心）<br>`step1_program.py`（演示） | `results/clusters_*.csv` |
| **题 2** | 在 customers 上做实验 | `step2_3_experiments.py` | `results/base_error_vs_k.csv` |
| **题 3** | 可视化：①散点图（颜色=簇）②误差 vs k | `step2_3_experiments.py` | 图 1、2、3 |
| **题 4** | 加参数「噪声比例」+ 向数据注入噪声 | `clustering.py` 的 `add_noise()`<br>`step4_noise.py`（演示） | `results/noise_example_5pct.csv`、图 4 |
| **题 5** | 噪声 1%/3%/5%/10% × k=3..9 实验 | `step5_6_noise_experiments.py` | `results/noise_error_matrix.csv` |
| **题 6** | 用同样方式可视化 | `step5_6_noise_experiments.py` | 图 5、6、7、8、9 |
| **题 7** | PDF 报告（4 章节） | `step7_report.py` | `Отчет_Разделительная_кластеризация.pdf` |

每个脚本开头都会打印 `ПУНКТ N` 抬头，单跑就是那一题的成果。

---

## 三、数据集

`data/customers.csv`：**850 行 × 10 列**（银行客户）

| 列 | 含义 | 是否用于聚类 |
|---|---|---|
| Row | 行号 | ❌ 标识符 |
| CustomerId | 客户 ID | ❌ 标识符 |
| Age | 年龄 | ✅ |
| Education | 教育程度（1–5 编码） | ✅ |
| YearsEmployed | 工龄 | ✅ |
| Income | 年收入（千） | ✅ |
| CardDebt | 信用卡欠款 | ✅ |
| OtherDebt | 其他欠款 | ✅ |
| Defaulted | 是否破产 | ❌ **有 150 个缺失值（17.6%），排除** |
| DebtIncomeRatio | 负债收入比 | ✅ |

**所以聚类用 7 个数值特征**，并做了 **z-score 标准化**（因为 Income 是几十、CardDebt 是几，
量纲差太多，不标准化会被 Income 主导）。

---

## 四、文件夹结构

```
1116数据挖掘/
├─ clustering.py                  ★ 核心：k-Means / k-Medoids / PCA / 噪声注入 / 命令行
├─ viz.py                         画图函数（散点图、误差-k 图、多面板图）
├─ step1_program.py               题 1：程序演示（坐标+簇+误差）
├─ step2_3_experiments.py         题 2+3：customers 实验 + 可视化
├─ step4_noise.py                 题 4：噪声参数演示
├─ step5_6_noise_experiments.py   题 5+6：噪声实验 + 可视化
├─ step7_report.py                题 7：生成 PDF 报告
├─ run_all.py                     一键跑完 7 个题
├─ data/
│  ├─ customers.csv               数据集
│  └─ custmers.txt                官方字段说明
├─ results/                       所有输出（CSV 表格 + PNG 图）
└─ Отчет_Разделительная_кластеризация.pdf
```

---

## 五、怎么运行

### 一键全跑（推荐）

```powershell
cd C:\Users\HP\Desktop\1116数据挖掘
python run_all.py
```

### 按题号单独跑

```powershell
python step1_program.py            # 题 1
python step2_3_experiments.py      # 题 2+3
python step4_noise.py              # 题 4
python step5_6_noise_experiments.py# 题 5+6
python step7_report.py             # 题 7（报告）
```

### 单独调主程序（验证"参数化"）

```powershell
python clustering.py --k 5                          # 数据集默认 customers，两个算法都跑
python clustering.py --k 4 --algo kmeans            # 只跑 k-Means
python clustering.py --k 6 --algo kmedoids --noise 0.05   # 加 5% 噪声
python clustering.py --data data/customers.csv --k 7      # 显式指定数据集
```

参数：`--data`（数据集）、`--k`（簇数）、`--algo`、`--noise`（噪声比例 0–1）、`--seed`。

**依赖**：`numpy pandas matplotlib reportlab`（本机 Python 3.12 已装好）

---

## 六、结果

### 1）无噪声：误差随 k 变化（题 2–3）

| k | k-Means 误差（SSE） | k-Medoids 误差（距离和） |
|---|---|---|
| 3 | 3702.58 | 1615.52 |
| 4 | 3249.79 | 1492.66 |
| 5 | 2881.37 | 1407.36 |
| 6 | 2603.87 | 1351.41 |
| 7 | 2430.43 | 1307.85 |
| 8 | 2274.72 | 1273.48 |
| 9 | 2152.76 | 1234.51 |

**注意**：两种算法的"误差"定义不同（SSE vs 距离和），**绝对值不能直接对比**，
只能看变化趋势。

### 2）加噪声（题 5–6）

| 噪声 | k-Means (k=5) | k-Medoids (k=5) | k-Means (k=9) | k-Medoids (k=9) |
|---|---|---|---|---|
| 0% | 2881.37 | 1407.36 | 2152.76 | 1234.51 |
| 1% | 2888.13 | 1408.97 | 2161.76 | 1238.35 |
| 3% | 2908.32 | 1415.32 | 2233.61 | 1246.07 |
| 5% | 2938.87 | 1424.37 | 2272.20 | 1276.22 |
| 10% | 3083.88 | 1446.03 | 2331.76 | 1292.85 |

### 3）核心结论

1. **误差随 k 单调下降**，这是必然的（簇越多，每个点离中心越近）——所以误差
   **不能单独用来选 k**。看图 3/7/8 找"拐点"（elbow），大约在 **k ≈ 5–6**。
2. **k-Medoids 更抗噪声**：k=5 时，噪声 0%→10%，k-Means 误差涨了 **7.0%**，
   k-Medoids 只涨了 **2.7%**。
   原因：质心是"平均值"，一个离群点就能把它拉偏；而 medoid 必须从**真实数据点**里选，
   天生更稳健。
3. **k 越大越怕噪声**：簇小的时候，一个噪声点就能"抢走"整个簇。

---

## 七、结果展示看哪些文件

| 文件 | 题号 | 展示什么 |
|---|---|---|
| `results/fig3_1_scatter_kmeans.png` | 题 3 | k-Means（k=5）散点图，颜色=簇，✕=质心 |
| `results/fig3_2_scatter_kmedoids.png` | 题 3 | k-Medoids（k=5）散点图，✕=medoid（真实点） |
| `results/fig3_3_error_vs_k.png` | 题 3 | 无噪声时误差 vs k（两算法） |
| `results/fig4_noise_examples.png` | 题 4 | 0% / 1% / 5% / 10% 噪声对比（黑圈=噪声点） |
| `results/fig6_1_scatter_kmeans_noise.png` | 题 6 | k-Means：4 个噪声水平的散点图 |
| `results/fig6_2_scatter_kmedoids_noise.png` | 题 6 | k-Medoids：4 个噪声水平的散点图 |
| `results/fig6_3_error_vs_k_kmeans.png` | 题 6 | k-Means：各噪声水平下误差 vs k |
| `results/fig6_4_error_vs_k_kmedoids.png` | 题 6 | k-Medoids：各噪声水平下误差 vs k |
| `results/fig6_5_error_vs_noise.png` | 题 6 | 误差 vs 噪声比例（k=5） |
| `Отчет_Разделительная_кластеризация.pdf` | 题 7 | 12 页报告，4 章节齐全 |

> 散点图为什么是"投影"？因为数据是 7 维的，画不了 7 维散点，所以用 **PCA**
> 降到 2 维（报告里写了每个轴解释了多少方差）。

---

## 八、算法实现要点（答辩可能问）

**k-Means**（`clustering.py`）
- 初始化：**k-means++**（避免随机初始化导致的不稳定）
- 迭代：Lloyd 算法（分配 → 更新质心），收敛条件=标签不再变化
- 误差：**SSE**（簇内平方误差和）

**k-Medoids**（`clustering.py`）
- 初始化：**PAM 的 BUILD**（贪心选使总代价最小的点，而不是"最远点"）
- 迭代：分配 → 每个簇内选"到簇内其他点距离和最小"的真实点作为 medoid
- 收尾：**SWAP 精修**（尝试用非 medoid 替换 medoid，若总距离下降就接受）
- 误差：**到 medoid 的距离和**

> 踩过的坑：最初 BUILD 用"选最远点"，结果把离群点选成 medoid，
> 簇大小变成 645/2/2/6/195（三个只含 2 个点的簇）。改成标准 PAM 的
> "最小化总代价"后就正常了（138/266/112/209/125）。

**噪声注入**（`add_noise`）
- 按比例随机选对象
- 每个对象随机选 **1–2 个坐标**
- 每个坐标 **±(0.5~1.5) 倍标准差**（数据已标准化，所以是标准化单位）

---

## 九、常见问题

**Q：为什么不用 sklearn？**
A：作业要求"разработайте программу"（自己开发程序），而且 k-Medoids 本来 sklearn 也没有。
所以两个算法都用 numpy 手写，老师能看出理解程度。

**Q：今天（16.11）要交什么？**
A：题 1–3，即：`step1_program.py` + `step2_3_experiments.py` 的产出
（程序演示 + 图 1/2/3 + `base_error_vs_k.csv`）。

**Q：报告里为什么没有结论说"哪个算法更好"？**
A：两种算法的误差定义不同，绝对值不可比。报告里比较的是**行为**
（谁更抗噪声、谁更快收敛），这个是公平的。
