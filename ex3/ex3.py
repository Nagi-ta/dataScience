import csv
import math

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
import matplotlib.font_manager as fm
import matplotlib.patheffects as path_effects

# read csv
rank_list = []
with open("./redwinerank.csv") as f:
    reader = csv.reader(f)
    for row in reader:
        row = [row[0]] + [int(r) for r in row[1:]]
        rank_list.append(row)
        print(row)

# CG
for ranks in rank_list:
    accum = 0
    CG = []
    for r in ranks[1:]:
        accum += r
        CG.append(accum)

    print(f"CG:{ranks[0]} {CG}")

# DCG
# CG
for ranks in rank_list:
    accum = 0
    DCG = []
    for i, r in enumerate(ranks[1:]):
        if i == 0:
            accum = r
        else:
            accum += r / math.log2(i + 1)
        DCG.append(accum)

    print(f"DCG:{ranks[0]} {DCG}")

# nDCG
ideal_list = [3] * 20


plt.figure(figsize=(10, 10))
for ranks in rank_list:
    DCG_accum = 0
    ideal_accum = 0
    nDCG = []
    for i, (r, ideal) in enumerate(zip(ranks[1:], ideal_list)):
        if i == 0:
            DCG_accum = r
            ideal_accum = ideal
        else:
            DCG_accum += r / math.log2(i + 1)
            ideal_accum += ideal / math.log2(i + 1)
        nDCG.append(DCG_accum / ideal_accum)

    print(f"nDCG:{ranks[0]} {nDCG}")

    plt.plot(
        list(range(1, 21)),
        nDCG,
        label=ranks[0],
        linewidth=3,
        path_effects=[path_effects.SimpleLineShadow(), path_effects.Normal()],
    )


plt.tick_params(labelsize=15)
plt.xlabel("Top 20", fontsize=20)  # ,font_properties=fp2)
plt.ylabel("nDCG", fontsize=20)  # ,font_properties=fp2)
plt.title(
    "Comparison of Google and Yahoo images",
    fontsize=20,
)  # font_properties=fp2)

plt.legend([r[0] for r in rank_list], loc="best", fontsize=20)
plt.savefig("ex3-nDCG.pdf")  # スケーラブルな PDF に出力
