"""Renders and saves every dashboard chart as a PNG for preview."""
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from collections import Counter
import os

os.makedirs("outputs/previews", exist_ok=True)

SPAM_COLOR = "#E74C3C"
HAM_COLOR  = "#2ECC71"
MED_COLOR  = "#e67e22"

plt.rcParams.update({
    "font.family"       : "DejaVu Sans",
    "font.size"         : 12,
    "axes.titlesize"    : 14,
    "axes.titleweight"  : "bold",
    "axes.labelsize"    : 12,
    "axes.labelweight"  : "bold",
    "axes.spines.top"   : False,
    "axes.spines.right" : False,
    "axes.grid"         : True,
    "grid.alpha"        : 0.35,
    "grid.linestyle"    : "--",
    "xtick.labelsize"   : 11,
    "ytick.labelsize"   : 11,
    "legend.fontsize"   : 11,
    "figure.dpi"        : 120,
})

data = pd.read_csv("outputs/spam_cleaned.csv")
spam = data[data["label"] == "spam"]
ham  = data[data["label"] == "ham"]

# ── 1. PIE CHART ──────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(5, 5))
sizes  = [len(spam), len(ham)]
labels = [f"Spam\n{len(spam):,} msgs\n({len(spam)/len(data)*100:.1f}%)",
          f"Ham\n{len(ham):,} msgs\n({len(ham)/len(data)*100:.1f}%)"]
ax.pie(sizes, labels=labels, colors=[SPAM_COLOR, HAM_COLOR],
       startangle=90, textprops={"fontsize": 12, "fontweight": "bold"},
       wedgeprops={"edgecolor": "white", "linewidth": 2})
ax.set_title("Spam vs Ham Distribution", pad=15)
fig.tight_layout()
fig.savefig("outputs/previews/01_pie_chart.png", bbox_inches="tight")
plt.close()
print("[saved] 01_pie_chart.png")

# ── 2. MESSAGE LENGTH HISTOGRAM ───────────────────────────────
fig, ax = plt.subplots(figsize=(8, 4.5))
for label, color in [("Spam", SPAM_COLOR), ("Ham", HAM_COLOR)]:
    d = data[data["label"] == label.lower()]["char_count"]
    ax.hist(d, bins=50, alpha=0.65, color=color, label=label, density=True)
sp_med = spam["char_count"].median()
hm_med = ham["char_count"].median()
ax.axvline(sp_med, color=SPAM_COLOR, linestyle="--", lw=2,
           label=f"Spam median: {sp_med:.0f} chars")
ax.axvline(hm_med, color=HAM_COLOR,  linestyle="--", lw=2,
           label=f"Ham median:  {hm_med:.0f} chars")
ax.set_title("How Long Are Spam vs Ham Messages?")
ax.set_xlabel("Number of Characters")
ax.set_ylabel("Density (proportion of messages)")
ax.legend(framealpha=0.9)
fig.tight_layout()
fig.savefig("outputs/previews/02_length_histogram.png", bbox_inches="tight")
plt.close()
print("[saved] 02_length_histogram.png")

# ── 3. WORD COUNT BOXPLOT ─────────────────────────────────────
fig, ax = plt.subplots(figsize=(7, 4.5))
bp = ax.boxplot(
    [spam["word_count"], ham["word_count"]],
    tick_labels=["Spam", "Ham"],
    patch_artist=True, notch=True, widths=0.45,
    medianprops={"color": "black", "linewidth": 2.5},
    flierprops={"marker": "o", "markersize": 4, "alpha": 0.4}
)
bp["boxes"][0].set_facecolor(SPAM_COLOR); bp["boxes"][0].set_alpha(0.75)
bp["boxes"][1].set_facecolor(HAM_COLOR);  bp["boxes"][1].set_alpha(0.75)
ax.set_title("Word Count: Spam vs Ham (Box Plot)")
ax.set_xlabel("Message Type")
ax.set_ylabel("Number of Words per Message")
ax.yaxis.grid(True, linestyle="--", alpha=0.5)
fig.tight_layout()
fig.savefig("outputs/previews/03_wordcount_boxplot.png", bbox_inches="tight")
plt.close()
print("[saved] 03_wordcount_boxplot.png")

# ── 4. FEATURE GROUPED BAR CHART ─────────────────────────────
names  = ["Has URL", "Has Number", "Has Prize Word",
          "Has 'FREE'", "Has 'CALL'", "Has 'TXT/TEXT'"]
scols  = ["has_url","has_number","has_currency","has_free","has_call","has_txt"]
srates = [spam[c].mean()*100 for c in scols]
hrates = [ham[c].mean()*100  for c in scols]
x = np.arange(len(names)); width = 0.38

fig, ax = plt.subplots(figsize=(12, 5))
bars_s = ax.bar(x - width/2, srates, width, label="Spam",
                color=SPAM_COLOR, alpha=0.85, zorder=3)
bars_h = ax.bar(x + width/2, hrates, width, label="Ham",
                color=HAM_COLOR,  alpha=0.85, zorder=3)
for bar in bars_s:
    h = bar.get_height()
    if h > 0.5:
        ax.text(bar.get_x()+bar.get_width()/2, h+0.8, f"{h:.1f}%",
                ha="center", va="bottom", fontsize=10, fontweight="bold", color=SPAM_COLOR)
for bar in bars_h:
    h = bar.get_height()
    if h > 0.5:
        ax.text(bar.get_x()+bar.get_width()/2, h+0.8, f"{h:.1f}%",
                ha="center", va="bottom", fontsize=10, fontweight="bold", color="#27AE60")
ax.set_title("How Often Does Each Feature Appear in Spam vs Ham Messages?")
ax.set_xlabel("Feature"); ax.set_ylabel("% of Messages Containing Feature")
ax.set_xticks(x); ax.set_xticklabels(names, rotation=15, ha="right")
ax.set_ylim(0, max(srates)*1.2); ax.legend()
ax.yaxis.grid(True, linestyle="--", alpha=0.4, zorder=0)
fig.tight_layout()
fig.savefig("outputs/previews/04_feature_bars.png", bbox_inches="tight")
plt.close()
print("[saved] 04_feature_bars.png")

# ── 5. WORD FREQUENCY BARS (spam) ─────────────────────────────
STOPWORDS = [
    "i","me","my","we","our","you","your","he","him","his","she","her","it",
    "its","they","them","this","that","am","is","are","was","were","be","been",
    "have","has","had","do","does","did","a","an","the","and","but","if","or",
    "of","at","by","for","with","to","from","in","on","not","no","so","u","ur",
    "r","ok","hi","hey","get","go","got","ll","just","now","will","can","all",
    "up","out","about","what","when","how","also","then","back","more","over"
]
def get_words(message):
    clean = []
    for word in str(message).lower().split():
        word = word.strip(".,!?:;()[]\"'")
        if word not in STOPWORDS and len(word) > 2:
            clean.append(word)
    return clean

spam_words = sum([get_words(m) for m in spam["message"]], [])
ham_words  = sum([get_words(m) for m in ham["message"]],  [])
spam_count = Counter(spam_words)
ham_count  = Counter(ham_words)
n = 15

for label, count, color, fname in [
    ("Spam", spam_count, SPAM_COLOR, "05_top_spam_words.png"),
    ("Ham",  ham_count,  HAM_COLOR,  "06_top_ham_words.png"),
]:
    top = pd.Series(dict(count.most_common(n))).sort_values()
    fig, ax = plt.subplots(figsize=(7, max(n*0.42, 4)))
    bars = ax.barh(top.index, top.values, color=color,
                   alpha=0.82, edgecolor="white", height=0.65)
    for bar in bars:
        w = bar.get_width()
        ax.text(w + top.values.max()*0.01,
                bar.get_y()+bar.get_height()/2,
                f"{int(w):,}", va="center", ha="left", fontsize=10)
    ax.set_title(f"Top {n} Words Found in {label} Messages", pad=10)
    ax.set_xlabel("Number of Occurrences"); ax.set_ylabel("Word")
    ax.set_xlim(0, top.values.max()*1.15)
    ax.xaxis.grid(True, linestyle="--", alpha=0.4); ax.yaxis.grid(False)
    fig.tight_layout()
    fig.savefig(f"outputs/previews/{fname}", bbox_inches="tight")
    plt.close()
    print(f"[saved] {fname}")

# ── 6. SEGMENTS BAR CHART (message length) ────────────────────
data2 = data.copy()
data2["group"] = pd.cut(data2["char_count"],
    bins=[0,50,100,160,300,9999],
    labels=["0–50 chars","51–100 chars","101–160 chars","161–300 chars","300+ chars"])
seg = data2.groupby("group", observed=True).agg(
    total=("label","count"), spam_count=("label_num","sum"))
seg["spam_rate"] = (seg["spam_count"]/seg["total"]*100).round(1)
avg    = data["label_num"].mean()*100
labels = [str(i) for i in seg.index]
rates  = seg["spam_rate"].values
totals = seg["total"].values
colors = [SPAM_COLOR if r>50 else MED_COLOR if r>20 else HAM_COLOR for r in rates]

fig, ax = plt.subplots(figsize=(11, 5.5))
bars = ax.bar(labels, rates, color=colors, width=0.5,
              alpha=0.85, edgecolor="white", linewidth=1.2, zorder=3)
ax.axhline(avg, color="#2980B9", linestyle="--", lw=2.5, zorder=4,
           label=f"Dataset average: {avg:.1f}%")
y_max = max(rates)
for bar, rate, total in zip(bars, rates, totals):
    bx = bar.get_x()+bar.get_width()/2
    ax.text(bx, bar.get_height()+y_max*0.02, f"{rate:.1f}%",
            ha="center", va="bottom", fontsize=13, fontweight="bold")
    ax.text(bx, -y_max*0.07, f"n={total:,}",
            ha="center", va="top", fontsize=10, color="grey")
ax.set_title("Spam Rate by Message Length", pad=14)
ax.set_xlabel("Message Length Group", labelpad=12)
ax.set_ylabel("Spam Rate (%)", labelpad=10)
ax.set_ylim(-y_max*0.12, y_max*1.22)
ax.yaxis.grid(True, linestyle="--", alpha=0.4, zorder=0)
p_red    = mpatches.Patch(color=SPAM_COLOR, alpha=0.85, label="High spam (>50%)")
p_orange = mpatches.Patch(color=MED_COLOR,  alpha=0.85, label="Medium spam (20–50%)")
p_green  = mpatches.Patch(color=HAM_COLOR,  alpha=0.85, label="Low spam (<20%)")
ax.legend(handles=[p_red, p_orange, p_green,
    plt.Line2D([0],[0], color="#2980B9", linestyle="--", lw=2.5,
               label=f"Average: {avg:.1f}%")], loc="upper left", framealpha=0.9)
fig.tight_layout()
fig.savefig("outputs/previews/07_segments_bar.png", bbox_inches="tight")
plt.close()
print("[saved] 07_segments_bar.png")

# ── 7. CHECK A MESSAGE — signal breakdown ─────────────────────
msg = "FREE prize! Call 08001234567 NOW to claim your reward!"
signals = {
    "Has a URL (http/www)"              : "http" in msg.lower() or "www" in msg.lower(),
    "Has a phone number (10+ digits)"   : any(len(w)>=10 and w.isdigit() for w in msg.split()),
    "Has prize words (prize/cash/win)"  : any(w in msg.lower() for w in ["prize","cash","win"]),
    "Has the word FREE"                 : "free" in msg.lower(),
    "Has the word CALL"                 : "call" in msg.lower(),
    "Has TXT or TEXT"                   : "txt" in msg.lower() or "text" in msg.lower(),
    "Has urgency words (urgent/claim/expire)": any(w in msg.lower() for w in ["urgent","claim","expire"]),
    "Message is long (>100 chars)"      : len(msg) > 100,
    "Has exclamation mark (!)"          : "!" in msg,
}
sig_labels = list(signals.keys())
sig_vals   = [int(v) for v in signals.values()]
bar_colors = [SPAM_COLOR if v else "#AEB6BF" for v in sig_vals]

fig, ax = plt.subplots(figsize=(9, 4))
bars = ax.barh(sig_labels, sig_vals, color=bar_colors,
               alpha=0.85, edgecolor="white", height=0.55)
ax.set_xlim(0, 1.55)
ax.set_xticks([0, 1]); ax.set_xticklabels(["Not Found", "Found"], fontsize=12)
ax.set_title("Which Spam Signals Were Detected in the Sample Message?", pad=10)
ax.set_xlabel("Signal Status"); ax.xaxis.grid(False)
for bar, val in zip(bars, sig_vals):
    label = "✔  Found" if val else "✘  Not found"
    ax.text(bar.get_width()+0.03, bar.get_y()+bar.get_height()/2,
            label, va="center", fontsize=10,
            color=SPAM_COLOR if val else "grey")
found_patch   = mpatches.Patch(color=SPAM_COLOR, alpha=0.85, label="Signal found")
missing_patch = mpatches.Patch(color="#AEB6BF",  alpha=0.85, label="Not found")
ax.legend(handles=[found_patch, missing_patch], loc="lower right")
fig.tight_layout()
fig.savefig("outputs/previews/08_signal_breakdown.png", bbox_inches="tight")
plt.close()
print("[saved] 08_signal_breakdown.png")

print("\nAll charts saved to outputs/previews/")
