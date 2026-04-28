"""
05_dashboard.py
===============
SMS Spam Data Exploration - Interactive Dashboard
Alok Chauhan - 251810700318
Aman Kumar   - 251810700231
Batch 2C

HOW TO RUN:
    streamlit run 05_dashboard.py

WHAT THIS FILE NEEDS (put these in the same folder):
    outputs/spam_cleaned.csv   ← created by notebook 1
    outputs/ml_results.json    ← created by train_model.py
    outputs/previews/*.png     ← created by save_charts.py
    spam_model.pkl             ← created by train_model.py  (optional)

INSTALL REQUIREMENTS:
    pip install streamlit pandas matplotlib seaborn scikit-learn

NOTE FOR BEGINNERS:
    Streamlit works like this — every time you interact with the page
    (click a button, type something), the whole script runs again from
    the top. That is totally normal. Streamlit handles this automatically.
"""

# ─── IMPORTS ──────────────────────────────────────────────────────────────────
# These are all the libraries we need. If any are missing, install with pip.

import os                          # for checking if files exist
import re                          # for finding phone numbers with regex
import json                        # for reading the ml_results.json file
import warnings                    # to hide unnecessary warnings
warnings.filterwarnings("ignore")  # keep the output clean

import streamlit as st             # the main dashboard library
import pandas as pd                # for working with data tables
import numpy as np                 # for math and arrays
import matplotlib.pyplot as plt    # for drawing charts
import matplotlib.patches as mpatches
from collections import Counter    # for counting word frequencies


# ─── PAGE SETUP ───────────────────────────────────────────────────────────────
# This must be the FIRST streamlit command in the file.
# It sets the browser tab title, icon, and layout.

st.set_page_config(
    page_title  = "SMS Spam Explorer",
    page_icon   = "📩",
    layout      = "wide",        # use full screen width
    initial_sidebar_state = "expanded"
)


# ─── COLOR CONSTANTS ──────────────────────────────────────────────────────────
# Defining colors once so we can reuse them everywhere.
# Red = spam, Green = ham — easy to remember!

SPAM_COLOR = "#E74C3C"   # red
HAM_COLOR  = "#2ECC71"   # green
MED_COLOR  = "#E67E22"   # orange (for medium spam rate)
BLUE       = "#2980B9"   # blue (for averages / model lines)


# ─── HELPER: FIGURE OUT WHERE FILES ARE ───────────────────────────────────────
# The project folder can be in different locations on different computers.
# This function tries to find files automatically.

def find_file(*possible_paths):
    """
    Try multiple file paths and return the first one that exists.
    This handles running from different folders.
    """
    for path in possible_paths:
        if os.path.exists(path):
            return path
    return None   # file not found anywhere


# ─── LOAD DATA ────────────────────────────────────────────────────────────────
# @st.cache_data tells streamlit: "load this only once, then remember it".
# Without this, the data would be loaded again on every page interaction.

@st.cache_data
def load_data():
    """Load the cleaned CSV file. Try a few possible locations."""
    csv_path = find_file(
        "OJT-SEM-2-project/spam_cleaned.csv", 
        "outputs/spam_cleaned.csv",   # when running from project root
        "spam_cleaned.csv",           # when running from outputs folder
        "../outputs/spam_cleaned.csv" # one level up
    )

    if csv_path is None:
        # Return None so we can show a helpful error message later
        return None, "Could not find spam_cleaned.csv. Run notebook 1 first."

    try:
        data = pd.read_csv(csv_path)
        return data, None   # data loaded fine, no error message
    except Exception as e:
        return None, f"Error reading CSV: {e}"


@st.cache_data
def load_ml_results():
    """Load the machine learning results JSON file."""
    json_path = find_file(
        "outputs/ml_results.json",
        "ml_results.json",
        "../outputs/ml_results.json"
    )

    if json_path is None:
        return None

    try:
        with open(json_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


# ─── LOAD SPAM CLASSIFIER MODEL (OPTIONAL) ────────────────────────────────────
# We try to load the trained model for the "Check a Message" feature.
# If it's missing, we fall back to our rule-based checker — no problem!

@st.cache_resource   # cache_resource is for heavy objects like ML models
def load_model():
    """Try to load the saved spam classifier."""
    try:
        import joblib
        model_path = find_file(
            "outputs/spam_model.pkl",
            "spam_model.pkl",
            "../outputs/spam_model.pkl"
        )
        if model_path:
            return joblib.load(model_path)
    except Exception:
        pass
    return None   # model not available


# ─── WORD CLEANING FUNCTION ───────────────────────────────────────────────────
# We need this to count words properly.
# Stopwords are common words like "the", "is", "a" that don't tell us much.

STOPWORDS = {
    "i","me","my","we","our","you","your","he","him","his","she","her",
    "it","its","they","them","this","that","am","is","are","was","were",
    "be","been","have","has","had","do","does","did","a","an","the",
    "and","but","if","or","of","at","by","for","with","to","from","in",
    "on","not","no","so","u","ur","r","ok","hi","hey","get","go","got",
    "ll","just","now","will","can","all","up","out","about","what","when",
    "how","also","then","back","more","over","da","lor","n","la","ah",
    "lol","wat","wah","k","like","know","want","good","time","come","one"
}

def clean_words(message):
    """
    Takes a message string, returns a list of meaningful words.
    Example: "FREE prize call NOW!" → ["free", "prize", "call"]
    """
    words = []
    for word in str(message).lower().split():
        # remove punctuation from start and end of each word
        word = word.strip(".,!?:;()[]\"'-")
        # keep the word only if it's not a stopword and at least 3 chars
        if word not in STOPWORDS and len(word) >= 3:
            words.append(word)
    return words


# ─── SPAM SIGNAL CHECKER ──────────────────────────────────────────────────────
# This is our rule-based system to check if a message looks like spam.
# Think of it like a checklist — more items ticked = more likely spam.

def check_signals(message):
    """
    Check a message against 9 spam signal rules.
    Returns a dictionary of signal_name → True/False.
    """
    msg_lower = message.lower()
    msg_words = msg_lower.split()

    # Rule 1: does it contain a web link?
    has_url = "http" in msg_lower or "www." in msg_lower

    # Rule 2: does it contain a phone number?
    # We look for sequences of 10 or more digits (removing spaces/dashes first)
    digits_only = re.sub(r"[\s\-\(\)]", "", message)
    has_phone   = bool(re.search(r"\d{10,}", digits_only))

    # Rule 3: does it mention prizes, winning, or cash?
    prize_words = {"prize", "cash", "win", "won", "winner", "reward", "award"}
    has_prize   = bool(prize_words & set(msg_words))

    # Rule 4: does it use the word FREE?
    has_free = "free" in msg_words

    # Rule 5: does it urge you to CALL?
    has_call = "call" in msg_words

    # Rule 6: does it ask you to TXT or TEXT?
    has_txt = "txt" in msg_words or "text" in msg_words

    # Rule 7: does it create urgency?
    urgency_words = {"urgent", "urgently", "claim", "expire", "expires",
                     "expiry", "immediately", "hurry", "limited", "act"}
    has_urgency = bool(urgency_words & set(msg_words))

    # Rule 8: is the message long? (over 100 characters)
    is_long = len(message) > 100

    # Rule 9: does it use exclamation marks?
    has_exclamation = "!" in message

    return {
        "Has a URL (http / www)"            : has_url,
        "Has a phone number (10+ digits)"   : has_phone,
        "Has prize words (prize/cash/win)"  : has_prize,
        "Has the word FREE"                 : has_free,
        "Has the word CALL"                 : has_call,
        "Has TXT or TEXT"                   : has_txt,
        "Has urgency words (urgent/claim)"  : has_urgency,
        "Message is long (> 100 chars)"     : is_long,
        "Has exclamation mark (!)"          : has_exclamation,
    }


def spam_verdict(signals, ml_model=None, message=""):
    """
    Given the signal dictionary, return a verdict: SPAM / LIKELY SPAM / SAFE.
    If an ML model is available, use that instead for better accuracy.
    """
    # Try ML model first — it's more accurate than counting signals
    if ml_model is not None and message:
        try:
            prob = ml_model.predict_proba([message])[0][1]   # probability of spam
            if prob >= 0.70:
                return "SPAM", prob, "ml"
            elif prob >= 0.35:
                return "LIKELY SPAM", prob, "ml"
            else:
                return "SAFE", prob, "ml"
        except Exception:
            pass  # if model fails, fall through to rule-based

    # Rule-based fallback
    score = sum(signals.values())   # count how many signals triggered
    if score >= 3:
        return "SPAM", score, "rules"
    elif score == 2:
        return "LIKELY SPAM", score, "rules"
    else:
        return "SAFE", score, "rules"


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN DASHBOARD — starts here
# ══════════════════════════════════════════════════════════════════════════════

# ─── LOAD DATA ────────────────────────────────────────────────────────────────
data, data_error = load_data()
ml_results       = load_ml_results()
spam_model       = load_model()

# ─── HEADER ───────────────────────────────────────────────────────────────────
st.title("📩 SMS Spam Data Exploration Dashboard")
st.markdown(
    "**Alok Chauhan** (251810700318)  ·  **Aman Kumar** (251810700231)  ·  Batch 2C"
)
st.markdown("---")

# Stop here if data didn't load
if data is None:
    st.error(f"❌ Data Error: {data_error}")
    st.info("Make sure you have run **Notebook 1** (01_data_cleaning.ipynb) first.")
    st.stop()

# ─── SIDEBAR NAVIGATION ───────────────────────────────────────────────────────
# The sidebar lets users switch between sections of the dashboard.

st.sidebar.title("📌 Navigation")
st.sidebar.markdown("Choose a section below:")

page = st.sidebar.radio(
    label   = "Go to",
    options = [
        "🏠  Overview",
        "📊  EDA Charts",
        "🔤  Word Analysis",
        "📏  Segmentation",
        "🤖  ML Model Results",
        "🔍  Check a Message",
    ]
)

st.sidebar.markdown("---")
st.sidebar.markdown(f"**Dataset:** {len(data):,} messages loaded")
if spam_model:
    st.sidebar.success("✅ ML model loaded")
else:
    st.sidebar.warning("⚠ ML model not found\n(using rule-based checker)")

# Prepare spam/ham splits (used in multiple sections)
spam = data[data["label"] == "spam"]
ham  = data[data["label"] == "ham"]


# ══════════════════════════════════════════════════════════════════════════════
#  PAGE 1 — OVERVIEW
# ══════════════════════════════════════════════════════════════════════════════

if page == "🏠  Overview":
    st.header("🏠 Project Overview")

    # ── Key Numbers (metric cards) ────────────────────────────────────────────
    # st.metric shows a big number with a label — great for summary stats.

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        label = "Total Messages",
        value = f"{len(data):,}"
    )
    col2.metric(
        label = "Spam Messages",
        value = f"{len(spam):,}",
        delta = f"{len(spam)/len(data)*100:.1f}% of total",
        delta_color = "inverse"   # red because spam is bad
    )
    col3.metric(
        label = "Ham (Normal) Messages",
        value = f"{len(ham):,}",
        delta = f"{len(ham)/len(data)*100:.1f}% of total"
    )
    col4.metric(
        label = "Features Extracted",
        value = "9"
    )

    st.markdown("---")

    # ── About Section ─────────────────────────────────────────────────────────
    col_left, col_right = st.columns([1, 1])

    with col_left:
        st.subheader("📌 About This Project")
        st.markdown("""
        This dashboard presents the results of our **SMS Spam Data Exploration** project
        using the **UCI SMS Spam Collection Dataset**.

        **What we did:**
        - Cleaned the raw data (removed duplicates, handled missing values)
        - Extracted features from each message (URL, phone number, keywords, length)
        - Compared spam vs ham across multiple dimensions
        - Found patterns that distinguish spam from normal messages
        - Built and compared 4 machine learning classifiers

        **Key finding:** Messages with **3 or more spam signals** are almost
        always spam (near 100% accuracy using simple rules).
        """)

    with col_right:
        st.subheader("🎯 Top 5 Insights")
        st.markdown(f"""
        1. 🔴 **Phone numbers** are the strongest signal — spam messages contain
           them far more often than ham.

        2. 🔗 **URLs** are {(spam["has_url"].mean()/max(ham["has_url"].mean(),0.001)):.0f}x
           more common in spam than in ham.

        3. 📏 **Message length** — spam messages average **{spam["char_count"].mean():.0f} chars**
           vs only **{ham["char_count"].mean():.0f} chars** for ham.

        4. 📊 Messages between **101–160 chars** have the highest spam rate (~40%)
           because spammers try to fit everything in one SMS (160 char limit).

        5. 🤖 **Linear SVM** was our best classifier with **95.1% F1 score**.
        """)

    st.markdown("---")

    # ── Data Quality Chart ────────────────────────────────────────────────────
    st.subheader("📋 Data Quality Summary")

    quality_img = find_file(
        "outputs/01_quality_chart.png",
        "outputs/previews/01_quality_chart.png",
        "01_quality_chart.png"
    )

    if quality_img:
        st.image(quality_img, use_container_width=True)
    else:
        # Draw it ourselves if the PNG is missing
        fig, ax = plt.subplots(figsize=(6, 3))
        categories = ["Raw\n(5572)", "No Duplicates\n(5169)", f"Final\n({len(data)})"]
        values     = [5572, 5169, len(data)]
        colors_bar = ["gray", "orange", "green"]
        bars = ax.bar(categories, values, color=colors_bar)
        ax.set_title("Row Count Before vs After Cleaning")
        ax.set_ylabel("Number of Rows")
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width()/2,
                    bar.get_height() + 30,
                    str(val), ha="center", fontweight="bold")
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    # Show a preview of the data
    st.subheader("🔍 Sample Data (first 10 rows)")
    # Only show the most useful columns
    display_cols = [c for c in ["label", "message", "char_count", "word_count",
                                "spam_signals", "has_url", "has_phone"]
                    if c in data.columns]
    st.dataframe(data[display_cols].head(10), use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
#  PAGE 2 — EDA CHARTS
# ══════════════════════════════════════════════════════════════════════════════

elif page == "📊  EDA Charts":
    st.header("📊 Exploratory Data Analysis")
    st.markdown("These charts help us understand the patterns in spam vs ham messages.")

    # ── Chart: Spam vs Ham Distribution ──────────────────────────────────────
    st.subheader("1. Spam vs Ham Distribution")

    col1, col2 = st.columns([1, 1])

    with col1:
        # Pie chart
        fig, ax = plt.subplots(figsize=(5, 5))
        sizes  = [len(spam), len(ham)]
        labels = [
            f"Spam\n{len(spam):,} msgs\n({len(spam)/len(data)*100:.1f}%)",
            f"Ham\n{len(ham):,} msgs\n({len(ham)/len(data)*100:.1f}%)"
        ]
        ax.pie(
            sizes, labels=labels,
            colors=[SPAM_COLOR, HAM_COLOR],
            startangle=90,
            textprops={"fontsize": 11, "fontweight": "bold"},
            wedgeprops={"edgecolor": "white", "linewidth": 2}
        )
        ax.set_title("Spam vs Ham Distribution", fontsize=13, fontweight="bold")
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    with col2:
        st.markdown("""
        **What this shows:**

        The dataset has a **class imbalance** — there are many more normal (ham)
        messages than spam messages. This is realistic because in real life,
        most messages we receive are legitimate.

        - **Ham:** 87.4% of all messages
        - **Spam:** only 12.6%

        This imbalance is called **class imbalance** in data science.
        It means we need to be careful — a model that just says "everything is ham"
        would already be 87% accurate! So we use **F1 score** instead of
        just accuracy to properly evaluate our models.
        """)

    st.markdown("---")

    # ── Chart: Message Length ─────────────────────────────────────────────────
    st.subheader("2. Message Length — How Long Are Spam vs Ham Messages?")

    if "char_count" in data.columns:
        fig, axes = plt.subplots(1, 2, figsize=(14, 4.5))

        # Left: Histogram
        for label, color in [("spam", SPAM_COLOR), ("ham", HAM_COLOR)]:
            d = data[data["label"] == label]["char_count"]
            axes[0].hist(d, bins=50, alpha=0.65, color=color,
                         label=label.capitalize(), density=True)

        sp_med = spam["char_count"].median()
        hm_med = ham["char_count"].median()
        axes[0].axvline(sp_med, color=SPAM_COLOR, linestyle="--", lw=2,
                        label=f"Spam median: {sp_med:.0f} chars")
        axes[0].axvline(hm_med, color=HAM_COLOR,  linestyle="--", lw=2,
                        label=f"Ham median:  {hm_med:.0f} chars")
        axes[0].set_title("Character Count Distribution", fontweight="bold")
        axes[0].set_xlabel("Number of Characters")
        axes[0].set_ylabel("Density (proportion of messages)")
        axes[0].legend()

        # Right: Box plot
        bp = axes[1].boxplot(
            [spam["char_count"], ham["char_count"]],
            tick_labels=["Spam", "Ham"],
            patch_artist=True, notch=True, widths=0.45,
            medianprops={"color": "black", "linewidth": 2.5}
        )
        bp["boxes"][0].set_facecolor(SPAM_COLOR); bp["boxes"][0].set_alpha(0.75)
        bp["boxes"][1].set_facecolor(HAM_COLOR);  bp["boxes"][1].set_alpha(0.75)
        axes[1].set_title("Character Count Box Plot", fontweight="bold")
        axes[1].set_ylabel("Number of Characters")
        axes[1].yaxis.grid(True, linestyle="--", alpha=0.5)

        plt.suptitle("How Long Are Spam vs Ham Messages?",
                     fontsize=13, fontweight="bold")
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

        col1, col2 = st.columns(2)
        col1.metric("Spam median length", f"{sp_med:.0f} chars")
        col2.metric("Ham median length",  f"{hm_med:.0f} chars",
                    delta=f"{sp_med-hm_med:.0f} chars shorter",
                    delta_color="normal")
    else:
        st.warning("char_count column not found. Did notebook 1 run fully?")

    st.markdown("---")

    # ── Chart: Feature Prevalence ─────────────────────────────────────────────
    st.subheader("3. How Often Does Each Feature Appear in Spam vs Ham?")

    feature_cols  = ["has_url", "has_number", "has_currency",
                     "has_free", "has_call", "has_txt"]
    feature_names = ["Has URL", "Has Number", "Has Prize Word",
                     "Has 'FREE'", "Has 'CALL'", "Has 'TXT/TEXT'"]

    # Only plot features that actually exist in the dataset
    valid_features = [(name, col) for name, col in zip(feature_names, feature_cols)
                      if col in data.columns]

    if valid_features:
        names = [n for n, c in valid_features]
        cols  = [c for n, c in valid_features]
        srates = [spam[c].mean() * 100 for c in cols]
        hrates = [ham[c].mean()  * 100 for c in cols]

        x     = np.arange(len(names))
        width = 0.38

        fig, ax = plt.subplots(figsize=(12, 5))
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        bars_s = ax.bar(x - width/2, srates, width,
                        label="Spam", color=SPAM_COLOR, alpha=0.85, zorder=3)
        bars_h = ax.bar(x + width/2, hrates, width,
                        label="Ham",  color=HAM_COLOR,  alpha=0.85, zorder=3)

        # Add percentage labels on top of bars
        for bar in bars_s:
            h = bar.get_height()
            if h > 0.5:
                ax.text(bar.get_x() + bar.get_width()/2, h + 0.8,
                        f"{h:.1f}%", ha="center", fontsize=10,
                        fontweight="bold", color=SPAM_COLOR)
        for bar in bars_h:
            h = bar.get_height()
            if h > 0.5:
                ax.text(bar.get_x() + bar.get_width()/2, h + 0.8,
                        f"{h:.1f}%", ha="center", fontsize=10,
                        fontweight="bold", color="#27AE60")

        ax.set_title("How Often Does Each Feature Appear in Spam vs Ham Messages?",
                     fontsize=13, fontweight="bold")
        ax.set_xlabel("Feature")
        ax.set_ylabel("% of Messages Containing Feature")
        ax.set_xticks(x)
        ax.set_xticklabels(names, rotation=15, ha="right")
        ax.set_ylim(0, max(srates) * 1.2)
        ax.legend()
        ax.yaxis.grid(True, linestyle="--", alpha=0.4, zorder=0)

        plt.tight_layout()
        st.pyplot(fig)
        plt.close()
    else:
        st.warning("Feature columns not found. Did notebook 1 extract features?")


# ══════════════════════════════════════════════════════════════════════════════
#  PAGE 3 — WORD ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════

elif page == "🔤  Word Analysis":
    st.header("🔤 Word Frequency Analysis")
    st.markdown("Which words appear most in spam vs ham messages?")

    # Number of top words to show
    n_words = st.slider(
        "How many top words to show?",
        min_value=5, max_value=25, value=15
    )

    # Count words in spam and ham (this takes a few seconds)
    with st.spinner("Counting words... please wait."):
        spam_words = sum([clean_words(m) for m in spam["message"]], [])
        ham_words  = sum([clean_words(m) for m in ham["message"]],  [])
        spam_count = Counter(spam_words)
        ham_count  = Counter(ham_words)

    # ── Side by side bar charts ───────────────────────────────────────────────
    col1, col2 = st.columns(2)

    with col1:
        st.subheader(f"🔴 Top {n_words} Spam Words")
        top_spam = pd.Series(dict(spam_count.most_common(n_words))).sort_values()

        fig, ax = plt.subplots(figsize=(7, max(n_words * 0.42, 4)))
        bars = ax.barh(top_spam.index, top_spam.values,
                       color=SPAM_COLOR, alpha=0.82, edgecolor="white", height=0.65)
        for bar in bars:
            w = bar.get_width()
            ax.text(w + top_spam.values.max() * 0.01,
                    bar.get_y() + bar.get_height() / 2,
                    f"{int(w):,}", va="center", fontsize=9)
        ax.set_title(f"Top {n_words} Words in Spam Messages",
                     fontsize=12, fontweight="bold")
        ax.set_xlabel("Number of Occurrences")
        ax.set_xlim(0, top_spam.values.max() * 1.15)
        ax.xaxis.grid(True, linestyle="--", alpha=0.4)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    with col2:
        st.subheader(f"🟢 Top {n_words} Ham Words")
        top_ham = pd.Series(dict(ham_count.most_common(n_words))).sort_values()

        fig, ax = plt.subplots(figsize=(7, max(n_words * 0.42, 4)))
        bars = ax.barh(top_ham.index, top_ham.values,
                       color=HAM_COLOR, alpha=0.82, edgecolor="white", height=0.65)
        for bar in bars:
            w = bar.get_width()
            ax.text(w + top_ham.values.max() * 0.01,
                    bar.get_y() + bar.get_height() / 2,
                    f"{int(w):,}", va="center", fontsize=9)
        ax.set_title(f"Top {n_words} Words in Ham Messages",
                     fontsize=12, fontweight="bold")
        ax.set_xlabel("Number of Occurrences")
        ax.set_xlim(0, top_ham.values.max() * 1.15)
        ax.xaxis.grid(True, linestyle="--", alpha=0.4)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    st.markdown("---")

    # ── Word Comparison Table ─────────────────────────────────────────────────
    st.subheader("📋 Word Comparison: How much more common in spam vs ham?")
    st.markdown(
        "The **spam ratio** tells you how many times more often a word "
        "appears in spam compared to ham. Higher = stronger spam signal."
    )

    top_spam_words = [w for w, _ in spam_count.most_common(30)]

    comparison_rows = []
    for word in top_spam_words:
        spam_freq = spam_count[word] / max(len(spam), 1)
        ham_freq  = ham_count[word]  / max(len(ham), 1)
        ratio     = spam_freq / max(ham_freq, 0.0001)   # avoid dividing by zero

        comparison_rows.append({
            "Word"              : word,
            "In Spam (count)"   : spam_count[word],
            "In Ham (count)"    : ham_count[word],
            "Spam Ratio (x more common)": round(ratio, 1)
        })

    df_compare = pd.DataFrame(comparison_rows).sort_values(
        "Spam Ratio (x more common)", ascending=False
    )

    st.dataframe(
        df_compare.style.background_gradient(
            subset=["Spam Ratio (x more common)"],
            cmap="Reds"
        ),
        use_container_width=True
    )

    st.info(
        "💡 Words like **urgent**, **prize**, **claim** appear almost exclusively "
        "in spam. These are the best single-word spam indicators!"
    )


# ══════════════════════════════════════════════════════════════════════════════
#  PAGE 4 — SEGMENTATION
# ══════════════════════════════════════════════════════════════════════════════

elif page == "📏  Segmentation":
    st.header("📏 Segmentation Analysis")
    st.markdown(
        "We divide messages into groups and check the **spam rate** in each group. "
        "This helps us find patterns and build moderation rules."
    )

    overall_spam_rate = data["label_num"].mean() * 100 if "label_num" in data.columns else 12.6

    st.metric("Overall Spam Rate (baseline)", f"{overall_spam_rate:.1f}%")
    st.markdown("---")

    # ── Segment 1: Message Length ─────────────────────────────────────────────
    st.subheader("1. Spam Rate by Message Length")

    if "char_count" in data.columns and "label_num" in data.columns:
        data_seg = data.copy()
        data_seg["length_group"] = pd.cut(
            data_seg["char_count"],
            bins=[0, 50, 100, 160, 300, 9999],
            labels=["0–50 chars", "51–100 chars", "101–160 chars",
                    "161–300 chars", "300+ chars"]
        )
        group1 = data_seg.groupby("length_group", observed=True).agg(
            total      = ("label", "count"),
            spam_count = ("label_num", "sum")
        )
        group1["spam_rate"] = (group1["spam_count"] / group1["total"] * 100).round(1)

        labels = [str(i) for i in group1.index]
        rates  = group1["spam_rate"].values
        totals = group1["total"].values
        colors = [SPAM_COLOR if r > 50 else MED_COLOR if r > 20 else HAM_COLOR
                  for r in rates]

        fig, ax = plt.subplots(figsize=(11, 5))
        bars = ax.bar(labels, rates, color=colors, width=0.5,
                      alpha=0.85, edgecolor="white", linewidth=1.2, zorder=3)
        ax.axhline(overall_spam_rate, color=BLUE, linestyle="--", lw=2.5, zorder=4,
                   label=f"Dataset average: {overall_spam_rate:.1f}%")

        y_max = max(rates) if len(rates) > 0 else 50
        for bar, rate, total in zip(bars, rates, totals):
            bx = bar.get_x() + bar.get_width() / 2
            ax.text(bx, bar.get_height() + y_max * 0.02,
                    f"{rate:.1f}%", ha="center", fontsize=13, fontweight="bold")
            ax.text(bx, -y_max * 0.07, f"n={total:,}",
                    ha="center", fontsize=10, color="grey")

        ax.set_title("Spam Rate by Message Length Group",
                     fontsize=13, fontweight="bold")
        ax.set_xlabel("Message Length Group")
        ax.set_ylabel("Spam Rate (%)")
        ax.set_ylim(-y_max * 0.12, y_max * 1.22)
        ax.yaxis.grid(True, linestyle="--", alpha=0.4, zorder=0)

        legend_handles = [
            mpatches.Patch(color=SPAM_COLOR, alpha=0.85, label="High spam (>50%)"),
            mpatches.Patch(color=MED_COLOR,  alpha=0.85, label="Medium spam (20–50%)"),
            mpatches.Patch(color=HAM_COLOR,  alpha=0.85, label="Low spam (<20%)"),
            plt.Line2D([0], [0], color=BLUE, linestyle="--", lw=2.5,
                       label=f"Average: {overall_spam_rate:.1f}%")
        ]
        ax.legend(handles=legend_handles, loc="upper right", framealpha=0.9)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

        st.success(
            "💡 **Key Finding:** Messages between 101–160 characters have a spam rate "
            "almost **3× higher** than the dataset average! This is because spammers "
            "try to squeeze their message into exactly one SMS (160 char limit)."
        )

    st.markdown("---")

    # ── Segment 2: Spam Signal Score ─────────────────────────────────────────
    st.subheader("2. Spam Rate by Spam Signal Score")
    st.markdown(
        "Each message gets a **spam signal score** — one point for each spam "
        "indicator found (URL, phone number, FREE, CALL, etc.)."
    )

    if "spam_signals" in data.columns and "label_num" in data.columns:
        group2 = data.groupby("spam_signals").agg(
            total      = ("label", "count"),
            spam_count = ("label_num", "sum")
        )
        group2["spam_rate"] = (group2["spam_count"] / group2["total"] * 100).round(1)

        # Display as table and chart side by side
        col1, col2 = st.columns([1, 1])

        with col1:
            st.dataframe(
                group2[["total", "spam_count", "spam_rate"]].rename(columns={
                    "total"      : "Total Messages",
                    "spam_count" : "Spam Count",
                    "spam_rate"  : "Spam Rate %"
                }),
                use_container_width=True
            )

        with col2:
            fig, ax = plt.subplots(figsize=(6, 4))
            labels_s = [str(i) for i in group2.index]
            rates_s  = group2["spam_rate"].values
            colors_s = [SPAM_COLOR if r > 50 else MED_COLOR if r > 20 else HAM_COLOR
                        for r in rates_s]
            ax.bar(labels_s, rates_s, color=colors_s, alpha=0.85, edgecolor="white")
            ax.axhline(overall_spam_rate, color=BLUE, linestyle="--",
                       label=f"Average: {overall_spam_rate:.1f}%")
            ax.set_title("Spam Rate by Signal Score", fontweight="bold")
            ax.set_xlabel("Number of Spam Signals")
            ax.set_ylabel("Spam Rate %")
            ax.legend()
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

        st.error(
            "🚨 **Rule Found:** Messages with **3 or more spam signals** are "
            "almost 100% spam. This single rule can block the most obvious spam!"
        )


# ══════════════════════════════════════════════════════════════════════════════
#  PAGE 5 — ML MODEL RESULTS
# ══════════════════════════════════════════════════════════════════════════════

elif page == "🤖  ML Model Results":
    st.header("🤖 Machine Learning Model Comparison")
    st.markdown(
        "We trained **4 classifiers** on the same dataset and compared their performance. "
        "All models use **TF-IDF** (Term Frequency-Inverse Document Frequency) to "
        "convert text into numbers before training."
    )

    if ml_results is None:
        st.warning(
            "ml_results.json not found. Run **train_model.py** first to generate results."
        )
        st.code("python train_model.py", language="bash")
        st.stop()

    meta = ml_results.get("_meta", {})

    # ── Dataset Split Info ────────────────────────────────────────────────────
    col1, col2, col3 = st.columns(3)
    col1.metric("Training Set", f"{meta.get('train_size', '?'):,} messages")
    col2.metric("Test Set",     f"{meta.get('test_size',  '?'):,} messages")
    col3.metric("Best Model",   meta.get("best_model", "?"))

    st.markdown("---")

    # ── Metrics Comparison Table ──────────────────────────────────────────────
    st.subheader("📊 Performance Metrics")

    model_names = meta.get("model_names", [])
    rows = []
    for name in model_names:
        r = ml_results.get(name, {})
        rows.append({
            "Model"       : name,
            "Accuracy"    : f"{r.get('accuracy', 0)*100:.2f}%",
            "Precision"   : f"{r.get('precision', 0)*100:.2f}%",
            "Recall"      : f"{r.get('recall', 0)*100:.2f}%",
            "F1 Score"    : f"{r.get('f1', 0)*100:.2f}%",
            "ROC-AUC"     : f"{r.get('roc_auc', 0):.4f}",
            "CV F1 (5-fold)" : f"{r.get('cv_f1', 0)*100:.2f}%",
        })

    df_metrics = pd.DataFrame(rows)
    st.dataframe(df_metrics, use_container_width=True)

    # ── Metric Explanations ───────────────────────────────────────────────────
    with st.expander("ℹ️ What do these metrics mean? (click to expand)"):
        st.markdown("""
        | Metric | Simple Explanation |
        |---|---|
        | **Accuracy** | % of all messages classified correctly |
        | **Precision** | Of all messages predicted as spam, how many actually were? |
        | **Recall** | Of all actual spam messages, how many did we catch? |
        | **F1 Score** | Balance between Precision and Recall (best overall metric) |
        | **ROC-AUC** | How well the model separates spam from ham (1.0 = perfect) |
        | **CV F1** | F1 score averaged over 5 test sets (more reliable) |

        > **Why not just use Accuracy?**
        > Because our dataset is imbalanced (87% ham). A model that calls
        > everything "ham" would be 87% accurate but useless! F1 score
        > accounts for this imbalance.
        """)

    st.markdown("---")

    # ── F1 Score Bar Chart ────────────────────────────────────────────────────
    st.subheader("📈 F1 Score Comparison")

    fig, ax = plt.subplots(figsize=(9, 4))
    f1_values = [ml_results.get(n, {}).get("f1", 0) * 100 for n in model_names]
    colors_ml = [SPAM_COLOR if n == meta.get("best_model") else "#95A5A6"
                 for n in model_names]

    bars = ax.bar(model_names, f1_values, color=colors_ml, alpha=0.85,
                  edgecolor="white", width=0.5)
    for bar, val in zip(bars, f1_values):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.3,
                f"{val:.2f}%", ha="center", fontweight="bold", fontsize=11)

    ax.set_title("F1 Score by Model (higher = better)", fontweight="bold")
    ax.set_ylabel("F1 Score (%)")
    ax.set_ylim(0, 105)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.yaxis.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    # ── ROC Curves ────────────────────────────────────────────────────────────
    st.subheader("📉 ROC Curves")
    st.markdown(
        "The ROC curve shows the trade-off between catching more spam (recall) "
        "and avoiding false alarms (precision). Closer to the top-left corner = better."
    )

    fig, ax = plt.subplots(figsize=(8, 5))
    curve_colors = [SPAM_COLOR, HAM_COLOR, BLUE, MED_COLOR]

    for name, color in zip(model_names, curve_colors):
        r = ml_results.get(name, {})
        fpr = r.get("roc_fpr", [])
        tpr = r.get("roc_tpr", [])
        auc = r.get("roc_auc", 0)
        if fpr and tpr:
            ax.plot(fpr, tpr, color=color, lw=2, label=f"{name} (AUC={auc:.4f})")

    # Diagonal line = random guessing
    ax.plot([0, 1], [0, 1], "k--", lw=1.5, alpha=0.5, label="Random (AUC=0.50)")
    ax.set_title("ROC Curves — All Models", fontsize=13, fontweight="bold")
    ax.set_xlabel("False Positive Rate (false alarms)")
    ax.set_ylabel("True Positive Rate (spam caught)")
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    # ── Confusion Matrices ────────────────────────────────────────────────────
    st.subheader("📋 Confusion Matrices")
    st.markdown(
        "A confusion matrix shows exactly how many messages were correctly vs "
        "incorrectly classified. Ideally, all numbers should be on the diagonal."
    )

    cols = st.columns(len(model_names))
    for col, name in zip(cols, model_names):
        cm = ml_results.get(name, {}).get("confusion_matrix", [[0, 0], [0, 0]])
        with col:
            st.markdown(f"**{name}**")
            fig, ax = plt.subplots(figsize=(3.5, 3))
            im = ax.imshow(cm, cmap="RdYlGn", aspect="auto",
                           vmin=0, vmax=max(cm[0][0], cm[1][1]))
            ax.set_xticks([0, 1]); ax.set_xticklabels(["Pred Ham", "Pred Spam"])
            ax.set_yticks([0, 1]); ax.set_yticklabels(["Actual Ham", "Actual Spam"])
            for i in range(2):
                for j in range(2):
                    ax.text(j, i, str(cm[i][j]), ha="center", va="center",
                            fontsize=16, fontweight="bold", color="black")
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()


# ══════════════════════════════════════════════════════════════════════════════
#  PAGE 6 — CHECK A MESSAGE
# ══════════════════════════════════════════════════════════════════════════════

elif page == "🔍  Check a Message":
    st.header("🔍 Check a Message for Spam")
    st.markdown(
        "Type any SMS message below and we'll check it against our spam detection rules "
        + ("and our **trained ML model**." if spam_model else
           "(rule-based system — run train_model.py for ML predictions).")
    )

    # ── Message Input ─────────────────────────────────────────────────────────
    # Example messages to make it easy to try
    example_messages = {
        "Type your own message..."  : "",
        "Obvious spam example"      : "FREE prize! Call 08001234567 NOW to claim your £1000 reward! Don't miss out. Reply STOP to opt out.",
        "Another spam example"      : "URGENT: Your account will be suspended. Click http://verify-now.com to confirm your details.",
        "Normal ham message"        : "Hey, are you coming to the party tonight? Let me know!",
        "Another ham message"       : "Sorry I missed your call. I'll call you back after 6pm.",
    }

    selected_example = st.selectbox(
        "Try an example or type your own:",
        options=list(example_messages.keys())
    )

    # Show the example text in the text area, or leave blank for custom
    default_text = example_messages[selected_example]
    user_message = st.text_area(
        "Message to check:",
        value    = default_text,
        height   = 120,
        max_chars= 1000,
        help     = "Paste or type any SMS message here"
    )

    # ── Analyse Button ────────────────────────────────────────────────────────
    if st.button("🔍 Analyse Message", type="primary", use_container_width=True):

        if not user_message.strip():
            st.warning("Please enter a message first.")
        else:
            signals = check_signals(user_message)
            verdict, score, method = spam_verdict(signals, spam_model, user_message)

            # ── Verdict Banner ────────────────────────────────────────────────
            if verdict == "SPAM":
                st.error(f"🚨 **VERDICT: {verdict}** — This message looks like spam!")
            elif verdict == "LIKELY SPAM":
                st.warning(f"⚠️ **VERDICT: {verdict}** — This message has suspicious signals.")
            else:
                st.success(f"✅ **VERDICT: {verdict}** — This message looks normal.")

            # Show how we decided
            if method == "ml":
                st.info(f"🤖 Decision made by **ML model** (spam probability: {score*100:.1f}%)")
            else:
                st.info(f"📏 Decision made by **rule-based system** ({score} signals detected)")

            st.markdown("---")

            # ── Signal Breakdown ──────────────────────────────────────────────
            st.subheader("📋 Signal Breakdown")

            # Draw the signal chart
            sig_labels = list(signals.keys())
            sig_values = [int(v) for v in signals.values()]
            bar_colors = [SPAM_COLOR if v else "#AEB6BF" for v in sig_values]

            fig, ax = plt.subplots(figsize=(9, 4))
            bars = ax.barh(sig_labels, sig_values, color=bar_colors,
                           alpha=0.85, edgecolor="white", height=0.55)
            ax.set_xlim(0, 1.6)
            ax.set_xticks([0, 1])
            ax.set_xticklabels(["Not Found", "Found"], fontsize=12)
            ax.set_title("Which Spam Signals Were Detected?",
                         fontsize=12, fontweight="bold")

            for bar, val in zip(bars, sig_values):
                label = "✔  Found" if val else "✘  Not found"
                ax.text(bar.get_width() + 0.04,
                        bar.get_y() + bar.get_height() / 2,
                        label, va="center", fontsize=10,
                        color=SPAM_COLOR if val else "grey")

            found_patch   = mpatches.Patch(color=SPAM_COLOR, alpha=0.85, label="Signal found")
            missing_patch = mpatches.Patch(color="#AEB6BF",  alpha=0.85, label="Not found")
            ax.legend(handles=[found_patch, missing_patch], loc="lower right")
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.xaxis.grid(False)

            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

            # ── Message Stats ─────────────────────────────────────────────────
            st.subheader("📐 Message Statistics")
            word_list = user_message.split()

            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Characters",        len(user_message))
            col2.metric("Words",             len(word_list))
            col3.metric("Signals Found",     sum(signals.values()))
            col4.metric("Exclamation Marks", user_message.count("!"))

            # ── How our rule pipeline works ───────────────────────────────────
            with st.expander("📖 How does the rule-based system work?"):
                st.markdown("""
                Our **rule pipeline** (from Notebook 4):

                1. **Score 3+** → 🚨 Block (almost 100% spam)
                2. **Score 2**  → ⚠️ Flag for review
                3. **Has phone number** → 🚨 Block (99.7% accurate)
                4. **Has URL** → ⚠️ Flag for review
                5. **Score 0-1** → ✅ Pass through

                The ML model (if loaded) gives a probability and is more accurate
                than simple rule counting because it considers the actual words used.
                """)


# ══════════════════════════════════════════════════════════════════════════════
#  FOOTER
# ══════════════════════════════════════════════════════════════════════════════

st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: grey; font-size: 13px;'>"
    "📩 SMS Spam Data Exploration &nbsp;|&nbsp; "
    "Alok Chauhan (251810700318) &nbsp;&amp;&nbsp; Aman Kumar (251810700231) "
    "&nbsp;|&nbsp; Batch 2C &nbsp;|&nbsp; "
    "Dataset: UCI SMS Spam Collection"
    "</div>",
    unsafe_allow_html=True
)