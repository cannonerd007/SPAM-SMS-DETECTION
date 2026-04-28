# 📩 SMS Spam Data Exploration

**Data Analytics / Data Science Project**

---

## 📌 About the Project

Messaging systems (SMS, WhatsApp, Email, Telecom Networks) receive a large number of unwanted messages every day. These include:

* Promotional messages
* Lottery scams
* Phishing attacks
* Fraudulent offers

Most platforms rely on **blacklists and simple keyword filters**, which are:

* Easy to bypass
* Not evidence-based
* Inefficient

This project performs **Exploratory Data Analysis (EDA) and Text Mining** on the **UCI SMS Spam Collection Dataset** to discover patterns in spam messages and generate **rule-based moderation insights**.

> ❗ Important:
> This project does **NOT** build a machine learning prediction model.
> It focuses on **understanding spam behavior** rather than classification.

---

## 🎯 Objective

The goal of the project is to help moderation or trust & safety teams answer:

* What characteristics differentiate spam and normal messages?
* Do spam messages follow a consistent structure?
* Which words or patterns indicate fraud?
* What rules can a filtering system implement?

---

## 📊 Dataset

**SMS Spam Collection Dataset (UCI Repository)**

Each message is labeled as:

| Label | Meaning                       |
| ----- | ----------------------------- |
| ham   | legitimate message            |
| spam  | unwanted / fraudulent message |

Columns:

| Column  | Description      |
| ------- | ---------------- |
| label   | spam or ham      |
| message | SMS text content |

---

## 🧰 Technology Stack

### Programming

* Python

### Data Analysis

* Pandas
* NumPy

### Visualization

* Matplotlib
* Seaborn
* WordCloud

### Text Processing

* Scikit-learn (statistics only)

### Development

* Jupyter Notebook
* Git & GitHub

### Dashboard

* Streamlit / Tableau

---

# 🧠 High Level Design (HLD)

The system works as a **data analytics pipeline** that converts raw SMS messages into moderation insights.

![HLD Pipeline](docs/diagrams/hld_pipeline.png)

### HLD Modules

1. Dataset Acquisition
2. Data Cleaning & Quality Check
3. Text Feature Extraction
4. Exploratory Data Analysis
5. Visualization Dashboard
6. Insight & Moderation Report

---

# ⚙️ Low Level Design (LLD)

The Low Level Design explains how the project is structured internally in terms of files and modules.

![LLD Structure](docs/diagrams/lld_structure.png)

### Project Folder Structure

```id="n4g0l1"
SMS-Spam-Exploration/
│
├── data/
│   ├── raw/
│   └── processed/
│
├── notebooks/
│   ├── 01_data_cleaning.ipynb
│   ├── 02_eda.ipynb
│   ├── 03_text_analysis.ipynb
│   └── 04_visualization.ipynb
│
├── src/
│   ├── download_data.py
│   ├── clean_data.py
│   ├── text_features.py
│   ├── analysis.py
│   └── visualization.py
│
├── dashboard/
│   └── app.py
│
├── reports/
│   └── insight_memo.pdf
│
└── requirements.txt
```

---

## 🔍 Feature Engineering

The system extracts measurable indicators from SMS messages:

| Feature         | Meaning              |
| --------------- | -------------------- |
| char_length     | message size         |
| word_count      | number of words      |
| has_url         | link presence        |
| has_number      | phone/OTP detection  |
| has_currency    | ₹ / $ scam indicator |
| uppercase_ratio | urgency detection    |

---

# 👤 Consumer Flow Diagram (CFD)

This diagram shows how a user (moderator / teacher / evaluator) interacts with the dashboard.

![Consumer Flow](docs/diagrams/consumer_flow.png)

### User Journey

1. User opens dashboard
2. Clean dataset loads
3. Overview statistics displayed
4. User explores analysis options
5. Views wordcloud & charts
6. Reads moderation insights
7. Downloads final report

---

# 🔁 Data Flow Diagram (DFD)

## Level-0 DFD (Simplified Data Flow)

![DFD Level 0](docs/diagrams/dfd_level0.png)

This shows the movement of data from dataset to final report.

---

## Level-1 DFD (Detailed Data Flow)

![DFD Level 1](docs/diagrams/dfd_level1.png)

Detailed processes include:

### Cleaning Module

* Remove duplicates
* Normalize text
* Handle missing values

### Feature Extraction

* Word count
* URL detection
* Currency symbol detection
* Number detection
* Uppercase ratio

### Analysis Engine

* Spam vs Ham comparison
* Keyword frequency
* N-gram detection

---

# ▶️ How to Run

### 1️⃣ Clone Repository

```id="c3zttg"
git clone https://github.com/<your-username>/sms-spam-data-exploration.git
cd sms-spam-data-exploration
```

### 2️⃣ Install Requirements

```id="fl0h9e"
pip install -r requirements.txt
```

### 3️⃣ Run Notebooks

```id="qst41z"
jupyter notebook
```

### 4️⃣ Run Dashboard

```id="2gnd0m"
streamlit run dashboard/app.py
```

---

# 📄 License

This project is licensed under the **Apache License 2.0**.

You are free to:

* Use the code
* Modify the code
* Distribute the project

You must:

* Provide attribution
* Include the same license in copies

Add a file named **LICENSE** in the root directory and paste the official Apache 2.0 license text from:
https://www.apache.org/licenses/LICENSE-2.0

---

# 🙏 Acknowledgement

Dataset provided by **UCI Machine Learning Repository — SMS Spam Collection Dataset**
