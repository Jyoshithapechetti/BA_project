# app.py
# Streamlit UI for Drug Reviews Analytics (no uploads, preprocessed on start)

import os
import re
import math
import itertools
from collections import Counter, defaultdict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from sklearn.metrics.pairwise import cosine_similarity
import streamlit as st
from ml import train_models, predict_text, get_metrics


# App Config

st.set_page_config(page_title="Drug Reviews Analytics", layout="wide")

DATA_DIR = "data"
DEFAULT_TRAIN = os.path.join(DATA_DIR, "drugsComTrain_raw.csv")
DEFAULT_TEST  = os.path.join(DATA_DIR, "drugsComTest_raw.csv")

# UI defaults
DEFAULT_TOPN_DRUGS = 15
DEFAULT_TOPN_SE    = 15
DEFAULT_TOP_EDGES  = 30
MIN_REVIEWS_VAR    = 30   # for stable variance

SIDE_EFFECTS = [
    "nausea","vomiting","headache","dizziness","insomnia","sleepiness","fatigue","weight gain",
    "weight loss","rash","itching","diarrhea","constipation","dry mouth","anxiety","depression",
    "palpitations","tremor","sweating","edema","cough","pain","back pain","muscle pain","joint pain"
]

# Existing single-token severity mapping (kept for compatibility)
SEVERITY_MAP = {
    # High
    "death":"High","cardiac arrest":"High","stroke":"High","seizure":"High",
    "suicidal":"High","respiratory failure":"High","liver failure":"High","kidney failure":"High",
    # Medium
    "depression":"Medium","anxiety":"Medium","bleeding":"Medium","hypertension":"Medium",
    "palpitations":"Medium","chest pain":"Medium","dizziness":"Medium","insomnia":"Medium",
    "hallucination":"Medium","edema":"Medium",
    # Low
    "nausea":"Low","vomiting":"Low","headache":"Low","fatigue":"Low","rash":"Low","itching":"Low",
    "dry mouth":"Low","diarrhea":"Low","constipation":"Low","sweating":"Low",
    "weight gain":"Low","weight loss":"Low","pain":"Low","back pain":"Low","muscle pain":"Low","joint pain":"Low","cough":"Low","tremor":"Low","sleepiness":"Low"
}

# New: phrase-level patterns for severity detection (multiword + synonyms)
# You can extend these lists as needed.
HIGH_SE_PATTERNS = [
    r"\b(anaphylaxis|anaphylactic shock)\b",
    r"\b(stevens[-\s]?johnson)\b",
    r"\b(dress syndrome)\b",
    r"\b(pancreatitis)\b",
    r"\b((heart|cardiac)\s+attack|myocardial\s+infarction|MI\b)\b",
    r"\b(stroke|cerebrovascular\s+accident|CVA\b)\b",
    r"\b(seizure|convulsion|epileptic)\b",
    r"\b(blood\s*clot|thromboembolism|pulmonary\s+embolism|PE\b|deep\s+vein\s+thrombosis|DVT\b)\b",
    r"\b(respiratory\s+failure|stopped\s+breathing|couldn'?t\s+breathe)\b",
    r"\b(liver\s+failure|hepatic\s+failure|fulminant\s+hepatitis)\b",
    r"\b(kidney\s+failure|renal\s+failure)\b",
    r"\b(suicidal\s+(thought|ideation|urge)s?|suicide\s+attempt)\b",
    r"\b(arrhythmia|torsades|ventricular\s+fibrillation|vfib|cardiac\s+arrest)\b",
    r"\b(coma|unconscious\s+for\s+\d+\s*(hours?|days?))\b",
]

MEDIUM_SE_PATTERNS = [
    r"\b(severe\s+bleeding|hemorrhage)\b",
    r"\b(chest\s+pain)\b",
    r"\b(hypertension|high\s+blood\s+pressure)\b",
    r"\b(severe\s+depression|major\s+depression)\b",
    r"\b(blackout|fainted|syncope)\b",
    r"\b(severe\s+allergic\s+reaction)\b",
    r"\b(jaundice)\b",
    r"\b(serotonin\s+syndrome)\b",
]

LOW_SE_PATTERNS = [
    r"\b(nausea|vomiting|headache|fatigue|dizziness|insomnia|itching|rash|diarrhea|constipation|dry\s+mouth|sweating|weight\s+gain|weight\s+loss)\b"
]

NEGATION_WORDS = {"no","not","never","without","none","neither","hardly","barely","seldom"}
NEGATION_WINDOW = 3  # tokens to the left to check for negation


# Helpers

def clean_df(df: pd.DataFrame) -> pd.DataFrame:
    required_cols = ["uniqueID","drugName","condition","review","rating","usefulCount"]
    for c in required_cols:
        if c not in df.columns:
            df[c] = np.nan

    # Improve HTML/entity cleaning: handle numeric entities like &#039;
    df["review"] = (
        df["review"].astype(str)
            .str.replace(r"<.*?>", " ", regex=True)         # remove HTML tags
            .str.replace(r"&#\d+;", "'", regex=True)        # numeric entities to apostrophe
            .str.replace(r"&\w+;", " ", regex=True)         # named entities
            .str.replace(r"\"{2,}", '"', regex=True)
            .str.replace(r"\s+", " ", regex=True)
            .str.strip()
    )
    df = df.dropna(subset=["review", "drugName"])
    df["rating"] = pd.to_numeric(df["rating"], errors="coerce")
    df["usefulCount"] = pd.to_numeric(df["usefulCount"], errors="coerce").fillna(0)
    df["review_len_chars"]  = df["review"].str.len()
    df["review_len_tokens"] = df["review"].str.split().str.len()
    return df

def vader_counts_series(text: str, analyzer: SentimentIntensityAnalyzer) -> pd.Series:
    pos = neg = 0
    for tok in re.findall(r"[A-Za-z]+", text.lower()):
        score = analyzer.lexicon.get(tok, 0)
        if score > 0:
            pos += 1
        elif score < 0:
            neg += 1
    return pd.Series({"pos_cnt": pos, "neg_cnt": neg})

def extract_side_effects_list(text: str, compiled_pat) -> list:
    found = re.findall(compiled_pat, text or "")
    return [f.lower() for f in found]

def build_cooccurrence(df: pd.DataFrame):
    se_reviews = df[df["se_list"].str.len() >= 2].copy()
    pair_counts = defaultdict(int)
    se_counts   = defaultdict(int)
    N = len(se_reviews)

    for lst in se_reviews["se_list"]:
        s = set(map(str.lower, lst))
        for se in s:
            se_counts[se] += 1
        for a,b in itertools.combinations(sorted(s), 2):
            pair_counts[(a,b)] += 1

    rows = []
    for (a,b), c_ab in pair_counts.items():
        c_a, c_b = se_counts[a], se_counts[b]
        pmi = math.log((c_ab / max(N,1)) / ((c_a / max(N,1)) * (c_b / max(N,1)) + 1e-12) + 1e-12)
        jacc = c_ab / (c_a + c_b - c_ab)
        rows.append((a,b,c_ab,pmi,jacc))
    cooc_df = pd.DataFrame(rows, columns=["a","b","count","PMI","Jaccard"]).sort_values("count", ascending=False)
    return cooc_df

def drug_se_matrix(df: pd.DataFrame):
    tmp = df[["drugName","se_list"]].explode("se_list").rename(columns={"se_list":"side_effect"})
    tmp["side_effect"] = tmp["side_effect"].astype(str).str.strip().str.lower()
    tmp = tmp[(tmp["side_effect"].ne("")) & (tmp["side_effect"].ne("nan"))]
    mat = pd.crosstab(tmp["drugName"], tmp["side_effect"]).astype(float)
    return mat

# ---- New: phrase-level severity classification on full review text ----
def _tokenize(text: str):
    return re.findall(r"[a-zA-Z']+", text.lower())

def _has_left_negation(tokens, hit_start_idx):
    left = tokens[max(0, hit_start_idx-NEGATION_WINDOW):hit_start_idx]
    return any(t in NEGATION_WORDS for t in left)

def classify_review_severity_fulltext(text: str) -> tuple[str|None, list, list, list]:
    """
    Returns: (level, high_hits, medium_hits, low_hits)
    - level: 'High' | 'Medium' | 'Low' | None  (highest applicable)
    - *_hits: list of matched phrases (deduped)
    """
    if not isinstance(text, str) or not text:
        return None, [], [], []

    t = text.lower()
    tokens = _tokenize(t)
    high_hits, med_hits, low_hits = set(), set(), set()

    def scan(patterns, bucket):
        for pat in patterns:
            for m in re.finditer(pat, t, flags=re.IGNORECASE):
                # estimate token index by matching first word of the phrase
                first_word = re.findall(r"[a-zA-Z']+", m.group(0).lower())
                if not first_word:
                    continue
                # find first occurrence index of that word near match start
                approx_idx = len(_tokenize(t[:m.start()]))
                if not _has_left_negation(tokens, approx_idx):
                    bucket.add(m.group(0).strip().lower())

    scan(HIGH_SE_PATTERNS, high_hits)
    scan(MEDIUM_SE_PATTERNS, med_hits)
    scan(LOW_SE_PATTERNS, low_hits)

    level = None
    if high_hits:
        level = "High"
    elif med_hits:
        level = "Medium"
    elif low_hits:
        level = "Low"

    return level, sorted(high_hits), sorted(med_hits), sorted(low_hits)

# Backward-compatible severity from se_list + SEVERITY_MAP (fallback)
def classify_severity_from_list(se_list: list) -> str|None:
    if not se_list:
        return None
    levels = []
    for se in se_list:
        lvl = SEVERITY_MAP.get(se.lower())
        if lvl:
            levels.append(lvl)
    if "High" in levels:
        return "High"
    if "Medium" in levels:
        return "Medium"
    if "Low" in levels:
        return "Low"
    return None

def assign_drug_risk(row: pd.Series) -> str:
    if row.get("High",0) >= 0.10:
        return "High Risk"
    elif row.get("Medium",0) >= 0.20:
        return "Medium Risk"
    else:
        return "Low Risk"

def make_radar(merged: pd.DataFrame, drugs: list):
    radar_cols = ["avg_rating","rating_var","se_rate","pos_words","neg_words","len_tokens_mean","useful_mean"]

    # normalize 0-1
    norm = merged[radar_cols].copy()
    for c in radar_cols:
        mn, mx = norm[c].min(), norm[c].max()
        norm[c] = 0 if (mx-mn)==0 else (norm[c]-mn)/(mx-mn)

    labels = radar_cols
    angles = np.linspace(0, 2*np.pi, len(labels), endpoint=False)
    angles = np.concatenate([angles, [angles[0]]])

    fig = plt.figure(figsize=(6,6))
    ax = plt.subplot(111, polar=True)

    for d in drugs:
        if d not in norm.index:
            continue
        vals = norm.loc[d, labels].to_numpy()
        vals = np.concatenate([vals, [vals[0]]])
        ax.plot(angles, vals, linewidth=2, label=d)
        ax.fill(angles, vals, alpha=0.1)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_yticklabels([])
    ax.set_title("Radar: normalized metrics")
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1), fontsize=8)
    plt.tight_layout()
    return fig


# Preprocess EVERYTHING once (no uploads)

@st.cache_data(show_spinner=True)
def preprocess_all():
    # 1) Load
    if not os.path.exists(DEFAULT_TRAIN) or not os.path.exists(DEFAULT_TEST):
        raise FileNotFoundError(
            f"CSV files not found in {DATA_DIR}/. "
            f"Expected: drugsComTrain_raw.csv and drugsComTest_raw.csv"
        )
    train = pd.read_csv(DEFAULT_TRAIN)
    test  = pd.read_csv(DEFAULT_TEST)
    df_raw = pd.concat([train.assign(source="train"),
                        test.assign(source="test")], ignore_index=True)

    # 2) Clean
    df = clean_df(df_raw.copy())

    # 3) Sentiment counts
    analyzer = SentimentIntensityAnalyzer()
    cnts = df["review"].apply(lambda t: vader_counts_series(t, analyzer))
    df = pd.concat([df, cnts], axis=1)

    # 4) Per-drug aggregation
    drug_sent = df.groupby("drugName", as_index=False).agg(
        avg_rating=("rating", "mean"),
        rating_var=("rating", "var"),
        pos_words=("pos_cnt","sum"),
        neg_words=("neg_cnt","sum"),
        n_reviews=("review","count"),
        useful_mean=("usefulCount","mean"),
        len_tokens_mean=("review_len_tokens","mean")
    ).fillna({"rating_var":0})

    # 5) Side-effects extraction (dictionary-based features)
    compiled_pat = re.compile(r"\b(" + "|".join(map(re.escape, SIDE_EFFECTS)) + r")\b", flags=re.I)
    df["se_list"] = df["review"].apply(lambda t: extract_side_effects_list(t, compiled_pat))

    # 6) Co-occurrence
    cooc_df = build_cooccurrence(df)

    # 7) Drug x SE + similarity
    drug_se_mat = drug_se_matrix(df)
    sim_matrix = pd.DataFrame()
    drug_se_norm = pd.DataFrame()
    if drug_se_mat.shape[0] > 0 and drug_se_mat.shape[1] > 0:
        n_reviews = df.groupby("drugName")["review"].count()
        drug_se_norm = (drug_se_mat.T / n_reviews).T.fillna(0)
        X = drug_se_norm.to_numpy(dtype=float)
        sim_matrix = pd.DataFrame(cosine_similarity(X), index=drug_se_norm.index, columns=drug_se_norm.index)

    # 7b) Boolean presence matrix
    drug_se_bool = (drug_se_mat > 0).astype(int)

    # 8) Severity & risk
    # 8a) Phrase-level severity per review
    sev_res = df["review"].apply(classify_review_severity_fulltext)
    df["severity_text"], df["severity_high_hits"], df["severity_med_hits"], df["severity_low_hits"] = zip(*sev_res)

    # 8b) Legacy list-based severity (fallback) if text-level None
    df["severity_list"] = df["se_list"].apply(classify_severity_from_list)

    # 8c) Final review severity = text-level if present else list-based
    def pick_final_sev(row):
        return row["severity_text"] if pd.notna(row["severity_text"]) and row["severity_text"] else row["severity_list"]
    df["severity"] = df.apply(pick_final_sev, axis=1)

    severity_summary = df.groupby("drugName")["severity"].value_counts(normalize=True).unstack().fillna(0)
    severity_summary["drug_risk_level"] = severity_summary.apply(assign_drug_risk, axis=1)

    # 9) Variance table
    variance_tbl = drug_sent[["drugName","rating_var","n_reviews"]].dropna()
    variance_tbl = variance_tbl[variance_tbl["n_reviews"] >= MIN_REVIEWS_VAR].sort_values("rating_var", ascending=False)

    # 10) Derived lists
    drug_list = sorted(drug_sent["drugName"].unique().tolist())
    top_by_reviews = drug_sent.sort_values("n_reviews", ascending=False).head(DEFAULT_TOPN_DRUGS)["drugName"].tolist()

    return {
        "df": df,
        "drug_sent": drug_sent,
        "cooc_df": cooc_df,
        "drug_se_mat": drug_se_mat,
        "drug_se_norm": drug_se_norm,
        "drug_se_bool": drug_se_bool,
        "sim_matrix": sim_matrix,
        "severity_summary": severity_summary,
        "variance_tbl": variance_tbl,
        "drug_list": drug_list,
        "top_by_reviews": top_by_reviews
    }


# Sidebar: Controls

st.sidebar.header("Controls")
TOPN_DRUGS = st.sidebar.slider("Top-N drugs to show", 5, 50, DEFAULT_TOPN_DRUGS, 1)
TOPN_SE    = st.sidebar.slider("Top-N side effects in bars", 5, 50, DEFAULT_TOPN_SE, 1)
TOP_EDGES  = st.sidebar.slider("Top co-occurrence pairs", 10, 200, DEFAULT_TOP_EDGES, 5)
# Drug–drug pairing controls
PAIR_TOPK       = st.sidebar.slider("Top drug pairs to show", 5, 100, 25, 5)
PAIR_SIM_THR    = st.sidebar.slider("Min cosine similarity (drug–drug)", 0.0, 1.0, 0.6, 0.05)
PAIR_MIN_SHARED = st.sidebar.slider("Min shared side-effects", 1, 20, 3, 1)
PAIR_TOP_SE_IN_PAIR = st.sidebar.slider("Top shared side-effects per pair", 3, 20, 8, 1)

st.sidebar.caption("Severity rules: High≥10% High-SE mentions; else Medium≥20%; else Low. Text-level detector uses medical phrases + negation.")


# Run preprocessing once, then render

with st.spinner("Preprocessing…"):
    data = preprocess_all()

df               = data["df"]
drug_sent        = data["drug_sent"]
cooc_df          = data["cooc_df"]
drug_se_mat      = data["drug_se_mat"]
drug_se_norm     = data["drug_se_norm"]
drug_se_bool     = data["drug_se_bool"]
sim_matrix       = data["sim_matrix"]
severity_summary = data["severity_summary"]
variance_tbl     = data["variance_tbl"]
drug_list        = data["drug_list"]
top_by_reviews   = data["top_by_reviews"]

@st.cache_resource(show_spinner=True)
def _train_cached(_df):
    # Train once per app session unless data changes
    return train_models(_df, threshold=7.0, test_size=0.2, random_state=42, max_features=50000)

with st.spinner("Training ML models (sentiment & rating)…"):
    models = _train_cached(df)
ml_metrics = get_metrics(models)


# UI Tabs

tabs = st.tabs([
    "Overview", "Drug Explorer", "SE Co-occurrence", "Drug Similarity",
    "Severity & Risk", "Variance", "Radar Compare", "ML Predictor"   
])

# --- Overview ---
with tabs[0]:
    st.header("Overview")
    col1, col2, col3, col4, col5 = st.columns(5)
    total_reviews = int(len(df))
    avg_rating = float(df["rating"].mean() if "rating" in df.columns else np.nan)
    se_any = (df["se_list"].apply(lambda x: len(x)>0).mean()) if "se_list" in df.columns else 0.0

    col1.metric("Total Reviews", f"{total_reviews:,}")
    col2.metric("Avg Rating", f"{avg_rating:.2f}" if not np.isnan(avg_rating) else "N/A")
    col3.metric("% reviews w/ side-effect", f"{se_any*100:.1f}%")
    col4.metric("Drugs", f"{df['drugName'].nunique():,}")
    col5.metric("Conditions", f"{df['condition'].nunique():,}" if "condition" in df.columns else "N/A")

    # st.subheader("Missing Values per Column")
    # fig = plt.figure(figsize=(8,3))
    # df.isnull().sum().plot(kind="bar")
    # plt.ylabel("Count")
    # st.pyplot(fig)

    st.subheader("Distribution of Review Lengths (tokens)")
    fig = plt.figure(figsize=(8,3))
    plt.hist(df["review_len_tokens"].clip(upper=500), bins=50)
    plt.xlabel("Token Count (capped at 500)")
    plt.ylabel("Number of Reviews")
    st.pyplot(fig)

# --- Drug Explorer ---
with tabs[1]:
    st.header("Drug Explorer")
    sel_drug = st.selectbox("Select a drug", options=drug_list, index=0 if drug_list else None)
    if sel_drug:
        st.subheader(f"Common Side Effects for {sel_drug}")
        se_counts = Counter([se for lst in df[df["drugName"]==sel_drug]["se_list"] for se in lst])
        se_series = pd.Series(se_counts).sort_values(ascending=False).head(TOPN_SE)
        if len(se_series) > 0:
            fig = plt.figure(figsize=(8,3))
            se_series.plot(kind="bar")
            plt.ylabel("Count")
            plt.xticks(rotation=45, ha="right")
            st.pyplot(fig)
        else:
            st.info("No side effects found for this drug (based on current dictionary).")

        st.subheader("Positive vs Negative Word Totals")
        row = drug_sent.set_index("drugName").loc[sel_drug, ["pos_words","neg_words"]]
        fig = plt.figure(figsize=(4,3))
        plt.bar(["Positive","Negative"], [row["pos_words"], row["neg_words"]])
        st.pyplot(fig)

        drow = drug_sent.set_index("drugName").loc[sel_drug]
        c1,c2,c3,c4 = st.columns(4)
        c1.metric("Avg Rating", f"{drow['avg_rating']:.2f}")
        c2.metric("Rating Variance", f"{drow['rating_var']:.3f}")
        c3.metric("#Reviews", int(drow["n_reviews"]))
        c4.metric("Mean Useful", f"{drow['useful_mean']:.2f}")

# --- SE Co-occurrence ---
with tabs[2]:
    st.header("Side-Effect Co-occurrence")
    if len(cooc_df) == 0:
        st.info("No co-occurrence available (not enough side-effect mentions).")
    else:
        # Table of top pairs
        st.subheader(f"Top {TOP_EDGES} co-occurring pairs (by count)")
        st.dataframe(cooc_df.head(TOP_EDGES))

        # Horizontal bar chart of top pairs by count
        top_pairs = cooc_df.head(TOP_EDGES).copy()
        top_pairs["pair"] = top_pairs["a"] + " — " + top_pairs["b"]
        fig = plt.figure(figsize=(10, max(4, int(TOP_EDGES*0.25))))
        plt.barh(top_pairs["pair"][::-1], top_pairs["count"][::-1])
        plt.xlabel("Count")
        plt.title("Top Side-Effect Pairs (by count)")
        st.pyplot(fig)

        # --- Drug–Drug pairs with similar side-effect profiles
        st.subheader("Drugs with Similar Side-Effect Profiles (paired)")
        if sim_matrix.shape[0] == 0 or drug_se_mat.shape[1] == 0:
            st.info("Not enough side-effect/similarity data to compute drug pairs.")
        else:
            # limit to top by mentions (for speed/readability)
            mention_totals = drug_se_mat.sum(axis=1).sort_values(ascending=False)
            cap = min(150, len(mention_totals))
            cand_drugs = mention_totals.index[:cap]

            pairs = []
            cols = drug_se_mat.columns

            for i, d1 in enumerate(cand_drugs):
                v1_presence = (drug_se_mat.loc[d1] > 0)
                for j in range(i+1, len(cand_drugs)):
                    d2 = cand_drugs[j]
                    sim = float(sim_matrix.loc[d1, d2])

                    if sim < PAIR_SIM_THR:
                        continue

                    shared_mask = v1_presence & (drug_se_mat.loc[d2] > 0)
                    shared_se = cols[shared_mask.values]
                    shared_n = len(shared_se)
                    if shared_n < PAIR_MIN_SHARED:
                        continue

                    comb_score = (drug_se_norm.loc[d1, shared_se] + drug_se_norm.loc[d2, shared_se]).sort_values(ascending=False)
                    top_shared = comb_score.index.tolist()[:PAIR_TOP_SE_IN_PAIR]

                    pairs.append({
                        "Drug A": d1,
                        "Drug B": d2,
                        "Similarity": round(sim, 3),
                        "#Shared SE": shared_n,
                        "Top shared SE": ", ".join(top_shared)
                    })

            if not pairs:
                st.info("No drug pairs met the current thresholds. Try lowering similarity or shared-SE minimum.")
            else:
                pairs_df = pd.DataFrame(pairs).sort_values(
                    by=["Similarity", "#Shared SE"], ascending=[False, False]
                ).head(PAIR_TOPK)

                st.dataframe(pairs_df, use_container_width=True)

                # Quick viz — top pairs by #shared SE
                st.caption("Top pairs by number of shared side-effects")
                fig2 = plt.figure(figsize=(10, max(3, int(len(pairs_df)*0.35))))
                lbls = (pairs_df["Drug A"] + " — " + pairs_df["Drug B"]).iloc[::-1]
                vals = pairs_df["#Shared SE"].iloc[::-1]
                plt.barh(lbls, vals)
                plt.xlabel("Count of Shared Side-Effects")
                st.pyplot(fig2)

                st.caption("Note: Similarity = cosine over normalized SE frequencies; shared SE ranking = combined normalized frequency within the pair.")

# --- Drug Similarity ---
with tabs[3]:
    st.header("Drug Similarity (by side-effect profiles)")
    if sim_matrix.shape[0] == 0:
        st.info("Similarity matrix is empty (no side-effects found).")
    else:
        mention_totals = drug_se_mat.sum(axis=1) if drug_se_mat.shape[1] > 0 else pd.Series(0, index=sim_matrix.index)
        subset = mention_totals.sort_values(ascending=False).head(TOPN_DRUGS).index
        S = sim_matrix.loc[subset, subset].to_numpy()

        st.subheader(f"Similarity Heatmap (Top {TOPN_DRUGS} by SE mentions)")
        fig = plt.figure(figsize=(9,6))
        plt.imshow(S, interpolation="nearest", aspect="auto")
        plt.xticks(ticks=np.arange(len(subset)), labels=subset, rotation=90)
        plt.yticks(ticks=np.arange(len(subset)), labels=subset)
        plt.colorbar(label="Cosine similarity")
        st.pyplot(fig)

# --- Severity & Risk ---
with tabs[4]:
    st.header("Severity & Risk")
    if "drug_risk_level" not in severity_summary.columns or severity_summary.shape[0] == 0:
        st.info("No severity data available.")
    else:
        st.subheader("Risk Level Counts")

        # original counts
        rc = severity_summary["drug_risk_level"].value_counts()
        # 🔹 scale counts by 36
        rc_scaled = rc * 36

        fig = plt.figure(figsize=(6,3))
        plt.bar(rc_scaled.index, rc_scaled.values)
        plt.ylabel("Count (scaled ×36)")
        st.pyplot(fig)

        st.subheader(f"Severity Distribution per Drug (Top {DEFAULT_TOPN_DRUGS} by reviews)")
        top_drugs = drug_sent.sort_values("n_reviews", ascending=False).head(DEFAULT_TOPN_DRUGS)["drugName"]
        has_cols = set(["High","Medium","Low"]).issubset(severity_summary.columns)
        sev_plot = severity_summary.loc[top_drugs, ["High","Medium","Low"]].fillna(0) if has_cols else pd.DataFrame()
        if sev_plot.shape[0] > 0:
            fig = plt.figure(figsize=(10,4))
            bottom = np.zeros(sev_plot.shape[0])
            for col in ["High","Medium","Low"]:
                plt.bar(sev_plot.index, sev_plot[col].values, bottom=bottom, label=col)
                bottom += sev_plot[col].values
            plt.xticks(rotation=45, ha="right")
            plt.legend()
            st.pyplot(fig)
        else:
            st.info("Not enough severity data to show stacked bars.")

# --- Variance ---
with tabs[5]:
    st.header("Rating Variance (mixed vs consistent)")
    if variance_tbl.shape[0] == 0:
        st.info("Variance table is empty (need enough reviews per drug).")
    else:
        st.caption(f"Showing drugs with at least {MIN_REVIEWS_VAR} reviews.")
        top_var = variance_tbl.head(20).set_index("drugName")["rating_var"]
        fig = plt.figure(figsize=(10,4))
        top_var.plot(kind="bar")
        plt.ylabel("Variance")
        plt.xticks(rotation=45, ha="right")
        st.pyplot(fig)

# --- Radar Compare ---
with tabs[6]:
    st.header("Radar Compare (select up to 3 drugs)")
    se_rate_series = df.groupby("drugName")["se_list"].apply(lambda x: (sum(1 for lst in x if len(lst)>0)/len(x)) if len(x)>0 else 0)
    merged = drug_sent.set_index("drugName").copy()
    merged["se_rate"] = se_rate_series.reindex(merged.index).fillna(0)
    merged["pos_words"] = merged["pos_words"] / merged["n_reviews"].replace(0, np.nan)
    merged["neg_words"] = merged["neg_words"] / merged["n_reviews"].replace(0, np.nan)
    merged = merged.fillna(0)

    sel = st.multiselect("Choose up to 3 drugs", options=drug_list, default=top_by_reviews[:2], max_selections=3)
    if len(sel) > 0:
        fig = make_radar(merged, sel)
        st.pyplot(fig)
    else:
        st.info("Select at least one drug to plot the radar.")



with tabs[7]:
    st.header("ML Predictor")

    # --- Metrics panel ---
    st.subheader("Hold-out Metrics")
    c1, c2, c3, c4 = st.columns(4)
    cls = ml_metrics["classification"]
    c1.metric("Accuracy", f'{cls["accuracy"]:.3f}')
    c2.metric("Precision", f'{cls["precision"]:.3f}')
    c3.metric("Recall", f'{cls["recall"]:.3f}')
    c4.metric("F1", f'{cls["f1"]:.3f}')
    if cls.get("roc_auc") is not None:
        st.caption(f'ROC AUC: {cls["roc_auc"]:.3f}  |  Test n={cls["n_test"]}')
    else:
        st.caption(f'Test n={cls["n_test"]}  |  ROC AUC not available for this solver')

    reg = ml_metrics["regression"]
    r1, r2, r3 = st.columns(3)
    r1.metric("RMSE", f'{reg["rmse"]:.3f}')
    r2.metric("MAE",  f'{reg["mae"]:.3f}')
    r3.metric("R²",   f'{reg["r2"]:.3f}')

    with st.expander("Confusion Matrix (tn, fp / fn, tp)"):
        cm = np.array(cls["confusion_matrix"])
        st.write(pd.DataFrame(cm, index=["Actual 0 (Neg)", "Actual 1 (Pos)"],
                                  columns=["Pred 0 (Neg)", "Pred 1 (Pos)"]))

    st.divider()

    # --- Prediction box ---
    st.subheader("Try a sentence")
    example = "This drug worked really well for me"
    user_text = st.text_area("Enter a review sentence/paragraph:", value=example, height=120)
    if st.button("Predict sentiment & rating"):
        res = predict_text(user_text, models)
        if res["label"] is None:
            st.warning("Please enter some text.")
        else:
            pcol1, pcol2, pcol3 = st.columns(3)
            pcol1.metric("Sentiment", res["label"])
            if res["prob_positive"] is not None:
                pcol2.metric("P(Positive)", f'{res["prob_positive"]:.3f}')
            else:
                pcol2.metric("P(Positive)", "N/A")
            pcol3.metric("Pred. Rating", f'{res["pred_rating"]:.2f}')


st.success("Ready.")

