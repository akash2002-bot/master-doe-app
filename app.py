# ==========================================================
# MASTER DESIGN OF EXPERIMENTS SYSTEM
# FULL 8-DESIGN STREAMLIT VERSION
# ==========================================================

import streamlit as st
import pandas as pd
import numpy as np
import io
from scipy.stats import f, t
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.stats.anova import anova_lm
import networkx as nx
import matplotlib.pyplot as plt

alpha = 0.05

# ==========================================================
# COMPATIBILITY CHECK
# ==========================================================

design_required_columns = {
    "CRD": ["Treatment", "Yield"],
    "RBD": ["Block", "Treatment", "Yield"],
    "Latin Square": ["Row", "Column", "Treatment", "Yield"],
    "General Factorial": "Flexible",
    "Split Plot": ["Rep", "A", "B", "Y"],
    "Strip Plot": ["Replication", "A", "B", "Y"],
    "BIBD": ["Block", "Treatment", "Y"],
    "SBBD": ["Block", "T1", "T2", "Yield"]
}

def check_compatibility(design_name, df):
    required = design_required_columns[design_name]
    if required == "Flexible":
        return True
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns for {design_name}: {missing}")
    return True


# ==========================================================
# CRD
# ==========================================================

def run_crd(df):

    df = df.dropna(subset=["Treatment","Yield"])
    df["Treatment"] = df["Treatment"].astype("category")

    model = smf.ols("Yield ~ C(Treatment)", data=df).fit()
    anova = sm.stats.anova_lm(model, typ=2).reset_index()
    anova.rename(columns={"index":"Source"}, inplace=True)

    means = df.groupby("Treatment")["Yield"].mean().reset_index()
    means = means.sort_values("Yield", ascending=False)

    interpretation = []
    interpretation.append("CRD — Type II ANOVA applied.")
    best = means.iloc[0]["Treatment"]
    interpretation.append(f"Best Treatment: {best}")

    return {
        "ANOVA": anova,
        "Treatment_Means": means,
        "Interpretation": interpretation
    }


# ==========================================================
# RBD
# ==========================================================

def run_rbd(df):

    df = df.dropna(subset=["Block","Treatment","Yield"])
    df["Block"] = df["Block"].astype("category")
    df["Treatment"] = df["Treatment"].astype("category")

    model = smf.ols("Yield ~ C(Treatment) + C(Block)", data=df).fit()
    anova = sm.stats.anova_lm(model, typ=2).reset_index()
    anova.rename(columns={"index":"Source"}, inplace=True)

    means = df.groupby("Treatment")["Yield"].mean().reset_index()
    means = means.sort_values("Yield", ascending=False)

    interpretation = []
    interpretation.append("RBD — Type II ANOVA applied.")
    interpretation.append(f"Best Treatment: {means.iloc[0]['Treatment']}")

    return {
        "ANOVA": anova,
        "Treatment_Means": means,
        "Interpretation": interpretation
    }


# ==========================================================
# LATIN SQUARE
# ==========================================================

def run_latin_square(df):

    df = df.dropna(subset=["Row","Column","Treatment","Yield"])
    df["Row"] = df["Row"].astype("category")
    df["Column"] = df["Column"].astype("category")
    df["Treatment"] = df["Treatment"].astype("category")

    model = smf.ols("Yield ~ C(Row)+C(Column)+C(Treatment)", data=df).fit()
    anova = sm.stats.anova_lm(model, typ=2).reset_index()
    anova.rename(columns={"index":"Source"}, inplace=True)

    means = df.groupby("Treatment")["Yield"].mean().reset_index()
    means = means.sort_values("Yield", ascending=False)

    interpretation = []
    interpretation.append("Latin Square — Type II ANOVA.")
    interpretation.append(f"Best Treatment: {means.iloc[0]['Treatment']}")

    return {
        "ANOVA": anova,
        "Treatment_Means": means,
        "Interpretation": interpretation
    }


# ==========================================================
# GENERAL FACTORIAL
# ==========================================================

def run_general_factorial(df):

    response = df.columns[-1]
    factors = df.columns[:-1].tolist()

    formula = response + " ~ " + " * ".join([f"C({f})" for f in factors])
    model = smf.ols(formula, data=df).fit()
    anova = sm.stats.anova_lm(model, typ=2).reset_index()
    anova.rename(columns={"index":"Source"}, inplace=True)

    means = df.groupby(factors)[response].mean().reset_index()
    means = means.sort_values(response, ascending=False)

    interpretation = []
    interpretation.append("General Factorial — Type II ANOVA.")
    best = means.iloc[0]
    combo = ", ".join([f"{f}={best[f]}" for f in factors])
    interpretation.append(f"Best Combination: {combo}")

    return {
        "ANOVA": anova,
        "Treatment_Means": means,
        "Interpretation": interpretation
    }


# ==========================================================
# SPLIT PLOT
# ==========================================================

def run_split_plot(df):

    df = df.dropna()
    df["Rep"] = df["Rep"].astype("category")
    df["A"] = df["A"].astype("category")
    df["B"] = df["B"].astype("category")

    model = smf.ols("Y ~ C(Rep)+C(A)+C(B)+C(A):C(B)", data=df).fit()
    anova = sm.stats.anova_lm(model, typ=2).reset_index()
    anova.rename(columns={"index":"Source"}, inplace=True)

    means = df.groupby(["A","B"])["Y"].mean().reset_index()
    means = means.sort_values("Y", ascending=False)

    interpretation = []
    interpretation.append("Split Plot — Type II ANOVA.")
    best = means.iloc[0]
    interpretation.append(f"Best Combination: A={best['A']}, B={best['B']}")

    return {
        "ANOVA": anova,
        "Treatment_Means": means,
        "Interpretation": interpretation
    }


# ==========================================================
# STRIP PLOT
# ==========================================================

def run_strip_plot(df):

    df = df.dropna()
    df["Replication"] = df["Replication"].astype("category")
    df["A"] = df["A"].astype("category")
    df["B"] = df["B"].astype("category")

    model = smf.ols("Y ~ C(Replication)+C(A)+C(B)+C(A):C(B)", data=df).fit()
    anova = sm.stats.anova_lm(model, typ=2).reset_index()
    anova.rename(columns={"index":"Source"}, inplace=True)

    means = df.groupby(["A","B"])["Y"].mean().reset_index()
    means = means.sort_values("Y", ascending=False)

    interpretation = []
    interpretation.append("Strip Plot — Type II ANOVA.")
    best = means.iloc[0]
    interpretation.append(f"Best Combination: A={best['A']}, B={best['B']}")

    return {
        "ANOVA": anova,
        "Treatment_Means": means,
        "Interpretation": interpretation
    }


# ==========================================================
# BIBD
# ==========================================================

def run_bibd(df):

    df = df.dropna()
    df["Block"] = df["Block"].astype("category")
    df["Treatment"] = df["Treatment"].astype("category")

    model = smf.ols("Y ~ C(Treatment)+C(Block)", data=df).fit()
    anova = sm.stats.anova_lm(model, typ=2).reset_index()
    anova.rename(columns={"index":"Source"}, inplace=True)

    means = df.groupby("Treatment")["Y"].mean().reset_index()
    means = means.sort_values("Y", ascending=False)

    interpretation = []
    interpretation.append("BIBD — Approximate Type II ANOVA.")
    interpretation.append(f"Best Treatment: {means.iloc[0]['Treatment']}")

    return {
        "ANOVA": anova,
        "Treatment_Means": means,
        "Interpretation": interpretation
    }


# ==========================================================
# SBBD
# ==========================================================

def run_sbbd(df):

    df = df.dropna()
    df["Block"] = df["Block"].astype("category")
    df["T1"] = df["T1"].astype("category")
    df["T2"] = df["T2"].astype("category")

    model = smf.ols("Yield ~ C(T1)*C(T2)+C(Block)", data=df).fit()
    anova = sm.stats.anova_lm(model, typ=2).reset_index()
    anova.rename(columns={"index":"Source"}, inplace=True)

    means = df.groupby(["T1","T2"])["Yield"].mean().reset_index()
    means = means.sort_values("Yield", ascending=False)

    # Graph
    G = nx.Graph()
    for _, row in df.iterrows():
        G.add_edge(row["T1"], row["T2"])

    fig, ax = plt.subplots(figsize=(6,6))
    pos = nx.spring_layout(G, seed=42)
    nx.draw(G, pos, with_labels=True, ax=ax)

    interpretation = []
    interpretation.append("SBBD — Interaction Model.")
    best = means.iloc[0]
    interpretation.append(f"Best Combination: T1={best['T1']}, T2={best['T2']}")

    return {
        "ANOVA": anova,
        "Treatment_Means": means,
        "Interpretation": interpretation,
        "Graph_Figure": fig
    }


# ==========================================================
# STREAMLIT UI
# ==========================================================

st.set_page_config(page_title="MASTER DOE SYSTEM", layout="wide")

st.title("MASTER DESIGN OF EXPERIMENTS SYSTEM")

design = st.selectbox(
    "Select Design",
    list(design_required_columns.keys())
)

uploaded_file = st.file_uploader("Upload Excel File (.xlsx)", type=["xlsx"])

if uploaded_file:

    try:
        df = pd.read_excel(uploaded_file)
        check_compatibility(design, df)

        if design == "CRD":
            result = run_crd(df)
        elif design == "RBD":
            result = run_rbd(df)
        elif design == "Latin Square":
            result = run_latin_square(df)
        elif design == "General Factorial":
            result = run_general_factorial(df)
        elif design == "Split Plot":
            result = run_split_plot(df)
        elif design == "Strip Plot":
            result = run_strip_plot(df)
        elif design == "BIBD":
            result = run_bibd(df)
        elif design == "SBBD":
            result = run_sbbd(df)

        st.subheader("ANOVA")
        st.dataframe(result["ANOVA"])

        if "Treatment_Means" in result:
            st.subheader("Means")
            st.dataframe(result["Treatment_Means"])

        st.subheader("Interpretation")
        for line in result["Interpretation"]:
            st.write("•", line)

        if "Graph_Figure" in result:
            st.subheader("SBBD Graph")
            st.pyplot(result["Graph_Figure"])

        # Download
        output = io.BytesIO()
        with pd.ExcelWriter(output, engine="openpyxl") as writer:
            for key, value in result.items():
                if isinstance(value, pd.DataFrame):
                    value.to_excel(writer, sheet_name=key, index=False)

        st.download_button(
            "Download Results",
            data=output.getvalue(),
            file_name=f"{design}_Output.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )

    except Exception as e:
        st.error(str(e))
