# -*- coding: utf-8 -*-
import json
import io
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
)

# =======================
# Config générale
# =======================
st.set_page_config(
    page_title="DiagDiabète • XGBoost",
    page_icon="🩺",
    layout="wide",
)

PRIMARY = "#2563eb"  # bleu pro
OK = "#16a34a"
WARN = "#f59e0b"
BAD = "#dc2626"
SUBTLE = "#64748b"

# =======================
# Constantes features
# =======================
FEATURES = [
    'age', 'hypertension', 'heart_disease', 'bmi', 'HbA1c_level', 'blood_glucose_level',
    'gender_Female', 'gender_Male', 'gender_Other',
    'smoking_history_current', 'smoking_history_ever', 'smoking_history_former',
    'smoking_history_never', 'smoking_history_not current', 'smoking_history_unknown'
]
NUMERIC_FEATURES = ['age', 'hypertension', 'heart_disease', 'bmi', 'HbA1c_level', 'blood_glucose_level']

CATEG_MAP_GENDER = ["Female", "Male", "Other"]
CATEG_MAP_SMOKE = ["never", "former", "current", "ever", "not current", "unknown"]

# =======================
# Chargement modèle, scaler & seuil
# =======================
@st.cache_resource
def load_assets():
    model = joblib.load("model_xgb.pkl")
    scaler = joblib.load("scaler.pkl")  # << IMPORTANT
    # Seuil par défaut plus réaliste si fichier absent
    threshold = 0.46
    try:
        with open("thresholds.json", "r") as f:
            th_all = json.load(f)
            threshold = float(th_all.get("xgb", threshold))
    except Exception:
        pass
    return model, scaler, threshold

model, scaler, THRESH = load_assets()

# =======================
# Helpers
# =======================
def one_hot_from_raw_row(age, hypertension, heart_disease, bmi, hba1c, glucose,
                         gender, smoking) -> pd.DataFrame:
    """Construit un DataFrame 1 ligne aligné sur FEATURES."""
    row = {
        'age': age,
        'hypertension': int(hypertension),
        'heart_disease': int(heart_disease),
        'bmi': float(bmi),
        'HbA1c_level': float(hba1c),
        'blood_glucose_level': float(glucose),

        # init one-hot à 0
        'gender_Female': 0, 'gender_Male': 0, 'gender_Other': 0,
        'smoking_history_current': 0, 'smoking_history_ever': 0, 'smoking_history_former': 0,
        'smoking_history_never': 0, 'smoking_history_not current': 0, 'smoking_history_unknown': 0,
    }
    gkey = f"gender_{gender}"
    if gkey in row:
        row[gkey] = 1
    skey = f"smoking_history_{smoking}"
    if skey in row:
        row[skey] = 1
    X = pd.DataFrame([row], columns=FEATURES, dtype=float)
    # scaler sur colonnes numériques uniquement
    X.loc[:, NUMERIC_FEATURES] = scaler.transform(X[NUMERIC_FEATURES])
    return X

def ensure_features_from_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Accepte:
      - format brut: ['age','hypertension','heart_disease','bmi','HbA1c_level','blood_glucose_level','gender','smoking_history']
      - format one-hot déjà aligné sur FEATURES
    Retourne un DataFrame aligné sur FEATURES et scaled (NUMERIC_FEATURES).
    """
    # Cas 1 : déjà one-hot complet
    if all(col in df.columns for col in FEATURES):
        X = df[FEATURES].copy()
        X.loc[:, NUMERIC_FEATURES] = scaler.transform(X[NUMERIC_FEATURES])
        return X

    # Cas 2 : format brut -> one-hot
    required_raw = ['age','hypertension','heart_disease','bmi','HbA1c_level','blood_glucose_level','gender','smoking_history']
    if not all(col in df.columns for col in required_raw):
        raise ValueError(
            "Colonnes manquantes. Fournissez soit toutes les colonnes one-hot attendues, "
            "soit le format brut : "
            f"{required_raw}"
        )

    tmp = df.copy()
    tmp['gender'] = tmp['gender'].astype(str).str.title()     # Female/Male/Other
    tmp['smoking_history'] = tmp['smoking_history'].astype(str).str.lower()

    oh = pd.DataFrame(0, index=tmp.index, columns=FEATURES, dtype=float)

    for col in NUMERIC_FEATURES:
        oh[col] = tmp[col].astype(float)

    for g in CATEG_MAP_GENDER:
        oh[f"gender_{g}"] = (tmp['gender'] == g).astype(int)

    for s in CATEG_MAP_SMOKE:
        oh[f"smoking_history_{s}"] = (tmp['smoking_history'] == s).astype(int)

    # scale
    oh.loc[:, NUMERIC_FEATURES] = scaler.transform(oh[NUMERIC_FEATURES])
    return oh[FEATURES].copy()

def predict_proba_batch(X: pd.DataFrame) -> np.ndarray:
    return model.predict_proba(X.values)[:, 1]

def predict_label_from_proba(p: float, th: float) -> int:
    return int(p >= th)

def risk_text(p: float, th: float) -> str:
    return "Diabète probable" if p >= th else "Faible risque"

# =======================
# UI – Header
# =======================
st.markdown(f"""
<h1 style="margin-bottom:0">🩺 DiagDiabète</h1>
<p style="color:{SUBTLE}; margin-top:0">
 Application Data Mining (Azure)
</p>
""", unsafe_allow_html=True)

tab_form, tab_csv = st.tabs(["🧍‍♀️ Formulaire individuel", "📁 Prédictions sur fichier CSV"])

# =======================
# Onglet 1 : Formulaire
# =======================
with tab_form:
    with st.form("form_indiv"):
        col1, col2, col3 = st.columns(3)
        with col1:
            age = st.slider("Âge (ans)", 1, 100, 45)
            bmi = st.number_input("IMC (BMI)", min_value=10.0, max_value=60.0, value=27.5, step=0.1)
            gender = st.selectbox("Sexe", CATEG_MAP_GENDER, index=1)  # Male par défaut

        with col2:
            hba1c = st.number_input("HbA1c (%)", min_value=3.5, max_value=15.0, value=5.8, step=0.1)
            glucose = st.number_input("Glycémie (mg/dL)", min_value=50.0, max_value=400.0, value=120.0, step=1.0)
            smoking = st.selectbox("Tabagisme", CATEG_MAP_SMOKE, index=0)

        with col3:
            hypertension = st.select_slider("Hypertension", options=[0,1], value=0)
            heart_disease = st.select_slider("Maladie cardiaque", options=[0,1], value=0)
            st.markdown("<br>", unsafe_allow_html=True)

        submitted = st.form_submit_button(" Lancer la prédiction")

    if submitted:
        X = one_hot_from_raw_row(age, hypertension, heart_disease, bmi, hba1c, glucose, gender, smoking)
        proba = float(model.predict_proba(X)[0,1])
        label = predict_label_from_proba(proba, THRESH)

        color = BAD if label==1 else OK
        st.markdown(f"""
        <div style="padding:14px;border:1px solid {color};border-radius:10px">
            <b>Résultat</b><br>
            Probabilité: <b>{proba:.3f}</b> 
            Verdict: <span style="color:{color};font-weight:700">{risk_text(proba, THRESH)}</span>
        </div>
        """, unsafe_allow_html=True)

        st.progress(min(max(proba,0.0), 1.0))

        with st.expander("Voir le vecteur de caractéristiques (ordre exact)"):
            st.write(pd.DataFrame(X, columns=FEATURES))

# =======================
# Onglet 2 : CSV
# =======================
with tab_csv:
    st.write("Le CSV peut être **brut** (colonnes: `age, hypertension, heart_disease, bmi, HbA1c_level, blood_glucose_level, gender, smoking_history`) "
             "ou déjà **one-hot** aligné sur les features attendues.")
    file = st.file_uploader("Déposer un fichier CSV", type=["csv"])

    if file is not None:
        try:
            df_raw = pd.read_csv(file)
            total_rows = len(df_raw)
            st.success(f"Fichier chargé ({total_rows} lignes).")

            X = ensure_features_from_df(df_raw)   # << applique aussi le scaler
            proba = predict_proba_batch(X)
            pred = (proba >= THRESH).astype(int)

            out = df_raw.copy()
            out["probability"] = proba
            out["prediction"] = pred

            colA, colB, colC = st.columns(3)
            with colA:
                st.metric("Positifs prédits", int(pred.sum()))
            with colB:
                st.metric("Taux positifs", f"{pred.mean()*100:.1f}%")
            

            fig, ax = plt.subplots(figsize=(6,3))
            ax.hist(proba, bins=30, color="#60a5fa", edgecolor="white")
            ax.set_title("Distribution des probabilités")
            ax.set_xlabel("Probabilité prédite")
            ax.set_ylabel("Nombre")
            ax.legend()
            st.pyplot(fig)

            if "diabetes" in df_raw.columns:
                y_true = df_raw["diabetes"].values.astype(int)
                try:
                    auc = roc_auc_score(y_true, proba)
                except Exception:
                    auc = np.nan
                acc = accuracy_score(y_true, pred)
                prec = precision_score(y_true, pred, zero_division=0)
                rec = recall_score(y_true, pred, zero_division=0)
                f1 = f1_score(y_true, pred, zero_division=0)
                cm = confusion_matrix(y_true, pred)

                st.subheader("📊 Métriques (si étiquettes présentes)")
                m1, m2, m3, m4, m5 = st.columns(5)
                m1.metric("Accuracy", f"{acc:.3f}")
                m2.metric("Precision", f"{prec:.3f}")
                m3.metric("Recall", f"{rec:.3f}")
                m4.metric("F1", f"{f1:.3f}")
                m5.metric("AUC", f"{auc:.3f}" if not np.isnan(auc) else "—")

                fig2, ax2 = plt.subplots(figsize=(4,3.2))
                im = ax2.imshow(cm, cmap="Blues")
                for (i, j), v in np.ndenumerate(cm):
                    ax2.text(j, i, str(v), ha='center', va='center', color="black")
                ax2.set_xticks([0,1]); ax2.set_xticklabels(["0","1"])
                ax2.set_yticks([0,1]); ax2.set_yticklabels(["0","1"])
                ax2.set_xlabel("Prédit"); ax2.set_ylabel("Réel")
                ax2.set_title("Matrice de confusion")
                st.pyplot(fig2)

            csv_bytes = out.to_csv(index=False).encode("utf-8")
            st.download_button(
                "💾 Télécharger les prédictions (CSV)",
                data=csv_bytes,
                file_name="predictions_diabetes.csv",
                mime="text/csv",
            )

            with st.expander("Aperçu des premières lignes"):
                st.dataframe(out.head(20))

        except Exception as e:
            st.error(f"Erreur lors du traitement du fichier : {e}")
