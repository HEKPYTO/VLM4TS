import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from src.vit4ts import ViT4TS
from src.vlm_refine import refine

st.set_page_config(page_title="VLM4TS Demo", layout="wide")
st.title("VLM4TS — TS → Plot → Anomaly (training-free)")
st.caption("ViT-B/32 screening + single VLM call verification. No training.")

with st.sidebar:
    alpha = st.slider("alpha (top quantile)", 0.005, 0.05, 0.01, 0.005)
    use_vlm = st.checkbox("VLM verification (needs OPENAI_API_KEY)", False)
    st.markdown("Upload CSV with a `value` column (or single column).")

f = st.file_uploader("CSV", type=["csv"])
if f is None:
    st.info("Upload a CSV or try synthetic: click Run Demo")
    if not st.button("Run Demo (synthetic)"):
        st.stop()
    np.random.seed(0)
    s = np.sin(np.linspace(0, 20, 500)) + np.random.randn(500) * 0.15
    s[200:210] += 3
    df = pd.DataFrame({"value": s})
else:
    df = pd.read_csv(f)
    s = df.iloc[:, 0].to_numpy() if "value" not in df.columns else df["value"].to_numpy()

m = ViT4TS(alpha=alpha, window_size=224)
with st.spinner("ViT4TS scoring..."):
    scores, _ = m.predict_scores(s)
cands = m.candidates(scores)
st.write(f"ViT4TS candidates: {len(cands)} / {len(s)}")

fig, ax = plt.subplots(figsize=(10, 3))
ax.plot(s, color="black", lw=1)
ax.set_title("Series")
for c in cands:
    ax.axvline(c, color="orange", alpha=0.3)
st.pyplot(fig)
fig, ax = plt.subplots(figsize=(10, 2))
ax.plot(scores, color="steelblue", lw=1)
ax.set_title("Anomaly score (1 - cosine)")
ax.axhline(np.quantile(scores, 1 - alpha), color="red", ls="--", label="threshold")
ax.legend()
st.pyplot(fig)

if use_vlm:
    kept = refine(scores, cands, s)
    st.success(f"VLM kept {len(kept)} / {len(cands)}: {kept[:20]}")
else:
    st.info(f"Intervals (ViT4TS-only): {cands[:20]}{'...' if len(cands)>20 else ''}")
    if st.button("Explain (VLM) — needs API key"):
        kept = refine(scores, cands, s)
        st.write("VLM filtered:", kept[:20])
