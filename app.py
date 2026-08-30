import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from vlm4ts.vit4ts import ViT4TS
from vlm4ts.vlm_refine import refine

st.set_page_config(page_title="VLM4TS prototype", layout="wide")
st.title("VLM4TS-inspired anomaly screening prototype")
st.caption("Independent educational prototype; not an official reproduction of the paper.")

with st.sidebar:
    alpha = st.slider("alpha (top quantile)", 0.005, 0.05, 0.01, 0.005)
    use_vlm = st.checkbox("VLM verification (needs OPENAI_API_KEY)")
    st.caption("Enabling this sends the rendered series to OpenAI and may incur API costs.")
    st.markdown("Upload a CSV with a `value` column, or use its first column.")

uploaded = st.file_uploader("CSV", type=["csv"])
if uploaded is None:
    if not st.button("Run synthetic demo"):
        st.info("Upload a CSV or run the synthetic demo.")
        st.stop()
    series = np.sin(np.linspace(0, 20, 256))
    series[120:130] += 3
else:
    try:
        frame = pd.read_csv(uploaded)
    except (OSError, pd.errors.ParserError, UnicodeDecodeError) as error:
        st.error(f"Could not read CSV: {error}")
        st.stop()
    if frame.empty:
        st.error("CSV has no rows.")
        st.stop()
    values = frame["value"] if "value" in frame else frame.iloc[:, 0]
    series = pd.to_numeric(values, errors="coerce").to_numpy()
    if not series.size or not np.isfinite(series).all():
        st.error("CSV values must be finite numbers.")
        st.stop()

model = ViT4TS(alpha=alpha, window_size=224)
with st.spinner("Scoring windows..."):
    scores, _ = model.predict_scores(series)
candidates = model.candidates(scores)
st.write(f"Candidates: {len(candidates)} / {len(series)}")

fig, axis = plt.subplots(figsize=(10, 3))
axis.plot(series, color="black", lw=1)
for candidate in candidates:
    axis.axvline(candidate, color="orange", alpha=0.3)
st.pyplot(fig)

fig, axis = plt.subplots(figsize=(10, 2))
axis.plot(scores, color="steelblue", lw=1)
evaluated_scores = scores[np.isfinite(scores)]
if evaluated_scores.size:
    axis.axhline(np.quantile(evaluated_scores, 1 - alpha), color="red", ls="--")
st.pyplot(fig)

if use_vlm:
    kept = refine(scores, candidates, series)
    st.success(f"VLM kept {len(kept)} / {len(candidates)}: {kept[:20]}")
else:
    st.info(f"VLM disabled; screening candidates: {candidates[:20]}")
