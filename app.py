"""Streamlit UI:  streamlit run app.py"""
import streamlit as st

from src.rag import answer

st.set_page_config(page_title="Scopus RAG", page_icon="📚")
st.title("📚 Ask the Maternity-Law Literature")
st.caption("Answers are generated only from the indexed Scopus papers, with citations.")

with st.sidebar:
    st.header("Filters")
    k = st.slider("Chunks to retrieve", 3, 12, 6)
    year_min = st.number_input("Published from year", 0, 2026, 0, help="0 = no filter")
    ctype = st.selectbox("Chunk type", ["all", "abstract", "fulltext"])

q = st.text_input("Your question", placeholder="What did studies find about maternity leave and breastfeeding in India?")

if st.button("Ask") and q:
    with st.spinner("Searching papers and generating answer..."):
        text, hits = answer(q, k=k, year_min=year_min or None,
                            chunk_type=None if ctype == "all" else ctype)
    st.subheader("Answer")
    st.write(text)
    st.subheader("Sources")
    for i, h in enumerate(hits, 1):
        m = h["meta"]
        with st.expander(f"[{i}] {m['title']} ({m['year']}) - {m['chunk_type']}"):
            st.write(f"**Journal:** {m['source']}  |  **Cited by:** {m['cited_by']}")
            if m["doi"]:
                st.write(f"**DOI:** https://doi.org/{m['doi']}")
            st.write(h["text"])
