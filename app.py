import os
import streamlit as st
from dotenv import load_dotenv
import fitz  # PyMuPDF
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from groq import Groq
import time

# Load API key
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    st.error("GROQ_API_KEY not found in .env file!")
    st.stop()

# Initialize models
embedder = SentenceTransformer('all-MiniLM-L6-v2')
groq_client = Groq(api_key=GROQ_API_KEY)

st.title("Research Paper Analyzer")
st.write("**Upload PDFs from your device!**")
st.write("Powered by **Llama-3.3-70B-Versatile** (best Groq model)")

# Upload PDFs
uploaded_files = st.file_uploader("Upload PDF files", type=["pdf"], accept_multiple_files=True)

if st.button("Start Analysis", type="primary"):
    if not uploaded_files:
        st.error("Please upload at least one PDF.")
        st.stop()

    papers_text = {}
    paper_titles = {}
    progress_bar = st.progress(0)

    # Read PDFs and extract text
    for idx, uploaded_file in enumerate(uploaded_files):
        progress_bar.progress((idx + 1) / len(uploaded_files))
        try:
            # Read PDF
            with fitz.open(stream=uploaded_file.read(), filetype="pdf") as doc:
                text = ""
                for page in doc:
                    text += page.get_text()

            title_guess = text[:1000].split('\n')[0].strip()[:120]
            if len(title_guess) < 10:
                title_guess = f"Paper {idx+1}"

            papers_text[uploaded_file.name] = text
            paper_titles[uploaded_file.name] = title_guess
            time.sleep(0.3)
        except Exception as e:
            st.error(f"Failed {idx+1}: {uploaded_file.name} → {e}")
            continue

    if not papers_text:
        st.error("No papers processed.")
        st.stop()

    # Build RAG index
    st.info("Building vector database...")
    index = faiss.IndexFlatL2(384)
    metadata = []
    for filename, text in papers_text.items():
        chunks = [text[i:i+1800] for i in range(0, len(text), 1400)]
        embeddings = embedder.encode(chunks, show_progress_bar=False)
        for chunk, emb in zip(chunks, embeddings):
            index.add(np.array([emb], dtype='float32'))
            metadata.append({"filename": filename, "title": paper_titles[filename], "text": chunk})

    def retrieve(query, filename_filter=None, k=10):
        q = embedder.encode([query])
        _, I = index.search(q, k*3)
        results = [metadata[i] for i in I[0] if i < len(metadata)]
        if filename_filter:
            results = [r for r in results if r["filename"] == filename_filter]
        return results[:k]

    # Generate per-paper summaries
    st.info("Generating high-quality summaries with Llama-3.3-70B...")
    summaries = {}
    prog = st.progress(0)
    for i, filename in enumerate(papers_text):
        prog.progress((i+1)/len(papers_text))
        chunks = retrieve("core idea, method, results, contribution", filename_filter=filename, k=12)
        context = "\n\n".join([c["text"] for c in chunks])

        prompt = f"""You are a world-class researcher. Write a clear, professional 250–350 word summary of this paper covering:
        • Problem & main contribution
        • Method
        • Key results
        • Limitations

        Paper:
        {context}"""

        resp = groq_client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.6,
            max_tokens=600
        )
        summaries[filename] = resp.choices[0].message.content.strip()

    all_summaries = "\n\n".join([f"### {paper_titles[f]}\n{summaries[f]}" for f in papers_text])

    # Overall summary
    overall = groq_client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": f"Give a concise 2–3 paragraph overview of these papers:\n\n{all_summaries}"}],
        temperature=0.6, max_tokens=900
    ).choices[0].message.content

    # Comparison
    comparison = groq_client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": f"Compare these papers: similarities, differences, strengths, weaknesses:\n\n{all_summaries}"}],
        temperature=0.6, max_tokens=1200
    ).choices[0].message.content

    # Novel ideas
    ideas = groq_client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": f"Based on these papers, propose 8 original, high-impact research ideas that fill gaps or combine ideas:\n\n{all_summaries}"}],
        temperature=0.95, max_tokens=1500
    ).choices[0].message.content

    # Display results
    st.success("Analysis Complete with Llama-3.3-70B!")
    st.subheader("Overall Summary")
    st.write(overall)

    st.subheader("Comparison")
    st.write(comparison)

    st.subheader("8 Novel Research Ideas")
    st.write(ideas)

    st.subheader("Individual Summaries")
    for filename in papers_text:
        with st.expander(f"{paper_titles[filename]}"):
            st.write(f"File: {filename}")
            st.write(summaries[filename])
