# Research Paper Analyzer (AI-Powered)

An advanced AI-driven research tool that leverages **Llama-3.3-70B** via Groq to analyze, summarize, and compare multiple research papers simultaneously. It uses **RAG (Retrieval-Augmented Generation)** to provide high-fidelity insights and even suggests novel research ideas based on the provided literature.

---

## Features

- **Multi-PDF Processing**: Upload and analyze multiple research papers in one go.
- **Deep Summarization**: Generates professional 250–350 word summaries covering Problem, Method, Results, and Limitations.
- **Comparative Analysis**: Automatically identifies similarities, differences, strengths, and weaknesses across papers.
- **RAG-Powered Intelligence**: Uses FAISS vector database and `all-MiniLM-L6-v2` embeddings for precise context retrieval.
- **Novel Idea Generation**: Proposes 8 original, high-impact research ideas that bridge gaps in the analyzed literature.
- **Premium UI**: Built with Streamlit for a clean, interactive, and responsive experience.

---

## Tech Stack

- **Large Language Model**: `Llama-3.3-70B-Versatile` (via [Groq](https://groq.com/))
- **Frontend**: [Streamlit](https://streamlit.io/)
- **Embeddings**: `all-MiniLM-L6-v2` ([Sentence-Transformers](https://www.sbert.net/))
- **Vector Database**: [FAISS](https://github.com/facebookresearch/faiss)
- **PDF Extraction**: [PyMuPDF (fitz)](https://pymupdf.readthedocs.io/)

---

## Installation & Setup

### 1. Clone the Repository
```bash
git clone https://github.com/your-username/research-paper-analyzer.git
cd Research-Paper-Analyzer
```

### 2. Set Up Environment
It is recommended to use a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Configure API Key
Create a `.env` file in the root directory and add your Groq API key:
```env
GROQ_API_KEY=your_groq_api_key_here
```

---

## Usage

1. **Launch the Dashboard**:
   ```bash
   streamlit run app.py
   ```
2. **Upload Research Papers**: Use the file uploader to select one or more PDF files.
3. **Start Analysis**: Click the **"Start Analysis"** button.
4. **Review Insights**:
   - **Overall Summary**: A global overview of all analyzed papers.
   - **Comparison Table**: Deep dive into how the papers relate.
   - **8 Novel Ideas**: High-impact directions for future research.
   - **Individual Summaries**: Expandable sections for each paper.

---

## Project Flow

1. **Document Loading**: PyMuPDF extracts raw text from uploaded PDFs.
2. **Chunking**: Text is split into 1800-character overlapping chunks for context preservation.
3. **Vectorization**: `SentenceTransformer` converts chunks into 384-dimensional embeddings.
4. **Indexing**: Embeddings are stored in a FAISS flat index for lightning-fast retrieval.
5. **Contextual Retrieval (RAG)**: The system searches for the most relevant segments based on specific prompts.
6. **Insight Synthesis**: `Llama-3.3-70B` processes the retrieved context to generate structured summaries and cross-paper insights.

---

## License

This project is licensed under the [MIT License](LICENSE).
