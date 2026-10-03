# RAG Evaluation Lab

An end-to-end Retrieval-Augmented Generation (RAG) system with an evaluation harness for measuring retrieval quality, answer quality, grounding, latency, and out-of-domain (OOD) rejection.

## Live Demo

**Render:** https://rag-evaluation-lab.onrender.com

## Overview

RAG Evaluation Lab is designed to build and systematically evaluate a Retrieval-Augmented Generation pipeline.

The system follows the workflow:

```text
Documents
    ↓
Document Ingestion
    ↓
Text Chunking
    ↓
Embeddings / Retrieval
    ↓
Vector Search
    ↓
Relevant Context
    ↓
LLM Generation
    ↓
Answer + Evaluation Metrics
```

The project also includes an evaluation harness that allows different RAG configurations to be compared using consistent metrics and datasets.

## Key Features

- Document ingestion from text files
- Configurable text chunking and overlap
- Semantic retrieval using Sentence Transformers and FAISS
- Lightweight TF-IDF retrieval for memory-constrained deployment
- LLM-based answer generation
- Retrieval threshold and OOD rejection
- Recall@K evaluation
- Mean Reciprocal Rank (MRR)
- Evidence Recall@K
- Semantic answer similarity
- Context support / grounding score
- LLM-based answer correctness evaluation
- Retrieval and generation latency measurement
- Chunk-size experiments
- Top-K experiments
- Embedding-model comparison
- Interactive Streamlit interface
- Docker-based deployment
- Ollama / Ollama Cloud integration

## Tech Stack

### Machine Learning / NLP

- Python
- Sentence Transformers
- `all-MiniLM-L6-v2`
- TF-IDF
- Embeddings
- Semantic Search
- Cosine Similarity

### Retrieval

- FAISS
- Lightweight TF-IDF Retriever
- Top-K Retrieval
- Similarity Thresholding

### Generative AI

- Ollama
- Ollama Cloud
- `gpt-oss:20b`
- Prompt Engineering
- Retrieval-Augmented Generation (RAG)

### Evaluation

- Recall@K
- Mean Reciprocal Rank (MRR)
- Evidence Recall@K
- Semantic Similarity
- Context Support Score
- OOD Rejection
- Latency Analysis

### Application / Deployment

- Streamlit
- Docker
- Render
- Git / GitHub

## Project Architecture

```text
                         ┌──────────────────┐
                         │     Documents    │
                         └────────┬─────────┘
                                  ↓
                         ┌──────────────────┐
                         │    Chunking      │
                         └────────┬─────────┘
                                  ↓
                    ┌──────────────────────────┐
                    │     Retrieval Layer      │
                    │                          │
                    │ Sentence Transformers    │
                    │        + FAISS           │
                    │           OR             │
                    │    Lightweight TF-IDF     │
                    └────────────┬─────────────┘
                                 ↓
                         ┌──────────────────┐
                         │ Relevant Context │
                         └────────┬─────────┘
                                  ↓
                         ┌──────────────────┐
                         │   OOD / Threshold│
                         │      Check       │
                         └────────┬─────────┘
                                  ↓
                         ┌──────────────────┐
                         │   Ollama / LLM   │
                         └────────┬─────────┘
                                  ↓
                         ┌──────────────────┐
                         │ Generated Answer │
                         └──────────────────┘
```

## Retrieval

### Semantic Retrieval

The primary local evaluation pipeline uses:

```text
Sentence Transformers
        ↓
all-MiniLM-L6-v2
        ↓
384-dimensional embeddings
        ↓
FAISS
        ↓
Cosine similarity
```

Embeddings are normalized before being indexed, allowing inner product search to represent cosine similarity.

### Lightweight Deployment Retrieval

The Render deployment uses a lightweight TF-IDF-based retriever.

This was introduced because the original Sentence Transformer + PyTorch stack exceeded the memory available on the Render free instance.

The deployment therefore uses:

```text
Query
 ↓
Tokenization
 ↓
TF-IDF weighting
 ↓
Vector similarity
 ↓
Top-K results
```

The local evaluation pipeline and the deployed pipeline therefore use different retrieval implementations.

## Dataset

The evaluation corpus contains documents covering:

- RAG fundamentals
- Embeddings
- Vector databases
- Chunking
- Retrieval
- RAG evaluation
- Football rules

The documents are stored in:

```text
data/documents/
```

## Evaluation Metrics

### Recall@K

Measures whether the expected source appears within the top K retrieved results.

### Mean Reciprocal Rank (MRR)

Measures how highly the first relevant result appears.

### Evidence Recall@K

Checks whether the expected evidence appears within the retrieved chunks.

### Semantic Answer Similarity

Compares generated and expected answers using embedding similarity.

### Context Support Score

Measures the semantic similarity between the generated answer and retrieved context.

This metric is used as a context-support signal rather than a definitive faithfulness measurement.

### OOD Rejection

Tests whether questions outside the knowledge base are rejected rather than answered using irrelevant retrieved information.

## Evaluation Results

The main semantic retrieval configuration produced the following results.

### Retrieval Evaluation

| Top-K | Recall@K | MRR | Evidence Recall@K |
|------:|---------:|----:|------------------:|
| 1 | 0.818 | 0.818 | 0.818 |
| 3 | 1.000 | 0.894 | 0.909 |
| 5 | 1.000 | 0.894 | 1.000 |

Increasing Top-K improved evidence retrieval while also increasing the amount of context passed to the generation stage.

### Chunking Experiment

| Chunk Size | Evidence Recall@3 | Answer Similarity | Grounding | Correctness |
|-----------:|------------------:|------------------:|----------:|------------:|
| 30 | 0.727 | 0.804 | 0.842 | 0.818 |
| 50 | 0.909 | 0.757 | 0.710 | 0.909 |
| 100 | 1.000 | 0.843 | 0.657 | 1.000 |

The experiment demonstrates the trade-off between retrieving focused chunks and preserving enough context to answer questions correctly.

### Embedding Model Experiment

| Model | Evidence Recall | Answer Similarity | Grounding | Retrieval |
|---|---:|---:|---:|---:|
| MiniLM | 0.909 | 0.757 | 0.710 | 0.018s |
| MPNet | 1.000 | 0.849 | 0.750 | 0.059s |

The MPNet experiment produced stronger retrieval and semantic-similarity results on this evaluation dataset, with higher retrieval latency.

### Top-K Experiment

| Top-K | Evidence Recall | Answer Similarity | Grounding | Retrieval | Generation | Total |
|------:|----------------:|------------------:|----------:|----------:|-----------:|------:|
| 1 | 0.818 | 0.736 | 0.704 | 0.016s | 8.817s | 8.832s |
| 3 | 0.909 | 0.752 | 0.706 | 0.017s | 9.501s | 9.518s |
| 5 | 1.000 | 0.816 | 0.714 | 0.014s | 10.923s | 10.938s |

Increasing Top-K improved evidence retrieval and answer similarity on this dataset, while generation latency also increased.

### Answer Correctness

Using the evaluation dataset with:

```text
Chunk size: 50
Top-K: 3
Threshold: 0.3
```

the system achieved:

```text
In-domain correctness: 10 / 11
Correctness: 90.9%
```

### OOD Evaluation

Six out-of-domain questions were tested.

```text
OOD questions: 6
Rejected: 6
OOD rejection rate: 100%
```

The observed top retrieval scores for the OOD questions remained below the selected semantic-retrieval threshold.

## Threshold Calibration

The semantic retrieval evaluation used a threshold of:

```text
0.3
```

Threshold experiments produced:

| Threshold | OOD Rejection | In-domain Acceptance |
|----------:|--------------:|---------------------:|
| 0.2 | 0.667 | 1.000 |
| 0.3 | 1.000 | 1.000 |
| 0.4 | 1.000 | 1.000 |
| 0.5 | 1.000 | 1.000 |

The threshold of `0.3` was selected for the semantic retrieval evaluation.

The Render deployment uses a different lightweight TF-IDF retriever whose score distribution is different. Therefore, a separate deployment threshold of:

```text
0.18
```

is used.

The `0.3` and `0.18` thresholds are specific to their respective retrieval configurations and should not be interpreted as universal RAG thresholds.

## Latency

The evaluation tracks three latency measurements:

- Retrieval latency
- Generation latency
- Total latency

For example, with Top-K = 5:

```text
Retrieval:   0.014s
Generation: 10.923s
Total:       10.938s
```

Generation accounts for most of the total latency in the evaluated configurations.

## LLM Generation

The project supports Ollama-based generation.

For local development, Ollama can run locally with models such as:

```text
qwen2.5:3b
llama3.2:1b
```

For the deployed Render application, generation uses Ollama Cloud through an API key.

The deployed architecture is therefore:

```text
Render
  ↓
Lightweight Retriever
  ↓
Relevant Context
  ↓
Ollama Cloud
  ↓
LLM
  ↓
Generated Answer
```

## Streamlit Interface

The project includes an interactive Streamlit interface containing:

- RAG system overview
- Query console
- Generated answer
- Retrieved evidence
- Similarity scores
- Retrieval latency
- Generation latency
- Total latency
- Evaluation information

The interface is designed as an AI evaluation console rather than a simple chatbot.

## Deployment

The application is containerized using Docker and deployed on Render.

The deployment uses a lightweight dependency set:

```text
streamlit
ollama
```

The heavier local evaluation dependencies, including:

```text
sentence-transformers
faiss-cpu
numpy
chromadb
```

remain available in the main local `requirements.txt`.

This separation allows the full evaluation environment to remain available locally while keeping the production deployment within the available memory constraints.

## Project Structure

```text
RAG Evaluation Lab/
│
├── app.py
├── Dockerfile
├── .dockerignore
├── .gitignore
├── requirements.txt
├── requirements-deploy.txt
│
├── assets/
│   └── rag_pic.png
│
├── data/
│   ├── documents/
│   │   ├── rag_intro.txt
│   │   ├── embeddings.txt
│   │   ├── vector_databases.txt
│   │   ├── chunking.txt
│   │   ├── retrieval.txt
│   │   ├── evaluation.txt
│   │   └── football_rules.txt
│   │
│   ├── chroma/
│   └── eval/
│
├── evaluation/
│   ├── retrieval_metrics.py
│   ├── answer_metrics.py
│   └── grounding_metrics.py
│
└── src/
    ├── ingestion.py
    ├── chunking.py
    ├── embeddings.py
    ├── vector_store.py
    ├── chroma_store.py
    ├── lightweight_retriever.py
    ├── retriever.py
    ├── generator.py
    └── rag_pipeline.py
```

## Local Setup

### 1. Clone the repository

```bash
git clone https://github.com/MagnumOpus31/rag-evaluation-lab.git
cd rag-evaluation-lab
```

### 2. Create a virtual environment

```bash
python -m venv venv
```

### 3. Activate the environment

For Git Bash:

```bash
source venv/Scripts/activate
```

### 4. Install dependencies

```bash
pip install -r requirements.txt
```

### 5. Start Ollama

Install Ollama and make sure the required model is available.

For example:

```bash
ollama pull qwen2.5:3b
```

### 6. Run the application

```bash
python -m streamlit run app.py
```

The application will open in the browser.

## Environment Variables

For Ollama Cloud deployment, configure:

```text
OLLAMA_API_KEY=<your-api-key>
```

The API key should be stored as an environment variable and should not be committed to GitHub.

## Evaluation Philosophy

The project does not treat a single metric as sufficient to evaluate a RAG system.

A configuration can retrieve the correct evidence while still producing a poor answer. Similarly, a generated answer can appear semantically similar to an expected answer without being fully supported by the retrieved evidence.

Therefore, the evaluation harness considers multiple dimensions:

```text
Retrieval Quality
      +
Evidence Retrieval
      +
Answer Similarity
      +
Context Support
      +
Correctness
      +
OOD Rejection
      +
Latency
```

This allows different RAG configurations to be compared using a consistent evaluation process.

## Limitations

- Evaluation results are specific to the current document collection and evaluation dataset.
- Similarity-based thresholds are not universal and require calibration for different retrievers and datasets.
- Context Support Score is an embedding-based signal and does not guarantee factual faithfulness.
- The lightweight deployment retriever is lexical and does not provide the same semantic retrieval behavior as Sentence Transformers.
- LLM generation latency depends on the model and inference service.
- The current evaluation dataset is relatively small and should not be treated as a universal benchmark.

## Future Improvements

Potential extensions include:

- Larger evaluation datasets
- Automated evaluation pipelines
- Additional LLM-as-a-judge metrics
- Better faithfulness and citation evaluation
- Hybrid lexical + semantic retrieval
- Reranking
- Query rewriting
- Automated experiment tracking
- More extensive OOD datasets
- Configuration comparison dashboards

## Author

**Jinay Shah**

B.Tech — Electronics and Telecommunication Engineering

Interested in Artificial Intelligence, Machine Learning, Data Science, and AI Engineering.