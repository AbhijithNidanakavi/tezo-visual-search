# Enterprise Visual Search

This project implements the Tezo case-study as a production-oriented visual search system. It indexes the supplied image dataset, retrieves the top matching images for a natural-language query, and returns concise AI-driven explanations describing why each result is relevant.

## What is included

- `download_images.py`: supplied helper for downloading the dataset locally
- `scripts/build_index.py`: offline preprocessing pipeline that embeds images and writes the search index
- `app/`: FastAPI API, HTML interface, retrieval logic, and explanation generation
- `tests/`: automated coverage for search quality, API behavior, and failure paths
- `Dockerfile` and `docker-compose.yml`: containerized deployment entrypoints

## Architecture

1. Download the image corpus from `photos_url.csv` into `data/images/`.
2. Build normalized image embeddings with OpenCLIP (`ViT-B-32` by default).
3. Score each image against a curated visual taxonomy to extract interpretable labels.
4. Persist embeddings in `data/indexes/embeddings.npy` and metadata in `data/indexes/metadata.json`.
5. At query time, encode the text with the same multimodal model, run cosine similarity search, and return the top 5 matches.
6. Generate business-safe explanations from matched labels, color cues, and similarity confidence.

The system is designed so the model backend can be swapped. Tests use a hashing embedder to avoid external downloads and keep CI deterministic.

## Local setup

```powershell
py -3.13 -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt
```

## Download the dataset

Place the provided files in the project root, then run:

```powershell
.venv\Scripts\python.exe download_images.py
```

The downloader writes optimized images to `images/` by default. For this project, move or copy them into `data/images/`, or change the output path inside the script to `data/images`.

## Build the index

```powershell
.venv\Scripts\python.exe scripts/build_index.py --images-dir data/images
```

For a faster smoke run:

```powershell
.venv\Scripts\python.exe scripts/build_index.py --images-dir data/images --limit 200
```

## Run the API and UI

```powershell
.venv\Scripts\python.exe -m uvicorn app.main:app --reload
```

Or use the one-command launcher:

```powershell
.\run.ps1
```

Useful options:

```powershell
.\run.ps1 -RebuildIndex
.\run.ps1 -RebuildIndex -IndexLimit 200
.\run.ps1 -SkipInstall
.\run.ps1 -UseReload
```

Endpoints:

- `GET /health`
- `GET /search?query=beach sunset`
- `GET /`

## Docker

```powershell
docker compose up --build
```

## Testing

```powershell
.venv\Scripts\python.exe -m pytest -q
```

## Edge cases covered

- empty query rejection
- search before indexing
- API health reporting when the index is missing
- deterministic retrieval behavior on synthetic fixtures
- explanation generation for color and semantic overlap

## Scale notes

- 25,000 image embeddings fit comfortably in memory for sub-second cosine similarity on CPU.
- For larger corpora, the current storage layer can be swapped for FAISS, pgvector, Qdrant, or Pinecone without changing the API contract.
- Image downloads, indexing, and metadata enrichment are separated from the online query path so search latency stays within the target range.
