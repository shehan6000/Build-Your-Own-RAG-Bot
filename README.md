# RAG Bot: Question Answering over Mixed Documents

A retrieval-augmented generation (RAG) chatbot that answers questions from a mix of **PDF, PowerPoint, and Markdown** files. The example application answers questions about the **Donut** document-understanding model using its paper (PDF), a slide deck (PPTX), and its GitHub README (Markdown).

The pipeline parses each file with [Unstructured](https://unstructured.io/), cleans out noise (references, headers), chunks the content by section title, embeds it into a Chroma vector store, and answers questions with a LangChain conversational retrieval chain that cites its sources.

---

## How It Works

```
 PDF (paper)    PPTX (slides)    Markdown (README)
      │               │                 │
      ▼               ▼                 ▼
  Unstructured API  partition_pptx   partition_md
  (hi_res, yolox,
   table structure)
      │               │                 │
      └──── clean: drop References + Headers ────┘
                      │
                      ▼
            chunk_by_title (section-aware chunks)
                      │
                      ▼
       OpenAI embeddings → Chroma vector store
                      │
                      ▼
      Retriever (similarity, k=6, optional source filter)
                      │
                      ▼
   ConversationalRetrievalChain + sources-aware QA chain
                      │
                      ▼
             Markdown answer with sources
```

## Features

- **Layout-aware PDF parsing:** the `hi_res` strategy with the YOLOX model detects titles, tables, headers, and body text, and infers table structure (exported as HTML).
- **Noise removal:** drops the References section (via `parent_id` of the "References" title) and page headers so they don't pollute retrieval.
- **Multi-format ingestion:** PDF, PPTX, and Markdown are normalized into the same element format and chunked together.
- **Title-based chunking:** `chunk_by_title` keeps chunks aligned with document sections instead of arbitrary character windows.
- **Source-filtered retrieval:** restrict a question to one file (e.g. only `donut_readme.md`) using a metadata filter.
- **Scoped assistant prompt:** answers only about Donut, says "Hmm, I'm not sure." rather than guessing, and politely declines off-topic questions.
- **Bring your own files:** a Panel upload widget lets you add your own documents.

## Tech Stack

| Layer | Tool |
|---|---|
| Document parsing | `unstructured`, `unstructured-client` (hosted API for PDFs) |
| Chunking | `chunk_by_title` |
| Embeddings | OpenAI (`OpenAIEmbeddings`) |
| Vector store | Chroma |
| Orchestration | LangChain (`ConversationalRetrievalChain`, `load_qa_with_sources_chain`) |
| LLM | OpenAI completion model (`temperature=0`) |
| UI / uploads | Panel |

## Setup

```bash
git clone https://github.com/<your-username>/rag-bot.git
cd rag-bot
python -m venv .venv && source .venv/bin/activate
pip install unstructured "unstructured-client" chromadb \
            langchain langchain-community langchain-openai \
            lxml panel ipython
```

Set your credentials:

```bash
export OPENAI_API_KEY="sk-..."
export UNSTRUCTURED_API_KEY="..."        # from unstructured.io
export UNSTRUCTURED_API_URL="https://api.unstructured.io/general/v0/general"
```

> **Important:** the original notebook gets its keys from a course helper (`Utils`). Replace those calls with your own keys, for example:
> ```python
> import os
> s = UnstructuredClient(
>     api_key_auth=os.environ["UNSTRUCTURED_API_KEY"],
>     server_url=os.environ["UNSTRUCTURED_API_URL"],
> )
> ```
> Also, `pdf` partitioning with `hi_res` and table inference needs extra system dependencies if you run it locally instead of through the API.

## Project Layout

```
rag-bot/
├── Build_Your_Own_RAG_Bot.ipynb   # main notebook
├── example_files/
│   ├── donut_paper.pdf
│   ├── donut_slide.pptx
│   └── donut_readme.md
├── images/                        # screenshots used in the notebook
└── README.md
```

## Walkthrough

1. **Partition the PDF** through the Unstructured API (`hi_res`, `yolox`, table structure on). Table elements expose `metadata.text_as_html`.
2. **Remove the References section:** find the `Title` element whose text is "References" and drop every element whose `parent_id` matches it.
3. **Remove headers:** filter out elements with category `Header`.
4. **Partition the slide deck and README** with `partition_pptx` and `partition_md`.
5. **Chunk** all elements together with `chunk_by_title`.
6. **Build documents:** convert chunks to LangChain `Document`s, set `metadata["source"]` to the filename, and remove the `languages` field so Chroma accepts the metadata.
7. **Index** with `Chroma.from_documents(documents, OpenAIEmbeddings())` and create a similarity retriever (`k=6`).
8. **Create the chain:** a `ConversationalRetrievalChain` with the retriever, a question-generator `LLMChain`, and a sources-aware map-reduce QA chain.
9. **Ask questions:**

```python
qa_chain.invoke({
    "question": "How does Donut compare to other document understanding models?",
    "chat_history": []
})["answer"]
```

10. **Filter by source** to answer only from the README:

```python
filter_retriever = vectorstore.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 1, "filter": {"source": "donut_readme.md"}},
)
```

## Using Your Own Documents

- Drop files into `example_files/` (or use the Panel upload widget in the notebook).
- Choose the matching partitioner: `partition_pptx`, `partition_md`, or the Unstructured API for PDFs.
- Update the prompt template so the assistant is scoped to your topic instead of Donut.
- Rebuild the Chroma index and re-create the chains.

## Known Issues and Improvements

- **Undefined `filter` variable.** The final filtered call passes `"filter": filter`, which refers to Python's built-in `filter`. The source filter is already set on `filter_retriever`, so that key should simply be removed.
- **Prompt placement.** The scoped "Donut only" prompt is attached to the question-generator chain, not the final answering chain, so it may not be enforced at answer time. Pass it to the QA chain (e.g. via `combine_prompt` / `question_prompt` or a `stuff` chain) if you want strict scoping.
- **Deprecated LangChain APIs.** `langchain.chains.ConversationalRetrievalChain` and `OpenAI` (completion) are legacy. Consider migrating to `create_history_aware_retriever` + `create_retrieval_chain` and a chat model.
- **In-memory index.** Chroma here is not persisted; pass a `persist_directory` to keep the index between runs.
- **Cost and API dependence.** PDF parsing runs on a hosted API and embeddings and generation use OpenAI, so there are per-call costs and data leaves your machine.
- **No evaluation.** There is no retrieval or answer-quality evaluation yet.

## Roadmap

- [ ] Persist the vector store
- [ ] Migrate to current LangChain retrieval chains and a chat model
- [ ] Return answers with explicit source citations in the UI
- [ ] Add a Panel chat interface on top of the chain
- [ ] Add a small evaluation set (questions with expected sources)
- [ ] Support local, open-source embeddings and LLMs

## Acknowledgments

Built on [Unstructured](https://unstructured.io/), [LangChain](https://www.langchain.com/), [Chroma](https://www.trychroma.com/), and the [Donut](https://github.com/clovaai/donut) model materials used as example data.

