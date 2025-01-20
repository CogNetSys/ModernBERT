# Semantic Search Demo: Building a Search Engine with ModernBERT 🔍

This demonstration showcases how to build a semantic search engine using the `lightonai/modernbert-embed-large` model. Unlike traditional keyword-based search, semantic search understands the meaning of the query and retrieves documents that are semantically related, even if they don't contain the exact keywords. This demo will guide you through the process of indexing a set of documents, encoding a search query, and finding the most relevant documents based on their semantic similarity.

---

## 🛠️ Prerequisites

Ensure you have the following installed:

-   Python 3.7 or higher
-   PyTorch 1.13.0 or higher
-   Hugging Face Transformers 4.26.0 or higher
-   NumPy
-   FAISS (for efficient similarity search)

You can install the required packages using pip:

```bash
pip install torch transformers numpy faiss-cpu # or faiss-gpu if you have a compatible GPU
```

---

##  1: Load the Model and Tokenizer

Import the necessary libraries and load the `lightonai/modernbert-embed-large` model and tokenizer.

```python
from transformers import AutoTokenizer, AutoModel
import torch
import faiss
import numpy as np

model_name = "lightonai/modernbert-embed-large"

# Load the tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Load the model
model = AutoModel.from_pretrained(model_name)

# Move the model to the GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
```

---

## 2: Prepare the Document Corpus

For this demo, we'll use a small set of example documents. In a real application, this could be a large collection of documents from your domain.

```python
documents = [
    "ModernBERT is a state-of-the-art language model for natural language processing.",
    "This document discusses the applications of semantic search in e-commerce.",
    "The quick brown fox jumps over the lazy dog.",
    "Large language models can generate human-quality text.",
    "Semantic search helps users find relevant information based on meaning.",
    "ModernBERT can handle sequences up to 8192 tokens long.",
    "This document provides a guide to fine-tuning ModernBERT.",
    "The cat sat on the mat."
]
```

---

## 3: Generate Document Embeddings

Tokenize the documents and generate embeddings using the ModernBERT model. We'll use the `[CLS]` token embedding as the document representation.

```python
# Tokenize the documents
inputs = tokenizer(documents, padding=True, truncation=True, return_tensors="pt", max_length=512)
inputs = {k: v.to(device) for k, v in inputs.items()}

# Generate embeddings
with torch.no_grad():
    outputs = model(**inputs)

# Use [CLS] token embedding as document representation
document_embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
```

---

## 4: Build a FAISS Index

We'll use FAISS (Facebook AI Similarity Search) to create an index for efficient similarity search.

```python
# Get the embedding dimension
embedding_dim = document_embeddings.shape[1]

# Create an index
index = faiss.IndexFlatL2(embedding_dim)  # L2 distance is good for comparing embeddings
# IndexFlatIP uses inner product, which works if embeddings are normalized

# Add the document embeddings to the index
index.add(document_embeddings.astype('float32')) #FAISS uses float32 by default
```

---

## 5: Encode the Search Query

Tokenize and generate the embedding for the search query.

```python
query = "What is ModernBERT?"

# Tokenize the query
query_inputs = tokenizer(query, return_tensors="pt", max_length=512)
query_inputs = {k: v.to(device) for k, v in query_inputs.items()}

# Generate query embedding
with torch.no_grad():
    query_outputs = model(**query_inputs)

# Use [CLS] token embedding as query representation
query_embedding = query_outputs.last_hidden_state[:, 0, :].cpu().numpy()
```

---

## 6: Perform Semantic Search

Use the FAISS index to find the documents most similar to the query embedding.

```python
# Search the index
k = 3  # Number of nearest neighbors to retrieve
distances, indices = index.search(query_embedding.astype('float32'), k)

print(f"Query: {query}")
print("\nTop 3 most relevant documents:")
for i, index in enumerate(indices[0]):
    print(f"  {i+1}. Document: {documents[index]}")
    print(f"     Distance: {distances[0][i]:.4f}")
```

---

## 🏁 Conclusion

Congratulations! You've built a basic semantic search engine using ModernBERT and FAISS. This demo illustrates the core principles of semantic search: encoding text into embeddings, indexing those embeddings, and efficiently searching for similar embeddings.

From here, you can explore more advanced features and applications:

-   Scaling up to a much larger document collection.
-   Using more sophisticated indexing structures in FAISS (e.g., `IndexIVF`, `IndexHNSW`).
-   Integrating the search engine into a web application or other user interface.
-   Fine-tuning ModernBERT on your specific domain or task to further improve search relevance.
-   Experimenting with different distance metrics (e.g., cosine similarity, inner product).
-   Combining semantic search with other search techniques (e.g., keyword-based search) to create hybrid search systems.

This demonstration provides a starting point for building powerful semantic search applications with ModernBERT. By understanding the meaning of queries and documents, semantic search can deliver more relevant results and enhance user experience across a wide range of applications.