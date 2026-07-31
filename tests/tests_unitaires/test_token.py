import pytest
import pandas as pd
from langchain_text_splitters import RecursiveCharacterTextSplitter

@pytest.fixture
def df_sample():
    return pd.read_csv("./tests/ge_events_sample.csv")

# tester si le splitter découpe bien le texte en plusieurs tokens
def test_chunking_basic(df_sample):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=700,
        chunk_overlap=200,
        separators=["\n\n", "\n", ".", " ", ""]
    )

    row = df_sample.iloc[0]
    chunks = splitter.split_text(row["texte_rag"])

    assert isinstance(chunks, list)
    assert len(chunks) > 0

# tester que la taille des tokens est respectée (700 caractères)
def test_chunking_size(df_sample):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=700,
        chunk_overlap=200,
        separators=["\n\n", "\n", ".", " ", ""]
    )

    row = df_sample.iloc[0]
    chunks = splitter.split_text(row["texte_rag"])

    assert all(len(c) <= 700 for c in chunks)

# tester qu'aucun taken est vide ou inutilisable et leur compatibilité avec les vraies données
def test_chunking_no_empty(df_sample):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=700,
        chunk_overlap=200,
        separators=["\n\n", "\n", ".", " ", ""]
    )

    row = df_sample.iloc[0]
    chunks = splitter.split_text(row["texte_rag"])

    assert all(len(c.strip()) > 0 for c in chunks)
