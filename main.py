import os
import json
import streamlit as st
from pathlib import Path

APP_DIR = Path(__file__).resolve().parent
DATA_DIR = APP_DIR / "data"
UPLOAD_DIR = DATA_DIR / "uploads"
EMBS_DIR = DATA_DIR / "embeddings"
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(EMBS_DIR, exist_ok=True)

####################################################################################################
from qdrant_haystack import QdrantDocumentStore
from qdrant_client import QdrantClient
from qdrant_client.http import models

QDRANT_URL = "localhost:6333"
qd_client = QdrantClient(url=QDRANT_URL)

def create_store(collection_name: str) -> QdrantDocumentStore:
    vectorstore = QdrantDocumentStore(
        url=QDRANT_URL,
        index=collection_name, # name of index, eg: "LegalDocs", "Manuals", etc
        # embedding_dim=384,  # should match dimensions of embeddings model (retrieval)
        embedding_dim=1536,  # should match dimensions of embeddings model (retrieval)
        recreate_index=False, # set to False for persistence, True to recreate/clear index
        hnsw_config={
            "m": 16, # num of bi-directional links for each index element (higher => better search accuracy, but higher memory usage and slower indexing)
            "ef_construct": 100, # dynamic list size for nearest neighbors (higher => better index accuracy, but slower indexing)
            "full_scan_threshold": 10000,
            },
        # quantization_config=models.ScalarQuantization(
        #     scalar=models.ScalarQuantizationConfig(
        #         type=models.ScalarType.INT8,
        #         quantile=0.99,
        #         always_ram=True,
        #     ),
        # ),
    )
    return vectorstore

def save_upload_file(upload_file) -> Path:
    file_path = UPLOAD_DIR / upload_file.name
    with open(file_path, "wb") as f:
        f.write(upload_file.getbuffer())
    return file_path


# TEXT CLEANING AND SPLITTING PREPROCESS
from haystack.nodes import PreProcessor
@st.cache_resource
def load_preprocessor() -> PreProcessor:
    preprocessor = PreProcessor(
        clean_empty_lines=True,
        clean_whitespace=True,
        clean_header_footer=False,
        split_by="sentence",
        split_length=100,
        split_overlap=5,
        split_respect_sentence_boundary=False,
    )
    return preprocessor


from haystack.nodes import TextConverter, PDFToTextConverter, DocxToTextConverter,  ImageToTextConverter
import pytesseract
pytesseract.pytesseract.tesseract_cmd = 'C:\\Program Files\\Tesseract-OCR\\tesseract.exe'
# DOCUMENT-TO-TEXT CONVERSION
def convert_to_text(type: str, file_path: Path, vectorstore: QdrantDocumentStore):
    if type == "text/plain":
        converter = TextConverter(remove_numeric_tables=False, valid_languages=["en"])
    elif type == "application/pdf":
        converter = PDFToTextConverter(remove_numeric_tables=False, valid_languages=["en"])
    elif type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
        converter = DocxToTextConverter(remove_numeric_tables=False, valid_languages=["en"])
    elif type in ["image/jpeg", "image/png", "image/bmp", "image/tiff"]:
        converter = ImageToTextConverter(remove_numeric_tables=False)
    else:
        st.error("Unsupported file type.")
        return

    document = converter.convert(file_path=Path(file_path), meta=None)[0]

    # Add filename to document metadata for downstream filtering
    # meta_doc = {"content": document.content, "meta": {"filename": file_path.name}}
    meta_doc = {"content": document.content, "meta": {"filename": file_path.name}}
    st.write(len(meta_doc['content']))

    # Preprocess document: character cleaning, text-splitting into chunks
    preprocessor = load_preprocessor()
    processed_docs = preprocessor.process(documents=[meta_doc])

    # Write documents to the document store and update embeddings
    # Important: duplicate_documents='skip' or else docs with same text will be overwritten
    vectorstore.write_documents(documents=processed_docs, duplicate_documents='skip')
    return processed_docs


# VECTORIZATION: GENERATE EMBEDDINGS
from sentence_transformers import SentenceTransformer
@st.cache_resource
def load_sentence_transformer() -> SentenceTransformer:
    st_model = SentenceTransformer("sentence-transformers/multi-qa-mpnet-base-dot-v1")
    return st_model
def vectorize(processed_docs) -> None:
    sentence_model = load_sentence_transformer()
    embeddings = sentence_model.encode([doc.content for doc in processed_docs])

    for doc, emb in zip(processed_docs, embeddings):
        embedding_path = EMBS_DIR / f"{doc.id}_embedding.json"
        with open(embedding_path, 'w') as ef:
            json.dump(emb.tolist(), ef)


def process_upload(upload_file, vectorstore: QdrantDocumentStore, collection_name: str):
    try:
        file_path = save_upload_file(upload_file)
        processed_docs = convert_to_text(upload_file.type, file_path, vectorstore)
        vectorize(processed_docs)
        return st.success(body=f"File '{upload_file.name}' uploaded to collection: {collection_name}!", icon="✔")
    except ValueError as ve:
        return st.error(body=f"An error occured: {str(ve)}", icon="❌")
    except Exception as e:
        return st.error(body=f"'{upload_file.name}' file upload unsuccessful: {e}", icon="❌")


import torch
from haystack.nodes import EmbeddingRetriever, BM25Retriever
def create_retriever(vectorstore: QdrantDocumentStore) -> EmbeddingRetriever:
    retriever = EmbeddingRetriever(
        document_store=vectorstore,
        # embedding_model="sentence-transformers/all-MiniLM-L6-v2", # 384 dimensions
        embedding_model="text-embedding-ada-002", # 1536 dimensions
        api_key=st.secrets['OPENAI_API_KEY'],
        batch_size=32,
        use_gpu=True,
        devices=[torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')],
        top_k=20,
    )
    return retriever


def collect_names(collect_json) -> list[str]:
    return [collection['name'] for collection in collect_json['collections']]

def retrieve_docs(user_query: str, documents: list[str], retriever: EmbeddingRetriever):
    docs = retriever.retrieve(query=user_query, filters={"filename": documents})
    return docs

def doc_filenames(documents) -> list[str]:
    return [doc.meta['filename'] if 'filename' in doc.meta else 'N/A' for doc in documents]


import openai

def answer_prompt(query: str, documents) -> str:
    client = openai.OpenAI(api_key=st.secrets['OPENAI_API_KEY'])
    pre_prompt = """You are a meticulous researcher who answers questions based solely on the combined context of different files. If a definitive answer can't be provided because of conflicting information between files, you can give specific reasons as to why or ask a clarifying question.
    """
    # Prepend context with document's filename
    context = ' '.join([f"({doc.meta['filename']}) {doc.content}" for doc in documents])
    response = client.chat.completions.create(
        model="gpt-3.5-turbo-1106",
        # response_format={ "type": "json_object" },
        messages=[
            {"role": "system", "content": pre_prompt},
            {"role": "user", "content": f"context:{context}\nquestion:{query}"},
        ],
        max_tokens=300,
        temperature=0,
    )
    # json_res = response.choices[0].message.content
    with st.expander(label="context"):
        st.caption(context)
    with st.expander(label="costs"):
        st.caption(response)
    return response.choices[0].message.content

####################################################################################################
st.title("RAG Chatbot with Haystack")

collect_json = json.loads(qd_client.get_collections().json())
collection_names = collect_names(collect_json)

with st.form(key="uploader", clear_on_submit=True):
    uploaded_file = st.file_uploader(
        label="Upload Documents:",
        type=["txt", "pdf", "docx", "jpg", "jpeg", "png", "bmp", "tiff"],
        key="file_upload",)

    c0, c1, c2 = st.columns([2,1,2])
    upsert_collection = c0.selectbox(
        label="Add to Collection:",
        options=collection_names,
        placeholder="Select existing collection.",
        index=None,
    )

    c1.divider()
    new_collection = c2.text_input(
        label="Create New Collection:",
        placeholder="Enter new collection name.")
    submit = st.form_submit_button(label="Upload", use_container_width=True,)

    form_collection_name = None

    if uploaded_file is not None:
        if upsert_collection and new_collection:
            st.error(body="Choose one: select an existing collection, or create a new one!", icon="⛔")
        if new_collection and not upsert_collection:
            form_collection_name = new_collection
        if upsert_collection and not new_collection:
            form_collection_name = upsert_collection
    else:
        st.info(body="Select an existing collection, or create a new one.", icon="ℹ")

    if form_collection_name is not None and submit:
            vectorstore = create_store(form_collection_name)
            process_upload(uploaded_file, vectorstore, form_collection_name)


select_collection = st.selectbox(label="Select Collection:", options=collection_names, placeholder="Select a collection.", index=None)
if select_collection:
    vectorstore = create_store(select_collection)
    if vectorstore:
        all_docs = vectorstore.get_all_documents()
        doc_names = doc_filenames(all_docs)
        if doc_names:
            select_names = list(set(doc_names))
            select_docs = st.multiselect(label="Select Documents:", options=select_names, default=select_names)
            if select_docs: st.write(select_docs)


user_query = st.text_input(label="User Query:", placeholder="Ask a question.")
if st.button(label="Ask"):
    if user_query:
        retriever = create_retriever(vectorstore)
        relevant_docs = retrieve_docs(user_query, select_docs, retriever)
        with st.expander("relevant docs"):
            st.write(relevant_docs)
        answer = answer_prompt(user_query, relevant_docs)
        st.write(answer)
    else: st.info("Please enter your question.")
