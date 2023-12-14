import streamlit as st
from haystack.nodes import PreProcessor

@st.cache_resource
def load_preprocessor(doc_length) -> PreProcessor:
    """TEXT CLEANING, SPLITTING, OVERLAPS"""
    if doc_length < 1000:
        char_split = 0
        overlap = 0
    elif doc_length < 5000:
        char_split = 1000
        overlap = 20
    else:
        char_split = 750
        overlap = 15

    preprocessor = PreProcessor(
        clean_empty_lines=True,
        clean_whitespace=True,
        clean_header_footer=False,
        split_by="sentence",
        split_length=char_split,
        split_overlap=overlap,
        split_respect_sentence_boundary=False,
    )
    return preprocessor
