import streamlit as st
import pandas as pd
import os
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain.llms import HuggingFacePipeline
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline

df = pd.read_csv("Training Dataset.csv")
df.columns = df.columns.str.strip()

system_prompt = (
    "You are a helpful and professional loan approval expert. "
    "Answer the user's question based only on the provided context.\n\n"
)

row_chunks = [
    f"Application {i+1}: " + ", ".join([f"{col}={row[col]}" for col in df.columns])
    for i, row in df.iterrows()
]

field_chunks = [
    "Loan_Status is Y for approved, N for denied.",
    "Loan approval depends on Credit_History, ApplicantIncome, CoapplicantIncome, LoanAmount, Education, Self_Employed, and Property_Area.",
    "Credit_History of 1 usually leads to higher chances of approval.",
    "Higher ApplicantIncome or CoapplicantIncome can help qualify for larger loans.",
]

all_chunks = row_chunks + field_chunks

embeddings = HuggingFaceEmbeddings(model_name="models/all-MiniLM-L6-v2")

vectordb = FAISS.from_texts(texts=all_chunks, embedding=embeddings)

model_name = "google/flan-t5-large"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
pipe = pipeline("text2text-generation", model=model, tokenizer=tokenizer, max_length=256)
llm = HuggingFacePipeline(pipeline=pipe)

retriever = vectordb.as_retriever(search_kwargs={"k": 3})
qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever)

st.set_page_config(page_title="Loan Approval Q&A Bot", page_icon="💰")
st.title("Loan Approval Q&A Bot")

query_input = st.text_input("Ask a question about loan approval:")

if query_input:
    full_query = system_prompt + query_input.lower()

    if "maximum loan amount" in full_query:
        max_amt = df["LoanAmount"].max()
        answer = f"The maximum loan amount in the dataset is {max_amt}."
    elif "minimum loan amount" in full_query:
        min_amt = df["LoanAmount"].min()
        answer = f"The minimum loan amount in the dataset is {min_amt}."
    elif "average loan amount" in full_query:
        avg_amt = df["LoanAmount"].mean()
        answer = f"The average loan amount in the dataset is {round(avg_amt, 2)}."
    elif "features" in full_query and "approve" in full_query:
        answer = (
            "Loan approvals are generally influenced by: Credit_History, "
            "ApplicantIncome, CoapplicantIncome, LoanAmount, Education, Self_Employed, "
            "and Property_Area."
        )
    else:
        docs = retriever.get_relevant_documents(full_query)
        if not docs or all(len(doc.page_content.strip()) < 10 for doc in docs):
            answer = "Sorry, I couldn’t find enough relevant information. Please rephrase your question."
        else:
            answer = qa.run(full_query)

    st.markdown(f"**Answer:** {answer}")

    with st.expander("Retrieved Context"):
        for i, doc in enumerate(retriever.get_relevant_documents(full_query)):
            st.markdown(f"**Snippet {i+1}:** {doc.page_content}")

