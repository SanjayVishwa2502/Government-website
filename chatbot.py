import os
import textwrap
from pathlib import Path

from google.colab import userdata
from IPython.display import Markdown
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import FlashrankRerank
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Qdrant
from langchain_community.document_loaders import UnstructuredMarkdownLoader
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from llama_parse import LlamaParse

os.environ["GROQ_API_KEY"] = "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"


def print_response(response):
    response_txt = response["result"]
    for chunk in response_txt.split("\n"):
        if not chunk:
            print()
            continue
        print("\n".join(textwrap.wrap(chunk, 100, break_long_words=False))
!mkdir data
!gdown 1oyOeUFljWUT0mFRlss-z0BlHGnr0d0iu -O "data/rpp.pdf"
import os
os.environ["LLAMA_PARSE_API_KEY"] = "llx-UMlkd1qUmZt9vzd6Dz4CzpykcVCY6uzrpqfAx9CqJlB06CzS"
instruction = """The provided document contains detailed financial information, including unaudited financial statements, management commentary, key highlights, and outlook disclosures related to a company's performance for a specific fiscal period. The document may contain various financial tables, figures, and data points. When answering questions based on the information in this document, strive to provide accurate and precise
responses by carefully referencing the relevant data and details presented in the document."""

parser = LlamaParse(
    api_key=os.getenv("LLAMA_PARSE_API_KEY"),
    result_type="markdown",
    parsing_instruction=instruction,
    max_timeout=5000,
)

llama_parse_documents = await parser.aload_data("./data/rpp.pdf")
parsed_doc = llama_parse_documents[0]
document_path = Path("data/parsed_document.md")
with document_path.open("w") as f:
    f.write(parsed_doc.text)

loader = UnstructuredMarkdownLoader(document_path)
loaded_documents = loader.load()
text_splitter = RecursiveCharacterTextSplitter(chunk_size=2048, chunk_overlap=128)
docs = text_splitter.split_documents(loaded_documents)
embeddings = FastEmbedEmbeddings(model_name="BAAI/bge-base-en-v1.5")
qdrant = Qdrant.from_documents(
    docs,
    embeddings,
    path="./db",
    collection_name="document_applications",
)
# Define the query before invoking the retriever
query = "How do I apply for a driver's license?"  # Example query

# Retrieve relevant documents
retriever = qdrant.as_retriever(search_kwargs={"k": 5})
retrieved_docs = retriever.invoke(query)  # Now 'query' is defined

# Process and print the retrieved documents
for doc in retrieved_docs:
    print(f"id: {doc.metadata['_id']}\n")
    print(f"text: {doc.page_content[:256]}\n")
    print("-" * 80)
compressor = FlashrankRerank(model="ms-marco-MiniLM-L-12-v2")
compression_retriever = ContextualCompressionRetriever(
    base_compressor=compressor, base_retriever=retriever
)
llm = ChatGroq(temperature=0, model_name="llama3-70b-8192")
prompt_template = """
Use the following pieces of information to answer the user's question regarding document applications and renewals.
If you don't know the answer, just say that you don't know, don't try to make up an answer.

Context: {context}
Question: {question}

Answer the question based on the relevant procedures, if applicable. Be succinct and clear.
"""

prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])

qa = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=compression_retriever,
    return_source_documents=True,
    chain_type_kwargs={"prompt": prompt, "verbose": True},
)
response = qa.invoke("How do I apply for a driver's license?")
print_response(response)
