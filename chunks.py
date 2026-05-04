import langchain
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader,PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_classic.retrievers import RePhraseQueryRetriever
#from langchain_core.documents.compressor import LLMChainExtractor

# %%
from langchain_community.llms import Ollama
llm=Ollama(model="llama3.1")
from langchain_community.embeddings import HuggingFaceEmbeddings
embedding_model=HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# %% [markdown]
# ### TEXT FILE CHUNKING

# %%
#Text Files chunking

#Function to fetch all folders existing inside a base folder
def get_all_folders(folder_path):
    folders = []
    for root, dirs, files in os.walk(folder_path):
        for dir in dirs:
            folders.append(os.path.join(root, dir))
    return folders

#Function to create chunks out of all text files inside a folder
def text_folder_chunk(folder_path):
    total_chunks=[]
    def load(file_path):
        loader=TextLoader(file_path,encoding="utf-8")
        docs=loader.load()
        return docs

    def textChunk(docs,filepath):
        splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=80,
        separators=["\n\n","\n"]
        )
        chunks = splitter.split_documents(docs)
        filename = os.path.basename(filepath)
        for i, chunk_doc in enumerate(chunks):
            chunk_doc.metadata["chunk_id"] = i
            chunk_doc.metadata["topic"] = filename.replace("-"," ").replace("_"," ")
            chunk_doc.metadata["source"] = filepath
        return chunks

    import os
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.endswith(".txt"):
                file_path=os.path.join(root, file)
                docs=load(file_path)
                chunks=textChunk(docs,file_path)
                total_chunks.extend(chunks)

    return total_chunks


folders = get_all_folders(r"data\textfiles")
text_files_chunk=[]
for folder in folders:
    text_chunk=text_folder_chunk(folder)
    text_files_chunk.extend(text_chunk)

# %%
vectorstore = Chroma.from_documents(
    documents=text_files_chunk,
    embedding=embedding_model,
    persist_directory="chroma_db_1"
)

retriever = vectorstore.as_retriever(
    search_kwargs={"k": 5},
    search_type="mmr"
)

# %%
query = "Help me debug a python script"

results = retriever.invoke(query)

for doc in results:
    print(doc.page_content)
    print(doc.metadata)
    print("-----")

# %%
query = "I cant speak in ms teams, what to do?"

results = retriever.invoke(query)

for doc in results:
    print(doc.page_content)
    print(doc.metadata)
    print("-----")

# %% [markdown]
# #### JSON FILE CHUNKING

# %%
import json
import os
from langchain_core.documents import Document

folder_path = r"data\jsonfiles"
json_files_chunk = []


def flatten_metadata(metadata, parent_key="", sep="_"):
    flat = {}
    for k, v in metadata.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            flat.update(flatten_metadata(v, new_key, sep))
        else:
            flat[new_key] = v
    return flat

for root, dirs, files in os.walk(folder_path):
    for file in files:
        if file.endswith(".jsonl"):
            file_path = os.path.join(root, file)
            json_chunks = []
            with open(file_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        chunk = json.loads(line)
                        doc = Document(
                            page_content=chunk.get("text", ""),  # text -> page_content
                            #metadata={k: v for k, v in chunk.items() if k == "metadata"}
                            metadata=flatten_metadata({k: v for k, v in chunk.items() if k != "text"}) 
                        )
                        json_chunks.append(doc)
            json_files_chunk.extend(json_chunks)

print(f"Total documents loaded: {len(json_files_chunk)}")

# %%
vectorstore = Chroma.from_documents(
    documents=json_files_chunk,
    embedding=embedding_model,
    persist_directory="chroma_db_2"
)

retriever = vectorstore.as_retriever(
    search_kwargs={"k": 5},
    search_type="mmr"
)

# %%
query = "is there a way i can pad my string in python?"

results = retriever.invoke(query)

for doc in results:
    print(doc.page_content)
    print(doc.metadata)
    print("-----")

# %%
query = "Why am I getting IndentationError in Python?"

results = retriever.invoke(query)

for doc in results:
    print(doc.page_content)
    print(doc.metadata)
    print("-----")

# %%
query = "How do I debug a Python script step by step"

results = retriever.invoke(query)

for doc in results:
    print(doc.page_content)
    print(doc.metadata)
    print("-----")

# %%
query = "My Python script works locally but fails in production — why"

results = retriever.invoke(query)

for doc in results:
    print(doc.page_content)
    print(doc.metadata)
    print("-----")

# %% [markdown]
# #### PDF CHUNKING

# %%
import pdfplumber
import re

def is_useless_page(text):
    if not text:
        return True

    text_lower = text.lower()

    noise_keywords = ["table of contents", "index"]

    if any(k in text_lower for k in noise_keywords):
        return True

    if re.search(r"\.{5,}", text):
        return True

    return False


def extract_lines_with_style(pdf_path):
    pages_data = []

    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:

            words = page.extract_words(
                extra_attrs=["size", "fontname"]
            )

            lines = {}

            for w in words:
                key = round(w["top"], 1)  # group by line position

                if key not in lines:
                    lines[key] = []

                lines[key].append(w)

            structured_lines = []

            for _, words_in_line in sorted(lines.items()):

                text = " ".join([w["text"] for w in words_in_line])
                avg_size = sum(w.get("size", 10) for w in words_in_line) / len(words_in_line)

                structured_lines.append({
                    "text": text,
                    "size": avg_size
                })

            pages_data.append(structured_lines)

    return pages_data

def detect_topics(pages_data):
    all_sections = []

    current_topic = None
    current_content = []

    for page in pages_data:
        for line in page:

            text = line["text"]
            size = line["size"]

            # 🔥 heading detection (font size heuristic)
            if size >= 13 and len(text.split()) < 12:
                # flush previous section
                if current_topic and current_content:
                    all_sections.append({
                        "title": current_topic,
                        "content": "\n".join(current_content)
                    })

                current_topic = text
                current_content = []

            else:
                if current_topic:
                    current_content.append(text)

    # last section
    if current_topic and current_content:
        all_sections.append({
            "title": current_topic,
            "content": "\n".join(current_content)
        })

    return all_sections

splitter = RecursiveCharacterTextSplitter(
    chunk_size=800,
    chunk_overlap=140
)

def build_chunks(sections, pdf_path):
    final_chunks = []
    filename = os.path.basename(pdf_path)

    for sec in sections:
        chunks = splitter.split_text(sec["content"])
        for chunk in chunks:
            final_chunks.append(
                Document(
                    page_content=chunk,  
                    metadata={
                        "type": "text",
                        "topic": sec["title"],
                        "source": filename
                    }
                )
            )
    return final_chunks


def process_all_pdfs(folder_path):
    all_chunks = []

    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.endswith(".pdf"):
                pdf_path = os.path.join(root, file)
                print(f"Processing: {pdf_path}")

                pages_data = extract_lines_with_style(pdf_path)
                sections = detect_topics(pages_data)
                chunks = build_chunks(sections, pdf_path)
                all_chunks.extend(chunks)

                print(f"  → {len(chunks)} chunks extracted")

    print(f"\nTotal chunks from all PDFs: {len(all_chunks)}")
    return all_chunks


pdf_folder = r"data\pdf"
pdf_files_chunk = process_all_pdfs(pdf_folder)

# %%
def remove_noise_chunks(chunks):
    noise_topics = ["disclaimer", "table of contents", "index"]

    before = len(chunks)
    cleaned_chunks = [
        chunk for chunk in chunks
        if chunk.metadata["topic"].strip().lower() not in noise_topics  # ✅ .metadata not ["metadata"]
    ]
    after = len(cleaned_chunks)

    print(f"Removed {before - after} noise chunks | Remaining: {after}")
    return cleaned_chunks

pdf_files_chunk = remove_noise_chunks(pdf_files_chunk)

# %%
pdf_files_chunk

# %%
vectorstore = Chroma.from_documents(
    documents=pdf_files_chunk,
    embedding=embedding_model,
    persist_directory="chroma_db_3"
)

retriever = vectorstore.as_retriever(
    search_kwargs={"k": 5},
    search_type="mmr"
)

# %%
query = "How do I fix session timeout issues in Oracle applications"

results = retriever.invoke(query)

for doc in results:
    print(doc.page_content)
    print(doc.metadata)
    print("-----")

# %%
query = "Why is my Oracle SQL query running slow"

results = retriever.invoke(query)

for doc in results:
    print(doc.page_content)
    print(doc.metadata)
    print("-----")

# %%
query = "Why is my Oracle microservice returning 500 internal server error"

results = retriever.invoke(query)

for doc in results:
    print(doc.page_content)
    print(doc.metadata)
    print("-----")

# %%
query = "Why is my Cisco VPN not connecting?"

results = retriever.invoke(query)

for r in results:
    print(r.page_content)
    print(r.metadata)
    print("-"*50)

# %% [markdown]
# ### Combining All Chunks

# %%
all_chunks = []
all_chunks.extend(pdf_files_chunk)
all_chunks.extend(text_files_chunk)
all_chunks.extend(json_files_chunk)

# %% [markdown]
# ### Embedding and Storing

# %%
vectorstore = Chroma.from_documents(
    documents=all_chunks,
    embedding=embedding_model,
    persist_directory="chroma_db"
)

# %% [markdown]
# ### Retrievers

# %%
#Simple retriever
simple_retriever = vectorstore.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 5}
)

#mmr retriever
mmr_retriever=vectorstore.as_retriever(
    search_type="mmr",
    search_kwargs={"k":5}
)

query = "help me debug my python script"

results = simple_retriever.invoke(query)

for r in results:
    print(r.page_content)
    print(r.metadata)
    print("-"*50)


# %%
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers.ensemble import EnsembleRetriever
# ─────────────────────────────────────────
# 3. Retriever — semantic + topic filter
# ─────────────────────────────────────────
def get_doc_topic(doc):
    """Safely extract topic from doc metadata."""
    meta = doc.metadata
    return (
        meta.get("topic", "") or 
        meta.get("metadata_topic", "")
    ).strip().lower()


def get_retriever(k=5):
    #search_kwargs = {"k": k}
    #if topic:
    #    search_kwargs["filter"] = {"topic": {"$eq": topic}}
    return vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={"k":k}
    )


def get_hybrid_retriever(topic=None, k=5):
    bm25_docs = all_chunks

    if topic:
        topic_lower = topic.strip().lower()
        bm25_docs = [
            doc for doc in all_chunks
            if get_doc_topic(doc).lower() == topic_lower  # ✅ clean and valid
        ]

    bm25_retriever = BM25Retriever.from_documents(bm25_docs)
    bm25_retriever.k = k

    semantic_retriever = get_retriever(k=k)

    return EnsembleRetriever(
        retrievers=[bm25_retriever, semantic_retriever],
        weights=[0.2, 0.8]
    )


# ─── Test ───
query = "how to fix oracle connection timeout"

#results = get_hybrid_retriever(k=5).invoke(query)

results = get_hybrid_retriever(topic="debugging", k=5).invoke(query)

for doc in results:
    topic  = doc.metadata.get("topic") or doc.metadata.get("metadata_topic", "N/A")
    source = doc.metadata.get("source") or doc.metadata.get("metadata_source", "N/A")
    print(f"Topic  : {topic}")
    print(f"Source : {source}")
    print(f"Content: {doc.page_content}\n")

# %% [markdown]
# ### Rag Implementation

# %%
def detect_topic_from_query(query, topic_list, llm):
    prompt = f"""
You are a topic classifier. Given a user query and a list of available topics, 
return ONLY the most relevant topic names from the list as a JSON array.
Return maximum 3 topics. Return empty array [] if nothing matches.
Do not explain. Do not create or invent new topics.
You MUST only return topics that exist EXACTLY in the provided list.

Available Topics:
{json.dumps(topic_list)}

User Query: {query}

Response (JSON array only, values must match exactly from the list above):
"""
    response = llm.invoke(prompt).strip()

    try:
        matched_topics = json.loads(response)

        # ✅ validate — remove any topic not in topic_list
        validated_topics = [t for t in matched_topics if t in topic_list]

        #if len(validated_topics) != len(matched_topics):
            #print(f"⚠️ Hallucinated topics removed: {set(matched_topics) - set(validated_topics)}")

        #print(f"Validated topics: {validated_topics}")
        return validated_topics

    except json.JSONDecodeError:
        #print(f"⚠️ Could not parse LLM response: {response}")
        return []

# %%
topic_list = []
for chunk in all_chunks:
    topic = chunk.metadata.get("topic") or chunk.metadata.get("metadata_topic")
    if topic and topic not in topic_list:  
        topic_list.append(topic)

print(f"Total unique topics: {len(topic_list)}")

# %%
# ─────────────────────────────────────────
# 3. RAG Chain — Retriever + LLM
# ─────────────────────────────────────────
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

prompt = ChatPromptTemplate.from_template("""
You are a helpful IT support assistant made for troubleshooting and SOPs providing. 
Answer the user's question based only on the provided context.
If the answer is not in the context, say "I don't have enough information to answer this."
Answer in step wise or point wise
Context:
{context}

Question:
{question}

Answer:
""")
def format_docs(docs):
    return "\n\n".join([doc.page_content for doc in docs])

def rag_answer(query):
    topics_lst=detect_topic_from_query(query,topic_list,llm)
    print(topics_lst)
    if len(topics_lst)!=0:
        results = get_hybrid_retriever(topic=topics_lst[0].lower(), k=5).invoke(query)
        context = format_docs(results)
    else:
        results=mmr_retriever.invoke(query)
    final_prompt=prompt.invoke({
        "context":context,
        "question":query
    })
    response=llm.invoke(final_prompt)
    print(response)
    
rag_answer("help me debugg a python code")



# %%
import pickle
with open("all_chunks.pkl", "wb") as f:
    pickle.dump(all_chunks, f)

# %%
