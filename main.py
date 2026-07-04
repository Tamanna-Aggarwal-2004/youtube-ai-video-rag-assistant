import re

from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api._errors import (
    NoTranscriptFound,
    TranscriptsDisabled,
)

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

from langchain_groq import ChatGroq

from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import (
    RunnableLambda,
    RunnableParallel,
    RunnablePassthrough,
)

from dotenv import load_dotenv
import os

load_dotenv()

# -----------------------------
# LLM
# -----------------------------

llm = ChatGroq(
    model="llama-3.1-8b-instant",
    api_key=os.getenv("GROQ_API_KEY"),
    temperature=0
)

# -----------------------------
# Extract Video ID
# -----------------------------

def extract_video_id(url):
    pattern = r"(?:v=|\/)([0-9A-Za-z_-]{11}).*"
    match = re.search(pattern, url)

    if match:
        return match.group(1)

    raise ValueError("Invalid YouTube URL")


# -----------------------------
# Fetch Transcript
# -----------------------------

def get_transcript(video_id):

    api = YouTubeTranscriptApi()

    transcript = api.fetch(video_id, languages=["en"])

    text = " ".join(chunk.text for chunk in transcript)

    return text


# -----------------------------
# Build Retriever
# -----------------------------

def build_retriever(text):

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )

    docs = splitter.create_documents([text])

    embedding = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    vectorstore = FAISS.from_documents(docs, embedding)

    return vectorstore.as_retriever(
        search_kwargs={"k":4}
    )


# -----------------------------
# Prompt
# -----------------------------

prompt = PromptTemplate(
    template="""
You are an AI assistant.

Answer ONLY from the transcript.

If the answer cannot be found in the transcript,
reply exactly:

"I couldn't find that information in this video. Please ask questions related to the uploaded video."

Context:
{context}

Question:
{question}
""",
    input_variables=["context","question"]
)


# -----------------------------
# LCEL
# -----------------------------

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


parser = StrOutputParser()


# -----------------------------
# User Input
# -----------------------------

url = input("Paste YouTube URL:\n")

video_id = extract_video_id(url)

transcript = get_transcript(video_id)

retriever = build_retriever(transcript)

rag_chain = (
    RunnableParallel(
        {
            "context": retriever | RunnableLambda(format_docs),
            "question": RunnablePassthrough()
        }
    )
    | prompt
    | llm
    | parser
)


# -----------------------------
# Summary
# -----------------------------

summary_prompt = f"""
Analyze the following transcript and generate:

1. Executive Summary

2. Key Points

3. Objectives

4. Important Concepts

5. Important Facts

6. Conclusion

Transcript:

{transcript}
"""

print("="*80)
print("VIDEO ANALYSIS")
print("="*80)

print(llm.invoke(summary_prompt).content)


# -----------------------------
# Chat Loop
# -----------------------------

print("\nNow you can ask questions from this video.")
print("Type exit to quit.\n")

while True:

    question = input("You : ")

    if question.lower()=="exit":
        break

    response = rag_chain.invoke(question)

    print("\nAI :",response)
    print()