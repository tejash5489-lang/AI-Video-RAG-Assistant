import re
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_text_splitters import RecursiveCharacterTextSplitter
from core.llm import get_llm

# Transcripts up to this length go into the prompt whole (no retrieval needed)
FULL_CONTEXT_LIMIT = 12000

STOPWORDS = {
    "the", "and", "for", "that", "this", "with", "was", "were", "are", "what",
    "who", "when", "where", "why", "how", "did", "does", "about", "from",
    "have", "has", "had", "you", "they", "them", "his", "her", "their", "can",
    "will", "would", "should", "could", "said", "say", "any", "all",
}


def _tokens(text: str) -> set:
    words = re.findall(r"[a-z0-9']+", text.lower())
    return {w for w in words if len(w) > 2 and w not in STOPWORDS}


def _make_context_fn(transcript: str):
    # Short transcript: send everything, no embeddings, no downloads
    if len(transcript) <= FULL_CONTEXT_LIMIT:
        return lambda question: transcript

    # Long transcript: simple keyword retrieval, pure Python
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    chunks = splitter.split_text(transcript)
    chunk_tokens = [_tokens(c) for c in chunks]

    def retrieve(question: str) -> str:
        q = _tokens(question)
        ranked = sorted(
            range(len(chunks)),
            key=lambda i: len(q & chunk_tokens[i]),
            reverse=True,
        )[:5]
        return "\n\n".join(chunks[i] for i in sorted(ranked))

    return retrieve


def build_rag_chain(transcript: str):
    context_fn = _make_context_fn(transcript)
    llm = get_llm()

    prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            """You are an expert meeting assistant. Answer the user's question
based ONLY on the meeting transcript context provided below.

If the answer is not found in the context, say:
"I could not find this information in the meeting transcript."

Always be concise and precise. If quoting someone, mention it clearly.

Context from meeting transcript:
{context}""",
        ),
        ("human", "{question}"),
    ])

    return (
        {"context": RunnableLambda(context_fn), "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )


def ask_question(rag_chain, question: str) -> str:
    print(f"Question : {question}")
    answer = rag_chain.invoke(question)
    print(f"answer : {answer}")
    return answer