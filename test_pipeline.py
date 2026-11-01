import sys, time
from dotenv import load_dotenv
load_dotenv()

from utils.audio_processor import process_input
from core.transcriber import transcribe_all
from core.summarizer import generate_title, summarize
from core.vector_store import build_vector_store


def step(name, fn):
    t = time.time()
    print(f"\n>>> {name} ...", flush=True)
    out = fn()
    print(f"<<< {name} done in {time.time() - t:.1f}s", flush=True)
    return out


source = sys.argv[1]
chunks = step("audio", lambda: process_input(source))
transcript = step("transcribe", lambda: transcribe_all(chunks, "english"))
print(transcript[:200], flush=True)
title = step("title", lambda: generate_title(transcript))
print(title, flush=True)
summary = step("summary", lambda: summarize(transcript))
step("vector store", lambda: build_vector_store(transcript))
print("\nALL STEPS OK")