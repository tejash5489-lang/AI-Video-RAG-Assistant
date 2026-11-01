import os
from langchain_groq import ChatGroq
from langchain_core.rate_limiters import InMemoryRateLimiter

_rate_limiter = InMemoryRateLimiter(
    requests_per_second=0.4,
    check_every_n_seconds=0.1,
    max_bucket_size=1,
)
# Use openai/gpt-oss-120b for best quality, or openai/gpt-oss-20b if you want faster responses and looser limits.

def get_llm(temperature: float = 0.3):
    return ChatGroq(
        model="openai/gpt-oss-120b",
        api_key=os.getenv("GROQ_API_KEY"),
        temperature=temperature,
        rate_limiter=_rate_limiter,
        max_retries=6,
    )
    
    
# import os
# from langchain_mistralai import ChatMistralAI
# from langchain_core.rate_limiters import InMemoryRateLimiter

# # One limiter shared by every module (about 1 request every 2 seconds)
# _rate_limiter = InMemoryRateLimiter(
#     requests_per_second=0.5,
#     check_every_n_seconds=0.1,
#     max_bucket_size=1,
# )

# def get_llm(temperature: float = 0.3):
#     return ChatMistralAI(
#         model="mistral-small-latest",
#         mistral_api_key=os.getenv("MISTRAL_API_KEY"),
#         temperature=temperature,
#         rate_limiter=_rate_limiter,
#         max_retries=6,
#     )