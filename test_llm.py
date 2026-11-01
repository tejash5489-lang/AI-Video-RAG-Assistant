from dotenv import load_dotenv
load_dotenv()
from core.llm import get_llm

print(get_llm().invoke("Say hi").content)