import os
import json
from typing import List
from pathlib import Path
from dotenv import load_dotenv
from pyaml import yaml
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain_core.prompts import (
    ChatPromptTemplate, 
    PromptTemplate, 
    FewShotPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate
)
from langchain_core.example_selectors import SemanticSimilarityExampleSelector
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.messages import BaseMessage, SystemMessage, HumanMessage, AIMessage
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

BASE = Path(__file__).parent.parent

load_dotenv(BASE / ".env", override=True)
MODEL = os.getenv("OPENAI_API_MODEL", "gpt-5")

with open(BASE / "data/style_guide.yaml", "r", encoding="utf-8") as f:
    STYLE = yaml.safe_load(f)

with open(BASE / "data/few_shots.json", "r", encoding="utf-8") as f:
    FEW_SHOTS = json.load(f)

with open(BASE / "data/faq.json", "r", encoding="utf-8") as f:
    FAQ = json.load(f)

# Setup Few-Shot Selector
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
example_prompt = PromptTemplate.from_template("Вопрос: {user}\nОтвет: {assistant}")
example_selector = SemanticSimilarityExampleSelector.from_examples(
    examples=FEW_SHOTS,
    embeddings=embeddings,
    vectorstore_cls=Chroma,
    k=3
)
few_shot_prompt = FewShotPromptTemplate(
    example_selector=example_selector,
    example_prompt=example_prompt,
    prefix="Примеры хороших ответов:",
    suffix="",
    input_variables=["question"]
)

class BrandResponse(BaseModel):
    answer: str = Field(description="Ответ на вопрос пользователя")
    tone: str = Field(description="Анализ тональности: совпадает ли тон (да/нет) + краткое пояснение")
    actions: List[str] = Field(description="Список действий для пользователя (шаги)")

# Custom memory with automatic system prompt prepending
class MemoryWithSystemPrepend(BaseChatMessageHistory):
    def __init__(self, system_prompt: str):
        self.system_prompt = system_prompt
        self._messages = []  # Store only conversation (without system)

    @property
    def messages(self):
        """When requesting history, add system at the beginning"""
        return [SystemMessage(content=self.system_prompt)] + self._messages

    def add_message(self, message: BaseMessage):
        if not isinstance(message, SystemMessage):
            self._messages.append(message)

    def clear(self):
        self._messages = []

def format_faq(faq_data: List[dict]) -> str:
    """Convert FAQ list to readable string format"""
    faq_entries = [f"В: {item['q']}\nО: {item['a']}" for item in faq_data]
    return "\n\n".join(faq_entries)

LLM = ChatOpenAI(model=MODEL, temperature=0)

def ask(question: str, memory: MemoryWithSystemPrepend = None) -> BrandResponse:
    # Generate dynamic examples
    examples_text = few_shot_prompt.format(question=question)
    
    system_parts = [
        f"Ты — помощник бренда {STYLE['brand']}.",
        f"Тон: {STYLE['tone']['persona']}.",
        f"Избегай: {', '.join(STYLE['tone']['avoid'])}.",
        f"Обязательно: {', '.join(STYLE['tone']['must_include'])}.",
        "",
        examples_text
    ]

    system_parts.append(f"\nКОНТЕКСТ (FAQ/Статусы):\n{format_faq(FAQ)}")

    system_prompt = "\n".join(system_parts)

    # Create memory if not provided
    if memory is None:
        memory = MemoryWithSystemPrepend(system_prompt)
    else:
        # Update memory's system prompt
        memory.system_prompt = system_prompt
    
    # Add user message
    memory.add_message(HumanMessage(content=question))
    # Invoke with full history
    response = LLM.with_structured_output(BrandResponse).invoke(memory.messages)
    # Add AI response to memory
    memory.add_message(AIMessage(content=response.model_dump_json()))
    return response

def get_prompt(data, prompt_name, version=None):
    current = data[prompt_name]["current"]
    if not version:
        version = current
    return data[prompt_name]["versions"].get(version)