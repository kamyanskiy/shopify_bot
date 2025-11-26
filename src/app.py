from dotenv import load_dotenv
import logging
import os
import json

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_openai import ChatOpenAI


load_dotenv()

# Настраиваем логирование
logging.basicConfig(
    filename="logs/chat_session.log", level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S",
    encoding="utf-8"
)


class СliBot:
    def __init__(self, model_name, system_prompt="Ты полезный ассистент. Ты сотрудник поддержки магазина «Shoply». Ты всегда дружелюбен и вежлив."):
        
        self.chat_model = ChatOpenAI(
            model=model_name,
            temperature=0,
            timeout=15
        )

        # Создаём Хранилище истории
        self.store = {}

        # Создаем шаблон промпта
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            MessagesPlaceholder(variable_name="history"),
            ("human", "{question}"),
        ])

        # Создаём цепочку
        self.chain = self.prompt | self.chat_model

        # Создаём цепочку с историей
        self.chain_with_history = RunnableWithMessageHistory(
            self.chain,
            self.get_session_history,
            input_messages_key="question",
            history_messages_key="history",
        )


    # Метод для получения истории по session_id
    def get_session_history(self, session_id: str):
        if session_id not in self.store:
            self.store[session_id] = InMemoryChatMessageHistory()
        return self.store[session_id]

    def __call__(self, session_id):
        print(
            "Чат-бот поддержки магазина «Shoply» запущен! Можете задавать вопросы. \n - Для выхода введите 'выход'.\n - Для очистки контекста введите 'сброс'.\n")
        logging.info("=== New session ===")
        while True:
            try:
                user_text = input("Вы: ").strip()
            except (KeyboardInterrupt, EOFError):
                print("\nБот: Завершение работы.")
                break
            if not user_text:
                continue

            logging.info(f"User: {user_text}")
            msg = user_text.lower()
            if msg in ("выход", "стоп", "конец"):
                print("Бот: До свидания!")
                logging.info("Пользователь завершил сессию. Сессия окончена.")
                break
            if msg == "сброс":
                if session_id in self.store:
                    del self.store[session_id]
                print("Бот: Контекст диалога очищен.")
                logging.info("Пользователь сбросил контекст.")
                continue

            # Обработка команды /order
            if user_text.startswith("/order"):
                parts = user_text.split()
                if len(parts) < 2:
                    print("Бот: Пожалуйста, укажите номер заказа. Пример: /order 12345")
                    continue
                
                order_id = parts[1]

                global orders_data  # просто используем глобально
                if order_id in orders_data:
                    order = orders_data[order_id]
                    status = order.get("status", "неизвестен")
                    details = []
                    if "eta_days" in order:
                        details.append(f"Ожидается через {order['eta_days']} дн.")
                    if "delivered_at" in order:
                        details.append(f"Доставлен: {order['delivered_at']}")
                    if "note" in order:
                        details.append(f"Примечание: {order['note']}")
                    
                    details_str = " ".join(details)
                    response = f"Заказ {order_id}: статус {status}. {details_str}"
                    print(f"Бот: {response}")
                    logging.info(f"Бот: {response}. Token usage: 0")
                else:
                    print(f"Бот: Заказ {order_id} не найден.")
                    logging.warning(f"Бот: Заказ {order_id} не найден. Token usage: 0")
                continue

            try:
                response = self.chain_with_history.invoke(
                    {"question": user_text},
                    {"configurable": {"session_id": session_id}}
                )
            except Exception as e:
                # Логируем и выводим ошибку, продолжаем чат
                logging.error(f"[error] {e}")
                print(f"[Ошибка] {e}")
                continue

            # Форматируем и выводим ответ
            bot_reply = response.content.strip()
            token_usage = self._get_token_usage(response)
            logging.info("Bot: %s. Token usage: %s", bot_reply, token_usage)
            print(f"Бот: {bot_reply}")

    def _get_token_usage(self, response):
        # Логируем использование токенов
        token_usage = 0
        if hasattr(response, 'response_metadata'):
            token_usage = response.response_metadata.get('token_usage', {})
        return token_usage
if __name__ == "__main__":
    model = os.getenv("OPENAI_API_MODEL", "x-ai/grok-4.1-fast:free")
    
    # Загружаем FAQ
    faq_path = os.path.join(os.path.dirname(__file__), "../data/faq.json")
    faq_text = ""
    try:
        with open(faq_path, "r", encoding="utf-8") as f:
            faq_data = json.load(f)
            faq_entries = [f"В: {item['q']}\nO: {item['a']}" for item in faq_data]
            faq_text = "\n\n".join(faq_entries)
    except Exception as e:
        logging.error(f"Не удалось загрузить FAQ: {e}")

    # Загружаем Orders
    orders_path = os.path.join(os.path.dirname(__file__), "../data/orders.json")
    orders_data = {}
    try:
        with open(orders_path, "r", encoding="utf-8") as f:
            orders_data = json.load(f)
    except Exception as e:
        logging.error(f"Не удалось загрузить Orders: {e}")

    system_prompt = f"""Ты полезный ассистент. Ты всегда дружелюбен и вежлив. Отвечай подробно и по существу.
Используй следующую информацию из базы знаний (FAQ) для ответов на вопросы, если они подходят:

{faq_text}
"""

    bot = СliBot(
        model_name=model,
        system_prompt=system_prompt
    )
    bot("user_123")
