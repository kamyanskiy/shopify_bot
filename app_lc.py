from dotenv import load_dotenv
import logging
import os
import json
import sys
from pathlib import Path

# Add src to path to import brand_chain
sys.path.append(str(Path(__file__).parent / "src"))
from brand_chain import ask, MemoryWithSystemPrepend

load_dotenv()

# Настраиваем логирование
logging.basicConfig(
    filename="logs/chat_session.log", level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S",
    encoding="utf-8"
)


class СliBot:
    def __init__(self):
        # Создаём Хранилище истории для разных сессий
        self.store = {}

    # Метод для получения истории по session_id
    def get_session_memory(self, session_id: str):
        if session_id not in self.store:
            self.store[session_id] = MemoryWithSystemPrepend("")
        return self.store[session_id]

    def __call__(self, session_id):
        print(
            "Чат-бот поддержки магазина «Shoply» запущен! Можете задавать вопросы. \n - Для выхода введите 'выход'.\n - Для очистки контекста введите 'сброс'.\n")
        logging.info("=== New session ===")
        
        memory = self.get_session_memory(session_id)
        
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
                    self.store[session_id].clear()
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
                # Use the ask function from brand_chain
                response = ask(user_text, memory=memory)
                bot_reply = response.answer
                
                # Log structured response
                logging.info(f"Bot: {bot_reply}")
                logging.info(f"Tone check: {response.tone}")
                if response.actions:
                    logging.info(f"Actions: {response.actions}")
                
                # Print response (only answer to user)
                print(f"Бот: {bot_reply}")
                    
            except Exception as e:
                # Логируем и выводим ошибку, продолжаем чат
                logging.error(f"[error] {e}")
                print(f"[Ошибка] {e}")
                continue

if __name__ == "__main__":
    # Загружаем Orders
    orders_path = os.path.join(os.path.dirname(__file__), "data/orders.json")
    orders_data = {}
    try:
        with open(orders_path, "r", encoding="utf-8") as f:
            orders_data = json.load(f)
    except Exception as e:
        logging.error(f"Не удалось загрузить Orders: {e}")

    bot = СliBot()
    bot("user_123")
