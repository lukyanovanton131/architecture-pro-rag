# main.py
import os
from dotenv import load_dotenv
from rag_bot import RAGBotWithPrompting
from telegram_bot import TelegramRAGBot
import asyncio
# Загрузка переменных окружения
load_dotenv()

async def main():
    # Конфигурация
    INDEX_PATH = os.getenv("SEARCH_INDEX_PATH")
    TELEGRAM_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
    OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama2")
    OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")  # НОВОЕ
    
    # Валидация конфигурации
    if not TELEGRAM_TOKEN:
        raise ValueError("❌ Ошибка: Необходимо установить TELEGRAM_BOT_TOKEN в .env файле")
    
    print(f"🔧 Конфигурация:")
    print(f"   • Telegram Token: {'*' * len(TELEGRAM_TOKEN[:-4]) + TELEGRAM_TOKEN[-4:] if TELEGRAM_TOKEN else 'НЕТ'}")
    print(f"   • Ollama Model: {OLLAMA_MODEL}")
    print(f"   • Ollama URL: {OLLAMA_BASE_URL}")
    print(f"   • Index Path: {INDEX_PATH}")
    
    # Проверка существования индекса
    if not os.path.exists(INDEX_PATH):
        raise ValueError(f"❌ Ошибка: Векторный индекс не найден по пути: {INDEX_PATH}")
    
    # Инициализация RAG-бота С ПРОВЕРКОЙ OLLAMA
    print("\n🚀 Инициализация RAG-бота...")
    try:
        rag_bot = RAGBotWithPrompting(
            index_path=INDEX_PATH,
            ollama_model=OLLAMA_MODEL,
            ollama_base_url=OLLAMA_BASE_URL  # ПЕРЕДАЕМ URL
        )
    except RuntimeError as e:
        print(f"\n{str(e)}")
        print("\n🛑 ЗАПУСК БОТА ПРЕРВАН ИЗ-ЗА ОШИБКИ OLLAMA")
        return
    
    # Инициализация Telegram-бота
    print("🤖 Инициализация Telegram-бота...")
    telegram_bot = TelegramRAGBot(token=TELEGRAM_TOKEN, rag_bot=rag_bot)
    
    # Запуск бота
    print("\n✅ Все компоненты инициализированы. Запуск Telegram-бота...")
    await telegram_bot.run()

    
if __name__ == '__main__':
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nБот остановлен.")