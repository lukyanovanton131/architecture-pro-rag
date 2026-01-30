# telegram_bot.py
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import Application, CommandHandler, MessageHandler, filters, CallbackContext, CallbackQueryHandler
import asyncio
from rag_bot import RAGBotWithPrompting
import logging
import json
from datetime import datetime

# Настройка логирования
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

class TelegramRAGBot:
    def __init__(self, token: str, rag_bot: RAGBotWithPrompting):
        """
        Инициализация Telegram-бота
        
        Args:
            token: токен бота от BotFather
            rag_bot: экземпляр RAG-бота
        """
        self.token = token
        self.rag_bot = rag_bot
        self.user_sessions = {}  # Хранение сессий пользователей
        self.application = None
        
    async def start(self, update: Update, context: CallbackContext) -> None:
        """Обработка команды /start"""
        user = update.effective_user
        welcome_text = f"""
🤖 Привет, {user.first_name}!

Я - интеллектуальный помощник с доступом к базе знаний.

**Что я умею:**
• Искать информацию в базе знаний
• Давать подробные ответы с источниками
• Объяснять свои рассуждения (Chain-of-Thought)

**Доступные команды:**
/help - показать справку
/search - поиск в базе знаний
/feedback - оставить отзыв
/settings - настройки

Просто напиши свой вопрос, и я постараюсь помочь!
        """
        
        keyboard = [
            [InlineKeyboardButton("🔍 Примеры запросов", callback_data="examples")],
            [InlineKeyboardButton("❓ Как пользоваться", callback_data="how_to_use")],
            [InlineKeyboardButton("⚙️ Настройки поиска", callback_data="settings")]
        ]
        reply_markup = InlineKeyboardMarkup(keyboard)
        
        await update.message.reply_text(welcome_text, reply_markup=reply_markup)
        
        # Инициализация сессии пользователя
        self.user_sessions[user.id] = {
            "last_query": None,
            "search_history": [],
            "settings": {"k_results": 5, "threshold": 0.3}
        }
    
    async def help_command(self, update: Update, context: CallbackContext) -> None:
        """Обработка команды /help"""
        help_text = """
**Справка по использованию бота:**

**Основные функции:**
1. **Поиск информации** - просто задайте вопрос
2. **Подробные ответы** - бот использует цепочку рассуждений
3. **Цитирование источников** - каждый ответ содержит ссылки на источники

**Техники промптинга:**
• **Few-shot learning** - бот обучается на примерах
• **Chain-of-Thought** - бот объясняет свои рассуждения
• **Контекстуальный поиск** - поиск в векторной базе знаний

**Примеры запросов:**
• "Объясни принцип работы нейронных сетей"
• "Какие типы машинного обучения существуют?"
• "Расскажи о..."

Для начала работы просто напишите свой вопрос!
        """
        await update.message.reply_text(help_text, parse_mode='Markdown')
    
    async def handle_message(self, update: Update, context: CallbackContext) -> None:
        """Обработка текстовых сообщений"""
        user_id = update.effective_user.id
        query = update.message.text
        
        # Сохранение в истории
        if user_id in self.user_sessions:
            self.user_sessions[user_id]["last_query"] = query
            self.user_sessions[user_id]["search_history"].append({
                "query": query,
                "timestamp": datetime.now().isoformat()
            })
        
        # Отправка статуса "печатает"
        await update.message.chat.send_action(action="typing")
        
        try:
            # Генерация ответа
            response = self.rag_bot.enhanced_generate(query)
            
            # Разбивка длинных сообщений (Telegram имеет ограничение)
            if len(response) > 4000:
                chunks = [response[i:i+4000] for i in range(0, len(response), 4000)]
                for chunk in chunks:
                    await update.message.reply_text(chunk, parse_mode='Markdown')
            else:
                await update.message.reply_text(response, parse_mode='Markdown')
                
        except Exception as e:
            logger.error(f"Ошибка при обработке запроса: {e}")
            await update.message.reply_text(
                "Извините, произошла ошибка при обработке вашего запроса. Пожалуйста, попробуйте еще раз."
            )
    
    async def search_command(self, update: Update, context: CallbackContext) -> None:
        """Обработка команды /search БЕЗ вызова handle_message"""
        if not context.args:
            await update.message.reply_text(
                "Пожалуйста, укажите поисковый запрос после команды /search\n"
                "Пример: /search принципы машинного обучения"
            )
            return
        
        query = " ".join(context.args)
        
        # Отправка статуса "печатает"
        await update.message.chat.send_action(action="typing")
        
        try:
            # Генерация ответа НАПРЯМУЮ через rag_bot
            response = self.rag_bot.enhanced_generate(query)
            
            # Разбивка длинных сообщений
            if len(response) > 4000:
                chunks = [response[i:i+4000] for i in range(0, len(response), 4000)]
                for i, chunk in enumerate(chunks):
                    await update.message.reply_text(
                        chunk, 
                        parse_mode='Markdown',
                        disable_web_page_preview=True
                    )
            else:
                await update.message.reply_text(
                    response, 
                    parse_mode='Markdown',
                    disable_web_page_preview=True
                )
                
        except Exception as e:
            logger.error(f"Ошибка в /search: {e}")
            await update.message.reply_text(
                "Извините, произошла ошибка при обработке запроса. "
                "Проверьте, что сервер Ollama запущен и модель загружена."
            )
    
    async def button_callback(self, update: Update, context: CallbackContext) -> None:
        """Обработка нажатий на inline-кнопки"""
        query = update.callback_query
        await query.answer()
        
        if query.data == "examples":
            examples_text = """
**Примеры запросов:**

📚 **Технические вопросы:**
• "Объясни разницу между supervised и unsupervised learning"
• "Что такое градиентный спуск?"
• "Какие архитектуры нейронных сетей существуют?"

💼 **Бизнес-вопросы:**
• "Как применить машинное обучение в бизнесе?"
• "Какие метрики важны для оценки моделей?"
• "Что такое MLOps?"

🔍 **Поиск конкретной информации:**
• "Найди информацию о [конкретная тема]"
• "Какие есть подходы к [проблема]"
• "Объясни [концепция] на примерах"
            """
            await query.edit_message_text(examples_text, parse_mode='Markdown')
            
        elif query.data == "how_to_use":
            usage_text = """
**Как пользоваться ботом:**

1. **Задавайте конкретные вопросы** - чем конкретнее вопрос, тем точнее ответ
2. **Используйте ключевые слова** - это улучшит поиск в базе знаний
3. **Запрашивайте объяснения** - бот умеет подробно объяснять свои ответы
4. **Проверяйте источники** - каждый ответ содержит ссылки на материалы

**Советы:**
• Начинайте с общих вопросов, затем уточняйте
• Используйте команду /search для прямого поиска
• Сообщайте о неточностях через /feedback
            """
            await query.edit_message_text(usage_text, parse_mode='Markdown')
            
        elif query.data == "settings":
            settings_text = """
**Настройки поиска:**

Текущие настройки:
• Количество результатов: 5
• Порог релевантности: 0.3

Изменить настройки можно через команды:
/set_results [число] - установить количество результатов
/set_threshold [0.1-0.9] - установить порог релевантности
            """
            await query.edit_message_text(settings_text, parse_mode='Markdown')
    
    async def set_results(self, update: Update, context: CallbackContext) -> None:
        """Установка количества результатов"""
        user_id = update.effective_user.id
        
        if not context.args:
            await update.message.reply_text(
                "Использование: /set_results [число]\n"
                "Пример: /set_results 10"
            )
            return
        
        try:
            k = int(context.args[0])
            if 1 <= k <= 20:
                if user_id in self.user_sessions:
                    self.user_sessions[user_id]["settings"]["k_results"] = k
                await update.message.reply_text(f"✅ Количество результатов установлено: {k}")
            else:
                await update.message.reply_text("Пожалуйста, укажите число от 1 до 20")
        except ValueError:
            await update.message.reply_text("Пожалуйста, укажите корректное число")
    
    async def feedback_command(self, update: Update, context: CallbackContext) -> None:
        """Обработка команды /feedback"""
        feedback_text = """
**Обратная связь**

Пожалуйста, отправьте ваш отзыв или предложение одним сообщением.

Что можно сообщить:
• Точность ответов
• Полноту информации
• Технические проблемы
• Предложения по улучшению

Ваш отзыв поможет улучшить бота!
        """
        await update.message.reply_text(feedback_text)
        
        # Сохраняем, что пользователь находится в режиме обратной связи
        user_id = update.effective_user.id
        if user_id in self.user_sessions:
            self.user_sessions[user_id]["awaiting_feedback"] = True
    
    async def save_feedback(self, update: Update, context: CallbackContext) -> None:
        """Сохранение отзыва пользователя"""
        user_id = update.effective_user.id
        
        if user_id in self.user_sessions and self.user_sessions[user_id].get("awaiting_feedback"):
            feedback = update.message.text
            
            # Сохранение отзыва в файл
            with open("feedback.json", "a") as f:
                feedback_data = {
                    "user_id": user_id,
                    "username": update.effective_user.username,
                    "feedback": feedback,
                    "timestamp": datetime.now().isoformat()
                }
                json.dump(feedback_data, f)
                f.write("\n")
            
            await update.message.reply_text("✅ Спасибо за ваш отзыв!")
            self.user_sessions[user_id]["awaiting_feedback"] = False
    
    async def error_handler(self, update: Update, context: CallbackContext) -> None:
        """Обработка ошибок"""
        logger.error(f"Ошибка: {context.error}")
        
        if update and update.effective_message:
            await update.effective_message.reply_text(
                "Произошла ошибка. Пожалуйста, попробуйте еще раз или обратитесь к администратору."
            )
    
    async def run(self):
        """Запуск бота с корректной обработкой жизненного цикла"""
        # Создание приложения
        self.application = Application.builder().token(self.token).build()
        
        # Регистрация обработчиков команд
        self.application.add_handler(CommandHandler("start", self.start))
        self.application.add_handler(CommandHandler("help", self.help_command))
        self.application.add_handler(CommandHandler("search", self.search_command))
        self.application.add_handler(CommandHandler("set_results", self.set_results))
        self.application.add_handler(CommandHandler("feedback", self.feedback_command))
        
        # Регистрация обработчиков сообщений (важен порядок!)
        # Сначала обработчик обратной связи (только когда ожидаем отзыв)
        self.application.add_handler(MessageHandler(
            filters.TEXT & filters.ChatType.PRIVATE & filters.UpdateFilter(lambda u: 
                u.effective_user.id in self.user_sessions and 
                self.user_sessions[u.effective_user.id].get("awaiting_feedback", False)
            ),
            self.save_feedback
        ))
        
        # Затем обычные сообщения (игнорируем команды)
        self.application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, self.handle_message))
        
        # Регистрация обработчика inline-кнопок
        self.application.add_handler(CallbackQueryHandler(self.button_callback))
        
        # Регистрация обработчика ошибок
        self.application.add_error_handler(self.error_handler)
        
        logger.info("🔄 Инициализация приложения...")
        
        # Корректная последовательность жизненного цикла приложения
        try:
            # Инициализация
            await self.application.initialize()
            await self.application.start()
            
            # Получаем информацию о боте ДО запуска опроса
            bot_info = await self.application.bot.get_me()
            logger.info(f"✅ Бот успешно запущен! @{bot_info.username} (ID: {bot_info.id})")
            print(f"\n✅ Бот запущен! Username: @{bot_info.username}")
            print("Нажмите Ctrl+C для остановки...\n")
            
            # Запуск опроса обновлений
            await self.application.updater.start_polling(
                allowed_updates=Update.ALL_TYPES,
                drop_pending_updates=True  # Игнорировать старые обновления при старте
            )
            
            # Ожидание неопределённое время (до прерывания)
            await asyncio.Event().wait()
            
        except asyncio.CancelledError:
            logger.info("🛑 Получен сигнал отмены задачи")
            raise
        finally:
            logger.info("🔄 Остановка приложения...")
            # Корректная остановка всех компонентов
            if self.application.updater.running:
                await self.application.updater.stop()
            if self.application.running:
                await self.application.stop()
            await self.application.shutdown()
            logger.info("✅ Приложение остановлено корректно")