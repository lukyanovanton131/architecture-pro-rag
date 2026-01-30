import os
import pickle
import json
import faiss
import numpy as np
from typing import List, Dict, Any, Optional  # ✅ Все типы импортированы
import requests
from requests.exceptions import ConnectionError, Timeout
from sentence_transformers import SentenceTransformer
from langchain_core.prompts import SystemMessagePromptTemplate
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_community.chat_models import ChatOllama
from langchain_core.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RAGBotWithPrompting:
    def __init__(self, index_path: str = "./knowledge_index", 
                 model_name: str = "all-MiniLM-L6-v2",
                 ollama_model: str = "llama2",
                 ollama_base_url: str = "http://localhost:11434"):
        """
        Инициализация RAG-бота с проверкой подключения к Ollama
        
        Args:
            index_path: путь к векторному индексу
            model_name: модель для эмбеддингов
            ollama_model: название модели Ollama
            ollama_base_url: базовый URL сервера Ollama
        """
        # Проверка подключения к Ollama ДО инициализации
        self._check_ollama_connection(ollama_base_url, ollama_model)
        
        # Загрузка модели для эмбеддингов
        self.embedding_model = SentenceTransformer(model_name)
        
        # Загрузка векторного индекса
        self.load_vector_index(index_path)
        
        # Инициализация LLM (Ollama) с указанием base_url
        self.llm = ChatOllama(
            model=ollama_model,
            base_url=ollama_base_url,  # КРИТИЧЕСКИ ВАЖНО!
            temperature=0.1,
            callbacks=[StreamingStdOutCallbackHandler()],
            streaming=True
        )
        
        # Few-shot примеры
        self.few_shot_examples = self._create_few_shot_examples()
        
        # System промпт
        self.system_prompt = self._create_system_prompt()
        
        logger.info(f"✅ RAG-бот инициализирован. Модель: {ollama_model} | URL: {ollama_base_url}")
    
    def _check_ollama_connection(self, base_url: str, model_name: str):
        """Проверка подключения к серверу Ollama и наличия модели"""
        try:
            # Проверка доступности сервера
            response = requests.get(f"{base_url}/api/tags", timeout=5)
            if response.status_code != 200:
                raise ConnectionError(f"Ollama вернул статус {response.status_code}")
            
            # Проверка наличия модели
            models_data = response.json()
            models = models_data.get("models", [])
            model_names = [m.get("name", "").split(":")[0] for m in models]
            
            if model_name not in model_names:
                logger.warning(f"⚠️ Модель '{model_name}' не найдена в Ollama.")
                logger.warning(f"   Доступные модели: {', '.join(model_names) if model_names else 'нет'}")
                logger.warning(f"   Загрузите модель командой: ollama pull {model_name}")
                raise ConnectionError(f"Модель '{model_name}' не загружена в Ollama")
            
            logger.info(f"✅ Ollama доступен по адресу {base_url}. Модель '{model_name}' найдена.")
            
        except (ConnectionError, Timeout, requests.RequestException) as e:
            error_msg = (
                f"\n❌ НЕ УДАЛОСЬ ПОДКЛЮЧИТЬСЯ К OLLAMA!\n"
                f"Причина: {str(e)}\n\n"
                f"РЕШЕНИЕ:\n"
                f"1. Запустите сервер Ollama в отдельном терминале: 'ollama serve'\n"
                f"2. Убедитесь, что сервер работает на {base_url}\n"
                f"3. Проверьте, что модель '{model_name}' загружена: 'ollama pull {model_name}'\n"
                f"4. Если используете удаленный сервер, установите переменную окружения:\n"
                f"   OLLAMA_BASE_URL=http://ваш-сервер:11434"
            )
            logger.error(error_msg)
            raise RuntimeError(error_msg) from e
    
    def load_vector_index(self, index_path: str):
        """Загрузка векторного индекса"""
        try:
            # Проверка существования пути
            if not os.path.exists(index_path):
                raise FileNotFoundError(f"Путь к индексу не существует: {index_path}")
            
            # Загрузка FAISS индекса
            index_file = os.path.join(index_path, "faiss_index.bin")
            if not os.path.exists(index_file):
                raise FileNotFoundError(f"FAISS индекс не найден: {index_file}")
            
            self.faiss_index = faiss.read_index(index_file)
            
            # Загрузка метаданных
            metadata_file = os.path.join(index_path, "metadata.pkl")
            if not os.path.exists(metadata_file):
                raise FileNotFoundError(f"Метаданные не найдены: {metadata_file}")
            
            with open(metadata_file, 'rb') as f:
                self.metadata = pickle.load(f)
            
            # Загрузка конфигурации
            config_file = os.path.join(index_path, "config.json")
            if os.path.exists(config_file):
                with open(config_file, 'r') as f:
                    self.config = json.load(f)
            else:
                self.config = {}
                logger.warning(f"Конфигурационный файл не найден: {config_file}")
            
            logger.info(f"✅ Индекс загружен. Векторов: {self.faiss_index.ntotal}")
            
        except Exception as e:
            logger.error(f"Ошибка при загрузке индекса: {e}")
            raise
    
    def _create_few_shot_examples(self) -> str:
        """Создание few-shot примеров для улучшения качества ответов"""
        examples = """
        Пример 1:
        Вопрос: "Какие основные принципы работы нейронных сетей?"
        Размышление: Нейронные сети основаны на принципах биологических нейронов. Основные принципы включают...
        Ответ: Основные принципы работы нейронных сетей: 1) Иерархическая обработка информации, 2) Обучение на примерах, 3) Распределенное представление знаний, 4) Способность к обобщению.
        
        Пример 2:
        Вопрос: "Что такое машинное обучение?"
        Размышление: Машинное обучение - это подраздел искусственного интеллекта, который позволяет компьютерам...
        Ответ: Машинное обучение - это область искусственного интеллекта, которая дает компьютерам способность учиться на данных без явного программирования. Основные типы: supervised, unsupervised и reinforcement learning.
        """
        return examples
    
    def _create_system_prompt(self) -> str:
        """Создание system промпта с инструкциями"""
        system_prompt = """Ты — экспертный русскоязычный ассистент по базе знаний. Твоя задача - точно и подробно отвечать на вопросы НА РУССКОМ ЯЗЫКЕ, используя предоставленный контекст.

Инструкции по ответу:
1. ВСЕГДА начинай с размышления (Chain-of-Thought)
2. Используй ТОЛЬКО информацию из предоставленного контекста
3. Если в контексте нет информации для ответа, честно скажи об этом
4. Приводи конкретные примеры и детали из контекста
5. Ссылайся на источники информации
6. Структурируй ответ, используя маркированные списки и заголовки
7. Сохраняй профессиональный, но доступный тон
8. В ответе опирайся на значения секции [solution] в контексте

Формат ответа (СТРОГО СОБЛЮДАТЬ):
[Размышление] - твои мысли о том, как подойти к ответу
[Ответ] - подробный, структурированный ответ на вопрос на русском языке
[Источники] - ссылки на использованные материалы

Few-shot примеры:
{examples}

Теперь используй следующий контекст для ответа на вопрос пользователя:"""
        return system_prompt.format(examples=self.few_shot_examples)
    
    def search_similar_chunks(self, query: str, k: int = 5, threshold: float = 0.3) -> List[Dict[str, Any]]:
        """
        Поиск похожих чанков в векторной базе
        
        Args:
            query: поисковый запрос
            k: количество возвращаемых результатов
            threshold: минимальное косинусное сходство
            
        Returns:
            Список релевантных чанков с метаданными
        """
        # Генерация эмбеддинга для запроса
        query_embedding = self.embedding_model.encode([query], normalize_embeddings=True)
        query_embedding = np.array(query_embedding).astype('float32')
        
        # Поиск в FAISS
        distances, indices = self.faiss_index.search(query_embedding, k)
        
        # Фильтрация и форматирование результатов
        results = []
        for i, (distance, idx) in enumerate(zip(distances[0], indices[0])):
            if idx < len(self.metadata) and distance >= threshold:
                chunk = self.metadata[idx].copy()
                chunk['similarity'] = float(distance)
                chunk['rank'] = i + 1
                
                # Форматирование цитирования
                source_file = chunk.get('source_file', 'unknown')
                original_row = chunk.get('original_row', 0)
                chunk['citation'] = f"[{source_file}, строка {original_row + 1}]"
                results.append(chunk)
        
        return results
    
    def create_rag_prompt(self, query: str, context_chunks: List[Dict[str, Any]]) -> str:
        """
        Создание промпта для RAG с контекстом
        
        Args:
            query: вопрос пользователя
            context_chunks: релевантные чанки
            
        Returns:
            Форматированный промпт
        """
        # Форматирование контекста с источниками
        context_text = "\n\n".join([
            f"Фрагмент {i+1} {chunk['citation']}:\n{chunk['text']}"
            for i, chunk in enumerate(context_chunks)
        ])
        
        # Создание полного промпта
        full_prompt = f"""{self.system_prompt}

[КОНТЕКСТ]
{context_text}

[ВОПРОС ПОЛЬЗОВАТЕЛЯ]
{query}

[РАЗМЫШЛЕНИЕ И ОТВЕТ]"""
        
        return full_prompt
    
    def generate_response(self, query: str, use_chain_of_thought: bool = True) -> Dict[str, Any]:
        """
        Генерация ответа с использованием RAG и техник промптинга
        
        Args:
            query: вопрос пользователя
            use_chain_of_thought: использовать ли цепочку мыслей
            
        Returns:
            Словарь с ответом и метаданными
        """
        logger.info(f"Обработка запроса: {query}")
        
        try:
            # Поиск релевантных чанков
            context_chunks = self.search_similar_chunks(query, k=5)
            
            if not context_chunks:
                return {
                    "answer": "В базе знаний не найдено информации по данному вопросу.",
                    "sources": [],
                    "context_used": False
                }
            ###################
            answer_parts = ["На основе базы знаний я нашел следующую информацию:\n"]
            for i, result in enumerate(context_chunks, 1):
                answer_parts.append(f"{i}. {result['text']}")                
                answer_parts.append("")  # Пустая строка для разделения
                        
            logger.info(f"{"\n".join(answer_parts)}")          
            ###################

            
            # Создание промпта
            prompt = self.create_rag_prompt(query, context_chunks)
            logger.info(f"{prompt}") 
            # Генерация ответа через Ollama
            messages = [
                SystemMessage(content="Ты полезный ассистент, который дает точные и подробные ответы."),
                HumanMessage(content=prompt)
            ]
            
            response = self.llm.invoke(messages)
            answer = response.content
            
            # Извлечение источников
            sources = list(set([chunk['citation'] for chunk in context_chunks]))
            
            return {
                "answer": answer,
                "sources": sources,
                "context_chunks": context_chunks,
                "context_used": True
            }
            
        except Exception as e:
            error_str = str(e).lower()
            logger.error(f"Ошибка при генерации ответа: {e}")
            
            # Специальная обработка ошибок подключения к Ollama
            if any(keyword in error_str for keyword in ["connection", "connect", "refused", "timeout"]):
                user_message = (
                    "❌ Сервер Ollama недоступен!\n\n"
                    "Возможные причины:\n"
                    "• Сервер Ollama не запущен\n"
                    "• Модель не загружена в Ollama\n"
                    "• Неправильный адрес сервера\n\n"
                    "Что делать:\n"
                    "1. Запустите 'ollama serve' в отдельном терминале\n"
                    "2. Выполните 'ollama pull llama2' (или вашу модель)\n"
                    "3. Перезапустите бота"
                )
            else:
                user_message = f"Извините, произошла ошибка при генерации ответа: {str(e)[:150]}"
            
            return {
                "answer": user_message,
                "sources": [],
                "context_used": False
            }
    
    def enhanced_generate(self, query: str) -> str:
        """
        Улучшенная генерация с анализом и валидацией
        
        Args:
            query: вопрос пользователя
            
        Returns:
            Форматированный ответ
        """
        # Генерация базового ответа
        result = self.generate_response(query)
        
        if not result["context_used"]:
            return result["answer"]
        
        # Форматирование финального ответа
        response_parts = []
        
        # Добавление ответа
        response_parts.append(result["answer"])
        
        # Добавление источников
        if result["sources"]:
            response_parts.append("\n\n📚 **Источники:**")
            for source in result["sources"][:3]:  # Ограничиваем количество источников
                response_parts.append(f"• {source}")
        
        # Добавление метаданных о поиске
        if len(result["context_chunks"]) > 0:
            best_match = result["context_chunks"][0]
            response_parts.append(f"\n\n🔍 Найдено {len(result['context_chunks'])} релевантных фрагментов. "
                                f"Лучшее совпадение: {best_match['similarity']:.2f}")
        
        return "\n".join(response_parts)