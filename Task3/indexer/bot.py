import os
import pandas as pd
import numpy as np
import faiss
from typing import List, Dict, Any, Tuple
from sentence_transformers import SentenceTransformer
from langchain_text_splitters  import RecursiveCharacterTextSplitter
import pickle
import json

class CSVVectorIndexer:
    def __init__(self, model_name: str = 'all-MiniLM-L6-v2'):
        """
        Инициализация индексатора
        
        Args:
            model_name: название модели для эмбеддингов
        """
        self.model = SentenceTransformer(model_name)
        self.index = None
        self.metadata = []
        self.dimension = self.model.get_sentence_embedding_dimension()
        
    def load_csv_files(self, csv_folder: str) -> List[Dict[str, Any]]:
        """
        Загрузка всех CSV файлов из папки
        
        Args:
            csv_folder: путь к папке с CSV файлами
            
        Returns:
            Список документов с метаданными
        """
        documents = []
        
        for filename in os.listdir(csv_folder):
            if filename.endswith('.csv'):
                file_path = os.path.join(csv_folder, filename)
                try:
                    df = pd.read_csv(file_path)
                    
                    # Проверяем наличие колонки 'result'
                    if 'result' not in df.columns:
                        print(f"Предупреждение: в файле {filename} нет колонки 'result'")
                        continue
                    
                    # Создаем документы для каждой строки
                    for idx, row in df.iterrows():
                        if pd.notna(row['result']):
                            doc = {
                                'text': str(row['result']),
                                'source_file': filename,
                                'row_index': idx,
                                'title': f"{filename}_row_{idx}"
                            }
                            documents.append(doc)
                            
                    print(f"Загружен файл {filename}: {len(df)} строк")
                    
                except Exception as e:
                    print(f"Ошибка при загрузке файла {filename}: {e}")
        
        print(f"Всего загружено документов: {len(documents)}")
        return documents
    
    def chunk_documents(self, documents: List[Dict[str, Any]], 
                        chunk_size: int = 1000,
                        chunk_overlap: int = 200) -> List[Dict[str, Any]]:
        """
        Разбивка документов на чанки с сохранением метаданных
        
        Args:
            documents: список документов
            chunk_size: размер чанка в символах
            chunk_overlap: перекрытие между чанками
            
        Returns:
            Список чанков с метаданными
        """
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )
        
        chunks = []
        chunk_id = 0
        
        for doc in documents:
            # Разбиваем текст на чанки
            text_chunks = text_splitter.split_text(doc['text'])
            
            for i, chunk_text in enumerate(text_chunks):
                chunk_metadata = {
                    'chunk_id': chunk_id,
                    'chunk_index': i,
                    'total_chunks': len(text_chunks),
                    'source_file': doc['source_file'],
                    'original_row': doc['row_index'],
                    'title': doc['title'],
                    'text': chunk_text,
                    'char_start': doc['text'].find(chunk_text) if chunk_text in doc['text'] else -1,
                    'char_end': -1
                }
                
                # Вычисляем конечную позицию
                if chunk_metadata['char_start'] != -1:
                    chunk_metadata['char_end'] = chunk_metadata['char_start'] + len(chunk_text)
                
                chunks.append(chunk_metadata)
                chunk_id += 1
        
        print(f"Создано {len(chunks)} чанков из {len(documents)} документов")
        return chunks
    
    def generate_embeddings(self, chunks: List[Dict[str, Any]]) -> np.ndarray:
        """
        Генерация эмбеддингов для чанков
        
        Args:
            chunks: список чанков
            
        Returns:
            Матрица эмбеддингов
        """
        texts = [chunk['text'] for chunk in chunks]
        print(f"Генерация эмбеддингов для {len(texts)} чанков...")
        
        # Генерируем эмбеддинги
        embeddings = self.model.encode(
            texts,
            show_progress_bar=True,
            normalize_embeddings=True,  # Нормализация для косинусного сходства
            batch_size=32
        )
        
        return np.array(embeddings).astype('float32')
    
    def build_index(self, chunks: List[Dict[str, Any]], embeddings: np.ndarray):
        """
        Построение FAISS индекса
        
        Args:
            chunks: список чанков с метаданными
            embeddings: матрица эмбеддингов
        """
        # Создаем индекс для косинусного сходства
        # Для этого используем IndexFlatIP (inner product) с предварительной нормализацией
        self.index = faiss.IndexFlatIP(self.dimension)
        
        # Добавляем эмбеддинги в индекс
        self.index.add(embeddings)
        
        # Сохраняем метаданные
        self.metadata = chunks
        
        print(f"Индекс построен. Размерность: {self.dimension}, Векторов: {self.index.ntotal}")
    
    def search(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        """
        Поиск по индексу
        
        Args:
            query: поисковый запрос
            k: количество возвращаемых результатов
            
        Returns:
            Список найденных чанков с метаданными и оценкой сходства
        """
        if self.index is None:
            raise ValueError("Индекс не построен. Сначала вызовите build_index()")
        
        # Генерируем эмбеддинг для запроса
        query_embedding = self.model.encode([query], normalize_embeddings=True)
        query_embedding = np.array(query_embedding).astype('float32')
        
        # Ищем k ближайших соседей
        distances, indices = self.index.search(query_embedding, k)
        
        # Формируем результаты
        results = []
        for i, (distance, idx) in enumerate(zip(distances[0], indices[0])):
            if idx < len(self.metadata):  # Проверка на валидность индекса
                result = self.metadata[idx].copy()
                result['score'] = float(distance)  # Косинусное сходство
                result['rank'] = i + 1
                results.append(result)
        
        return results
    
    def save_index(self, save_path: str):
        """
        Сохранение индекса и метаданных
        
        Args:
            save_path: путь для сохранения
        """
        if self.index is None:
            raise ValueError("Индекс не построен")
        
        # Создаем папку если не существует
        os.makedirs(save_path, exist_ok=True)
        
        # Сохраняем FAISS индекс
        faiss.write_index(self.index, os.path.join(save_path, "faiss_index.bin"))
        
        # Сохраняем метаданные
        with open(os.path.join(save_path, "metadata.pkl"), 'wb') as f:
            pickle.dump(self.metadata, f)
        
        # Сохраняем конфигурацию
        config = {
            'dimension': self.dimension,
            'model_name': 'all-MiniLM-L6-v2',
            'total_chunks': len(self.metadata)
        }
        
        with open(os.path.join(save_path, "config.json"), 'w') as f:
            json.dump(config, f, indent=2)
        
        print(f"Индекс сохранен в {save_path}")
    
    def load_index(self, load_path: str):
        """
        Загрузка индекса и метаданных
        
        Args:
            load_path: путь к сохраненному индексу
        """
        # Загружаем FAISS индекс
        self.index = faiss.read_index(os.path.join(load_path, "faiss_index.bin"))
        
        # Загружаем метаданные
        with open(os.path.join(load_path, "metadata.pkl"), 'rb') as f:
            self.metadata = pickle.load(f)
        
        print(f"Индекс загружен. Векторов: {self.index.ntotal}, Чанков: {len(self.metadata)}")




# Класс для бота с цитированием
class KnowledgeBaseBot:
    def __init__(self, index_path: str = "./vector_index"):
        """
        Инициализация бота с доступом к векторной БД
        
        Args:
            index_path: путь к сохраненному индексу
        """
        self.indexer = CSVVectorIndexer()
        self.indexer.load_index(index_path)
    
    def query_with_citation(self, query: str, k: int = 5, 
                           threshold: float = 0.5) -> List[Dict[str, Any]]:
        """
        Поиск с цитированием источников
        
        Args:
            query: поисковый запрос
            k: количество результатов
            threshold: порог сходства
            
        Returns:
            Результаты с цитатами и источниками
        """
        results = self.indexer.search(query, k=k)
        
        # Фильтрация по порогу
        filtered_results = [r for r in results if r['score'] >= threshold]
        
        # Форматирование для цитирования
        for result in filtered_results:
            result['citation'] = {
                'source': result['source_file'],
                'row': result['original_row'],
                'chunk': result['chunk_index'],
                'position': f"char {result.get('char_start', 'N/A')}-{result.get('char_end', 'N/A')}"
            }
        
        return filtered_results
    
    def get_detailed_answer(self, query: str, k: int = 3) -> str:
        """
        Получение детализированного ответа с цитатами
        
        Args:
            query: вопрос пользователя
            k: количество используемых чанков
            
        Returns:
            Форматированный ответ с цитатами
        """
        results = self.query_with_citation(query, k=k)
        
        if not results:
            return "В базе знаний не найдено информации по данному вопросу."
        
        # Формируем ответ
        answer_parts = ["На основе базы знаний я нашел следующую информацию:\n"]
        
        for i, result in enumerate(results, 1):
            answer_parts.append(f"{i}. {result['text']}")
            answer_parts.append(f"   [Источник: {result['citation']['source']}, строка {result['citation']['row']}]")
            answer_parts.append("")  # Пустая строка для разделения
        
        # Добавляем сводку
        answer_parts.append("---")
        answer_parts.append(f"Всего найдено {len(results)} релевантных фрагментов.")
        
        return "\n".join(answer_parts)


# Скрипт для запуска создания индекса
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Создание бота')
    
    parser.add_argument('--output_path', type=str, default='./vector_index',
                       help='Путь для чтения индекса (по умолчанию: ./vector_index)')
    
    args = parser.parse_args()
    bot = KnowledgeBaseBot(args.output_path)
    user_input = input()
    answer = bot.get_detailed_answer(user_input)
    print(f"ответ = '{answer}'")
    