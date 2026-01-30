```mermaid
sequenceDiagram
title Обработка запроса в RAG-боте с точками отказа и оценкой качества

    actor User as Пользователь (Telegram)
    participant TB as Telegram Bot
    participant QL as Query Logger
    participant RB as RAG Bot
    participant VI as Vector Index (FAISS)
    participant OLL as Ollama Server
    participant EH as Error Handler

    User->>TB: Отправляет запрос "/search нейронные сети"
    activate TB
    
    %% === ЭТАП 1: ЛОГИРОВАНИЕ НАЧАЛА ЗАПРОСА ===
    TB->>QL: log_query_start(query, user_id)
    activate QL
    QL-->>TB: Запись в журнал
    deactivate QL
    
    %% === ЭТАП 2: ПОИСК В ВЕКТОРНОМ ИНДЕКСЕ ===
    TB->>RB: enhanced_generate(query)
    activate RB
    
    alt ✅ Индекс загружен и валиден
        RB->>VI: search_similar_chunks(query, k=5)
        activate VI
        
        alt ✅ Найдены релевантные чанки (similarity > 0.3)
            VI-->>RB: [chunk1, chunk2, chunk3]
            deactivate VI
            
            %% === ЭТАП 3: ФОРМИРОВАНИЕ ПРОМПТА ===
            RB->>RB: create_rag_prompt(query, chunks)
            
            alt ✅ Чанки валидны (метаданные корректны)
                RB->>OLL: llm.invoke(prompt)
                activate OLL
                
                %% === ЭТАП 4: ГЕНЕРАЦИЯ ОТВЕТА ===
                alt ✅ Успешная генерация (статус 200)
                    OLL-->>RB: response (текст ответа)
                    deactivate OLL
                    
                    %% === ЭТАП 5: ОЦЕНКА КАЧЕСТВА ОТВЕТА ===
                    group 🔍 Оценка качества ответа
                        RB->>RB: _is_successful_response(response)
                        Note over RB: Критерии:<br/>• Длина > 80 символов<br/>• Нет ключевых слов ошибок<br/>• Есть структура [ОТВЕТ]/[РАЗМЫШЛЕНИЕ]<br/>• Источники извлечены
                    end
                    
                    alt ✅ Ответ успешный (successful=true)
                        RB-->>TB: Форматированный ответ с источниками
                        deactivate RB
                        
                        %% === ЭТАП 6: ОТПРАВКА ПОЛЬЗОВАТЕЛЮ ===
                        TB->>User: Ответ с источниками и метриками
                        
                        %% === ЭТАП 7: ФИНАЛЬНОЕ ЛОГИРОВАНИЕ ===
                        TB->>QL: log_query_complete(успешно)
                        activate QL
                        QL-->>TB: Запись завершена
                        deactivate QL
                        
                    else ❌ Ответ неудачный (короткий/ошибка)
                        RB-->>TB: Заглушка "В базе нет информации"
                        deactivate RB
                        
                        TB->>User: Информативное сообщение об ошибке
                        TB->>QL: log_query_complete(неудачно, причина)
                    end
                    
                else ❌ Ошибка генерации (404/таймаут/соединение)
                    OLL-->>RB: HTTP 404 / ConnectionError
                    deactivate OLL
                    
                    RB->>EH: handle_generation_error(e)
                    activate EH
                    EH-->>RB: Обработанное сообщение об ошибке
                    deactivate EH
                    
                    RB-->>TB: "❌ Сервер Ollama недоступен"
                    deactivate RB
                    
                    TB->>User: Сообщение об ошибке Ollama
                    TB->>QL: log_query_complete(ошибка_генерации)
                end
                
            else ❌ Невалидный чанк (битые метаданные)
                VI-->>RB: ChunkValidationError
                deactivate VI
                
                RB->>EH: handle_chunk_error(e)
                activate EH
                EH-->>RB: Обработанная ошибка
                deactivate EH
                
                RB-->>TB: "⚠️ Ошибка обработки данных"
                deactivate RB
                
                TB->>User: Сообщение об ошибке данных
                TB->>QL: log_query_complete(ошибка_чанка)
            end
            
        else ⚠️ Чанки не найдены (similarity < 0.3)
            VI-->>RB: []
            deactivate VI
            
            RB->>RB: Генерация заглушки "нет информации"
            RB-->>TB: "В базе знаний не найдено информации..."
            deactivate RB
            
            TB->>User: Заглушка без источников
            TB->>QL: log_query_complete(без_чанков)
        end
        
    else ❌ Ошибка индекса (пустой/битый/не загружен)
        VI-->>RB: FileNotFoundError / IndexError
        deactivate VI
        
        RB->>EH: handle_index_error(e)
        activate EH
        EH-->>RB: Обработанная ошибка индекса
        deactivate EH
        
        RB-->>TB: "❌ Векторный индекс недоступен"
        deactivate RB
        
        TB->>User: Сообщение об ошибке индекса
        TB->>QL: log_query_complete(ошибка_индекса)
    end
    
    deactivate TB
    User->>User: Получает ответ с источниками и метриками
   ```