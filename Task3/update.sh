#!/bin/bash


PROJECT_PATH="/home/user/architecture-pro-rag"
VENV_PATH="$PROJECT_PATH/.venv"
PYTHON_PATH="$VENV_PATH/bin/python"
SCRIPT_PATH="$PROJECT_PATH/vector_indexer.py"
CSV_FOLDER="$PROJECT_PATH/csv"
OUTPUT_PATH="$PROJECT_PATH/knowledge_index"
LOG_FILE="$PROJECT_PATH/update_log.log"

# Логирование начала обновления
echo "=========================================" >> "$LOG_FILE"
echo "Начало обновления базы знаний: $(date)" >> "$LOG_FILE"

# Активация виртуального окружения и запуск скрипта
cd "$PROJECT_PATH"
source "$VENV_PATH/bin/activate"
"$PYTHON_PATH" "$SCRIPT_PATH" --csv_folder "$CSV_FOLDER" --output_path "$OUTPUT_PATH" >> "$LOG_FILE" 2>&1

# Проверка результата выполнения
if [ $? -eq 0 ]; then
    echo "Обновление успешно завершено: $(date)" >> "$LOG_FILE"
else
    echo "Ошибка при обновлении: $(date)" >> "$LOG_FILE"
fi

echo "=========================================" >> "$LOG_FILE"