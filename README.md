# optimization-course

Монорепозиторий для курса **«Методы оптимизации»**.

Проект разделён на переиспользуемую библиотеку `optimlib` и отдельные директории
лабораторных работ. Лабораторные импортируют ядро библиотеки и содержат только
свои функции, конфиги и скрипты запуска.

## Структура

```text
.
+-- optimlib/          # библиотека: ядро, оптимизаторы, эксперименты, графики
+-- lab1/              # лабораторная 1: одномерный поиск
+-- lab2/              # лабораторная 2: классические градиентные методы
+-- lab3/              # лабораторная 3: momentum и адаптивные методы
+-- legacy/            # старая версия первой лабораторной для сверки
+-- README.md
```

## Установка

Из корня репозитория установите библиотеку в режиме разработки:

```powershell
.\.venv\Scripts\pip.exe install -e .\optimlib[dev]
```

Если виртуальное окружение ещё не создано:

```powershell
py -m venv .venv
.\.venv\Scripts\pip.exe install -e .\optimlib[dev]
```

## Запуск лабораторных

```powershell
.\.venv\Scripts\python.exe lab1\run.py
.\.venv\Scripts\python.exe lab2\run.py
.\.venv\Scripts\python.exe lab3\run.py
```

Результаты сохраняются в `outputs/lab*/`: сводная таблица `summary.csv` и
графики в подкаталоге `plots/` (PNG).

## Проверки

```powershell
cd optimlib
..\.venv\Scripts\python.exe -m pytest
..\.venv\Scripts\python.exe -m mypy src ..\lab1 ..\lab2 ..\lab3
```
