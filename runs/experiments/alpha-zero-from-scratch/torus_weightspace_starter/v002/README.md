# Torus9 WSP v002

Отдельный эксперимент с переиспользованием каналов, непрерывными полями и
аналитической программой весов для неизменённой GraphNet 8×120.

- [Архитектурный анализ и компилятор](ARCHITECTURE.md).
- [Протокол на 66 проходов и воспроизведение](PROTOCOL.md).
- [Результаты независимой оценки](RESULTS.md).
- [Финальная программа](compiler.py) и [checkpoint](candidates/harmonic-stable/checkpoint.pt).
- [Фиксация моделей и исходников](session64/freeze.json).

Все варианты создаются от нуля, без обучения/fitting/импорта готовых весов.
Стандартная модель исполняет всю игровую эвристику. Внешний referee используется
для правил, стандартных входных плоскостей, легальности, поиска и оценки партий.
v000/v001, их сессия и production-файлы сохранены.

Большие отчёты сохраняются в Git как lossless gzip-копии. Для анализа в новом
checkout распакуйте их с проверкой SHA (существующие файлы не перезаписываются):

```python
import gzip, hashlib, json
from pathlib import Path
root = Path('v002')  # запуск из torus_weightspace_starter
for item in json.loads((root/'evidence/manifest.json').read_text()):
    data = gzip.decompress((root/item['archive']).read_bytes())
    assert hashlib.sha256(data).hexdigest() == item['sha256']
    target = root/item['original']
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        assert target.read_bytes() == data
    else:
        with target.open('xb') as stream:
            stream.write(data)
```

`source-snapshots/<sha256>.py` позволяет восстановить точные исходники каждого
запроса, включая ранние неудачные варианты. Для первой редакции runner снимок
восстановлен из известных правок и сверен с уже записанным SHA; последующие
reservation автоматически сохраняют содержимое. Базовая ревизия также указана
в каждом request. Хеш состояния модели — надёжнее хеша контейнера между разными
версиями PyTorch; использованный runtime записан в metadata.
