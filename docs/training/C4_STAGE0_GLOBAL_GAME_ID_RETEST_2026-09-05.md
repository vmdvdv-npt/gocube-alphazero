# C4 Stage 0: global Game ID retest

- Исправлено: registry Game ID вынесен из `data/<run-name>/` в общий persistent `data/.gocube-game-ids`; счётчик `C4` защищён межпроцессной блокировкой.
- Тесты: `.venv/bin/python -m pytest -q` — 247 passed.
- Первый диагностический run: `C4-000001`, `C4-000002`, `C4-000003`.
- Второй диагностический run: `C4-000004`, `C4-000005`, `C4-000006`.

STAGE 0: PASS
