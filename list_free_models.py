"""Показывает бесплатные модели OpenRouter, подходящие для извлечения полей заявки.

Запуск:
    poetry run python list_free_models.py
    poetry run python list_free_models.py --max-size 40     # только модели до ~40B параметров
    poetry run python list_free_models.py --allow-reasoning # показать и «думающие» модели

Список берётся из публичного API OpenRouter (ключ не нужен), поэтому он всегда актуален:
бесплатные модели появляются и исчезают каждые несколько недель.

Критерии:
    * бесплатная (цена запроса и ответа = 0), на выходе только текст
    * поддерживает structured outputs (строгий JSON по схеме)
    * не «думающая» по названию (reasoning, thinking, r1): такие модели тратят токены на
      рассуждения, отвечают десятки секунд, а иногда возвращают пустой ответ. Модели, которые
      умеют рассуждать (колонка «можно»), не отсеиваются, но стоят в списке ниже остальных
    * не классификатор безопасности, не эмбеддинги и не модель для картинок
"""
from __future__ import annotations

import argparse
import re
import sys
from typing import Any, Optional

import httpx

URL = "https://openrouter.ai/api/v1/models"
# Модели, которые не умеют отвечать на запросы вида «верни JSON»
BAD_NAME_PARTS = ("safety", "guard", "moderation", "embed", "rerank", "image", "riverflow", "tts", "whisper", "audio")
REASONING_NAME_PARTS = ("reasoning", "thinking", "-r1", "think")


def is_free(model: dict[str, Any]) -> bool:
    pricing = model.get("pricing") or {}
    try:
        return float(pricing.get("prompt", 1)) == 0 and float(pricing.get("completion", 1)) == 0
    except (TypeError, ValueError):
        return False


def size_b(model_id: str) -> Optional[float]:
    """Размер в миллиардах параметров по названию. Для MoE берём число активных параметров (…-a12b)."""
    name = model_id.lower()
    active = re.search(r"-a(\d+(?:\.\d+)?)b", name)
    total = re.search(r"(\d+(?:\.\d+)?)b", name)
    match = active or total
    return float(match.group(1)) if match else None


def describe(model: dict[str, Any]) -> dict[str, Any]:
    params = model.get("supported_parameters") or []
    model_id = model["id"]
    arch = model.get("architecture") or {}
    outputs = arch.get("output_modalities") or []
    name = model_id.lower()
    return {
        "id": model_id,
        "context": model.get("context_length") or 0,
        "size": size_b(model_id),
        # у бесплатных моделей строгий JSON часто заявлен как response_format
        "structured": "structured_outputs" in params or "response_format" in params,
        # «думает» по названию (reasoning, thinking, r1): это надёжный признак
        "reasoning": any(p in name for p in REASONING_NAME_PARTS),
        # умеет рассуждать (параметр reasoning): такая модель может думать по умолчанию, ставим ниже в списке
        "can_reason": "reasoning" in params,
        "text_only_output": not outputs or outputs == ["text"],
        "bad": any(p in name for p in BAD_NAME_PARTS),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--max-size", type=float, default=None, help="максимум млрд параметров (если размер угадывается по имени)")
    parser.add_argument("--allow-reasoning", action="store_true", help="не отсеивать «думающие» модели")
    args = parser.parse_args()

    try:
        response = httpx.get(URL, timeout=20)
        response.raise_for_status()
        models = response.json()["data"]
    except (httpx.HTTPError, KeyError, ValueError) as e:
        print(f"Не удалось получить список моделей: {e}")
        return 1

    free = [describe(m) for m in models if is_free(m)]
    usable = [r for r in free if r["text_only_output"] and not r["bad"]]
    structured = [r for r in usable if r["structured"]]
    rows = structured if args.allow_reasoning else [r for r in structured if not r["reasoning"]]
    if args.max_size is not None:
        rows = [r for r in rows if r["size"] is None or r["size"] <= args.max_size]

    # сводка по шагам отбора: видно, на каком этапе модели отсеиваются
    print(
        f"Всего моделей: {len(models)} | бесплатных: {len(free)} | "
        f"текст и не служебные: {len(usable)} | со строгим JSON: {len(structured)} | "
        f"после остальных фильтров: {len(rows)}\n"
    )

    # порядок: не думающие и не умеющие рассуждать, затем «можно», затем «да»;
    # внутри группы сначала небольшие (известный размер по возрастанию), потом без указанного размера
    rows.sort(key=lambda r: (r["reasoning"], r["can_reason"], r["size"] is None, r["size"] or 0, -r["context"]))

    if not rows:
        print("Подходящих бесплатных моделей не найдено. Попробуйте --allow-reasoning или уберите --max-size.")
        return 0

    def think_label(r: dict[str, Any]) -> str:
        return "да" if r["reasoning"] else ("можно" if r["can_reason"] else "нет")

    print(f"{'модель':58} {'размер':>8} {'контекст':>9}  думает")
    print("-" * 86)
    for r in rows:
        size = f"~{r['size']:g}B" if r["size"] is not None else "?"
        print(f"{r['id']:58} {size:>8} {r['context']:>9}  {think_label(r)}")

    top = [r["id"] for r in rows if not r["reasoning"]][:3]
    if top:
        print("\nДля .env (основная + 2 запасные):")
        print(f"OPENROUTER_MODEL={top[0]}")
        if len(top) > 1:
            print(f"OPENROUTER_FALLBACK_MODELS={','.join(top[1:])}")
    print(
        "\nВажно: размер определён по названию и может быть неточным, а «лёгкая» не значит «хорошая»: "
        "совсем маленькие модели (1-3B) часто путают поля. Проверьте выбор на 3-4 реальных сообщениях."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
