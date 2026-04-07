"""
Stream simulator — wraps any iterable of transactions as an async generator.
Supports configurable delay to mimic real-time ingestion.
"""

import asyncio
from typing import AsyncGenerator, Iterable, Dict, Any


async def stream_transactions(
    transactions: Iterable[Dict[str, Any]],
    delay_seconds: float = 0.05,
) -> AsyncGenerator[Dict[str, Any], None]:
    """
    Async generator that yields transactions with an artificial delay.

    Args:
        transactions:  Any iterable of transaction dicts.
        delay_seconds: Sleep between emitted events (0 = as fast as possible).

    Usage:
        async for txn in stream_transactions(my_list, delay_seconds=0.1):
            await process(txn)
    """
    for txn in transactions:
        yield txn
        if delay_seconds > 0:
            await asyncio.sleep(delay_seconds)


async def stream_from_generator(
    gen_func,
    delay_seconds: float = 0.05,
    **kwargs,
) -> AsyncGenerator[Dict[str, Any], None]:
    """
    Convenience wrapper: calls gen_func(**kwargs) and streams results.

    Example:
        async for txn in stream_from_generator(generate_mixed_fraud, delay_seconds=0.01):
            ...
    """
    transactions = gen_func(**kwargs)
    async for txn in stream_transactions(transactions, delay_seconds):
        yield txn
