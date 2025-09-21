import asyncio
import time
from typing import Any, Dict, List, Optional, Tuple

import httpx
from openai import APIError, AsyncOpenAI, RateLimitError

try:
    import pandas as pd
except ImportError:
    pd = None


class Wurun:
    _http_client: Optional[httpx.AsyncClient] = None
    _client: Optional[AsyncOpenAI] = None
    _deployment: Optional[str] = None
    _max_retries: int = 2

    # ---------- setup / teardown ----------
    @classmethod
    async def setup(
        cls,
        *,
        endpoint: str,
        api_key: str,
        deployment_name: str,
        timeout: float = 2.0,
        max_connections: int = 32,
        max_keepalive: int = 16,
        http2: bool = True,
        max_retries: int = 2,
    ) -> None:
        """
        Initialize shared HTTP/2 pool and AsyncOpenAI client.
        Call once per kernel. Safe to call again; it reuses the pool.
        """
        if cls._http_client is None:
            cls._http_client = httpx.AsyncClient(
                http2=http2,
                timeout=timeout,
                limits=httpx.Limits(max_connections=max_connections, max_keepalive_connections=max_keepalive),
            )
        cls._client = AsyncOpenAI(
            base_url=endpoint,
            api_key=api_key,
            http_client=cls._http_client,
            max_retries=max_retries,
        )
        cls._deployment = deployment_name
        cls._max_retries = max_retries

    @classmethod
    async def close(cls) -> None:
        """Close the shared HTTP client when you’re done."""
        if cls._http_client is not None:
            await cls._http_client.aclose()
        cls._http_client = None
        cls._client = None
        cls._deployment = None

    # ---------- single call (messages are pre-built) ----------
    @classmethod
    async def _chat_once(
        cls,
        *,
        messages: List[Dict[str, Any]],
        timeout: Optional[float] = None,
    ) -> str:
        """
        Send a single chat.completions call. Expects `messages` as a ready list of
        {'role': 'system'|'user'|'assistant', 'content': '...'} dicts. No modification.
        """
        if cls._client is None or cls._deployment is None:
            raise RuntimeError("Wurun.setup(...) must be called before use.")
        resp = await cls._client.chat.completions.create(
            model=cls._deployment,
            messages=messages,  # type: ignore
            timeout=timeout,
        )
        content = resp.choices[0].message.content
        if content is None:
            raise ValueError("No content in response")
        return content

    @classmethod
    async def ask(
        cls,
        messages: List[Dict[str, Any]],
        *,
        semaphore: Optional[asyncio.Semaphore] = None,
        attempts: int = 5,
        initial_backoff: float = 0.5,
        max_backoff: float = 8.0,
        timeout: Optional[float] = None,
        return_meta: bool = False,
    ) -> str | Tuple[str, Dict[str, Any]]:
        """
        Robust call with optional concurrency guard and retries.
        `messages` must already be built. No system prompt injection.
        """

        async def _work():
            start = time.perf_counter()
            delay = initial_backoff
            for i in range(attempts):
                try:
                    ans = await cls._chat_once(messages=messages, timeout=timeout)
                    lat = time.perf_counter() - start
                    return ans, {"latency": lat, "retries": i}
                except (RateLimitError, APIError) as e:
                    if i == attempts - 1:
                        lat = time.perf_counter() - start
                        return f"[ERROR] {type(e).__name__}: {e}", {"latency": lat, "retries": i}
                    await asyncio.sleep(delay)
                    delay = min(delay * 2, max_backoff)

        if semaphore is None:
            ans, meta = await _work()
        else:
            async with semaphore:
                ans, meta = await _work()

        return (ans, meta) if return_meta else ans

    # ---------- batch: preserve order ----------
    @classmethod
    async def run_gather(
        cls,
        all_messages: List[List[Dict[str, Any]]],
        *,
        concurrency: int = 5,
        timeout: Optional[float] = None,
        return_meta: bool = False,
    ) -> List[Any]:
        """
        Run a batch and preserve input order. `all_messages` is a list of
        message lists. Returns answers aligned with input order.
        """
        sem = asyncio.Semaphore(concurrency)
        coros = [cls.ask(msgs, semaphore=sem, timeout=timeout, return_meta=return_meta) for msgs in all_messages]
        return await asyncio.gather(*coros)  # type: ignore

    # ---------- batch: as-finished (with indices) ----------
    @classmethod
    async def run_as_completed(
        cls,
        all_messages: List[List[Dict[str, Any]]],
        *,
        concurrency: int = 5,
        timeout: Optional[float] = None,
        return_meta: bool = False,
    ) -> List[Tuple[int, Any]]:
        """
        Run a batch and collect results as they finish.
        Returns a list of (index, answer_or_tuple) in completion order.
        """
        sem = asyncio.Semaphore(concurrency)

        # Simple approach: use asyncio.wait with FIRST_COMPLETED
        tasks = {
            asyncio.create_task(cls.ask(msgs, semaphore=sem, timeout=timeout, return_meta=return_meta)): idx
            for idx, msgs in enumerate(all_messages)
        }

        finished: List[Tuple[int, Any]] = []
        while tasks:
            done, pending = await asyncio.wait(tasks.keys(), return_when=asyncio.FIRST_COMPLETED)
            for task in done:
                idx = tasks.pop(task)
                res = await task
                finished.append((idx, res))
        return finished

    # ---------- convenience printers (no prompt mutation) ----------
    @classmethod
    async def print_qna_ordered(
        cls,
        all_messages: List[List[Dict[str, Any]]],
        *,
        concurrency: int = 5,
        timeout: Optional[float] = None,
    ) -> None:
        answers = await cls.run_gather(all_messages, concurrency=concurrency, timeout=timeout)
        for msgs, a in zip(all_messages, answers):
            # try to display the last user message for readability
            last_user = next((m["content"] for m in reversed(msgs) if m.get("role") == "user"), "<no user msg>")
            print(f"Q: {last_user}\nA: {a}\n")

    @classmethod
    async def print_as_ready(
        cls,
        all_messages: List[List[Dict[str, Any]]],
        *,
        concurrency: int = 5,
        timeout: Optional[float] = None,
    ) -> None:
        start = time.perf_counter()
        results = await cls.run_as_completed(all_messages, concurrency=concurrency, timeout=timeout)
        for idx, a in results:
            print(f"[{idx}] A: {a}")
        print(f"\nDone in {time.perf_counter() - start:.2f}s for {len(all_messages)} prompts")

    # ---------- dataframe helper ----------
    @classmethod
    async def run_dataframe(
        cls,
        df,
        messages_column: str,
        *,
        concurrency: int = 5,
        timeout: Optional[float] = None,
        return_meta: bool = False,
    ) -> List[Any]:
        """
        Process messages from a DataFrame column and return answers in order.

        Args:
            df: DataFrame containing messages
            messages_column: Column name containing List[Dict[str, str]] messages
            concurrency: Number of concurrent requests
            timeout: Request timeout
            return_meta: Whether to return metadata

        Returns:
            List of answers in same order as DataFrame rows
        """
        if pd is None:
            raise ImportError("pandas is required for DataFrame support. Install with: pip install pandas")

        all_messages = df[messages_column].tolist()
        results = await cls.run_gather(all_messages, concurrency=concurrency, timeout=timeout, return_meta=return_meta)
        return results
