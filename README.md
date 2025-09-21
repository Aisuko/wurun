# Wurun

Async OpenAI API wrapper optimized for Jupyter notebooks with connection pooling, retry logic, and batch processing.

## Features

- **HTTP/2 Connection Pooling** - Shared client for efficient API calls
- **Robust Retry Logic** - Exponential backoff for rate limits and errors
- **Batch Processing** - Concurrent API calls with semaphore control
- **Jupyter Optimized** - Clean notebook output and error handling
- **Zero Configuration** - Simple setup, works with OpenAI and Azure OpenAI

## Installation

```bash
# Production use
pip install .

# Development (includes testing tools)
pip install .[dev]
```

## Quick Start

```python
from wurun import Wurun

# Setup once per kernel
await Wurun.setup(
    endpoint="https://api.openai.com/v1",
    api_key="your-api-key",
    deployment_name="gpt-3.5-turbo"
)

# Single question
messages = [{"role": "user", "content": "Explain asyncio"}]
answer = await Wurun.ask(messages)
print(answer)

# Batch processing
questions = [
    [{"role": "user", "content": "What is Python?"}],
    [{"role": "user", "content": "What is JavaScript?"}]
]
answers = await Wurun.run_gather(questions, concurrency=2)

# Cleanup
await Wurun.close()
```

## API Reference

### Setup
- `Wurun.setup()` - Initialize client (call once per kernel)
- `Wurun.close()` - Clean up resources

### Single Calls
- `Wurun.ask()` - Single API call with retry logic
- `return_meta=True` - Include latency and retry count

### Batch Processing
- `Wurun.run_gather()` - Preserve input order
- `Wurun.run_as_completed()` - Process as results finish
- `concurrency` parameter controls parallel requests

### Notebook Helpers
- `Wurun.print_qna_ordered()` - Pretty print Q&A format
- `Wurun.print_as_ready()` - Print results as they complete

## Configuration

```python
await Wurun.setup(
    endpoint="https://api.openai.com/v1",
    api_key="your-key",
    deployment_name="gpt-3.5-turbo",
    timeout=30.0,
    max_connections=32,
    max_keepalive=16,
    http2=True,
    max_retries=2
)
```

## Error Handling

```python
# Custom retry settings
result = await Wurun.ask(
    messages,
    attempts=3,
    initial_backoff=1.0,
    max_backoff=10.0
)

# Get metadata
answer, meta = await Wurun.ask(messages, return_meta=True)
print(f"Latency: {meta['latency']:.2f}s, Retries: {meta['retries']}")
```

## Development

```bash
# Install dev dependencies
pip install .[dev]

# Run tests
pytest test_wurun.py -v
```

## Release Process

1. **Create PR**: Make changes and create pull request to `main`
2. **Auto Draft**: Release Drafter automatically creates/updates draft release
3. **Publish Release**: Go to GitHub Releases, edit draft, and publish
4. **Auto Deploy**: Publishing triggers automatic PyPI deployment

### Manual Version Update
```bash
# Update version in pyproject.toml
python scripts/update_version.py 1.2.3
```

## License

MIT