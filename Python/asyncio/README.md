# Python Language Features
The following features have been part of the language since Python 3.5 (PEP 492).
- `async def`: define a coroutine function. Calling the function creates a coroutine object and runs none of its lines.
- `await x`: use the coroutine directly, or call `type(x).__await__(x)` once. You get a `TypeError` for things that can't be awaited. Whatever `x` yields is passed up unchanged, and the awaiting function pauses on its `await` line.

# `asyncio` Core Components
- **Event Loop**
- **Ready Queue**
- **Scheduled Queue**
- **Task**
