import asyncio

async def fetch_data(id, timeout):
  print(f"Executing fetch_data({id})")

  # asyncio.sleep(5) will yield a Future object,
  # which will float up to the Task that starts 
  # this coroutine chain.
  # Python will pause here
  print(f"Pausing fetch_data({id})\n")
  await asyncio.sleep(timeout)

  # Python will resume here
  print(f"Resuming fetch_data({id})")
  return f"Data {id}"

async def main():
  # `asyncio.create_task()` creates a `asyncio.Task` object and then put it into the event loop's ready queue immediately.
  # Meaning: the created task is ready to be executed by the event loop
  task1 = asyncio.create_task(fetch_data(id=1, timeout=30))
  print("Created task1\n")

  print("await fetch_data(2)")

  # Since fetch_data() yields a Future object,
  # Python will pause here and pass the Future object up
  # to the corresponding Task that started this coroutine.
  # 
  # Since the event loop's ready queue contains task1, the event loop
  # will execute task1
  # 
  # Later, the event loop will resume this coroutine here.
  # Therefore, fetch_data() will be resumed and finally it returns "Data 2",
  # which will be assigned to res2
  res2 = await fetch_data(id=2, timeout=3)
  print(f"res2 = {res2}\n")

  print("Waiting task1")

  # If task1 is not done yet, Python will pause here and yield control back to 
  # the event loop.
  # 
  # Later, the event loop will resume this coroutine here.
  res1 = await task1
  print(f"res1 = {res1}")


# NOTE: main() is a coroutine
# Implicitly, asyncio.run() will create a corresponding asyncio.Task for main()
# and put it into the event loop's ready queue immediately.
# So the event loop will execute the main task immediately.
asyncio.run(main())
