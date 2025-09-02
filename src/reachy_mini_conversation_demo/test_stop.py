import asyncio
from reachy_mini import ReachyMini

async def test_loop():
    while True:
        print("doing")
        await asyncio.sleep(1)


async def main():
    current_robot = ReachyMini()

    tasks = [
        asyncio.create_task(test_loop(), name="test")
    ]

    try:
        await asyncio.gather(*tasks, return_exceptions=True)
    except asyncio.CancelledError:
        print("got stop")

    print("tasks")
    tasks = asyncio.all_tasks()
    for t in tasks:
        print(t)
    
    # IS REQUIRED TO EXIT THE THREAD
    current_robot.client.disconnect()
    print("done")
    # os._exit(0)

if __name__ == "__main__":
    asyncio.run(main())
