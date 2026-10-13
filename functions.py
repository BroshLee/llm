"""Demo data and business functions used by the API routes."""

import asyncio

# In-memory demo data. Changes remain until the API process restarts.
ORDERS_BY_USER = {
    1: [
        {"id": 1, "item": "Laptop", "price": 1000},
        {"id": 2, "item": "Phone", "price": 500},
    ],
    2: [
        {"id": 3, "item": "Tablet", "price": 300},
        {"id": 4, "item": "Headphones", "price": 100},
    ],
}
DEFAULT_ORDERS = [
    {"id": 5, "item": "Monitor", "price": 200},
    {"id": 6, "item": "Keyboard", "price": 50},
    {"id": 7, "item": "Mouse", "price": 30},
]


async def get_user(user_id: int) -> dict:
    await asyncio.sleep(0.2)
    names = {
        1: ("Alice", "alice@example.com"),
        2: ("Bob", "bob@example.com"),
    }
    name, email = names.get(user_id, ("John Doe", "john@example.com"))
    return {"id": user_id, "name": name, "email": email}


async def get_orders(user_id: int) -> dict:
    await asyncio.sleep(0.2)
    orders = ORDERS_BY_USER.get(user_id, DEFAULT_ORDERS)
    return {"user_id": user_id, "orders": orders}


async def update_order_price(order_id: int, price: float) -> dict:
    for orders in (*ORDERS_BY_USER.values(), DEFAULT_ORDERS):
        for order in orders:
            if order["id"] == order_id:
                order["price"] = price
                return order
    raise LookupError(f"Order {order_id} was not found")


async def get_payments(user_id: int) -> dict:
    await asyncio.sleep(0.2)
    orders_result = await get_orders(user_id)
    total = sum(order["price"] for order in orders_result["orders"])
    status = "success" if user_id in (1, 2) else "pending"
    return {
        "user_id": user_id,
        "payments": [{"id": user_id, "amount": total, "status": status}],
    }


async def get_product(product_id: int) -> dict:
    await asyncio.sleep(0.2)
    return {"id": product_id, "name": "Product A", "price": 100}

# import asyncio
# import httpx

# async def fetch_post():
#     async with httpx.AsyncClient() as client:
#         response = await client.get(
#             "https://jsonplaceholder.typicode.com/posts/1"
#         )
#         return response.json()

# async def main():
#     post = await fetch_post()
#     print(f"Post: {post}")

# asyncio.run(main())

# import asyncio

# async def task1():
#     print("Task1 is started")
#     await asyncio.sleep(3)
#     print("Task1 is Completed")

# async def main():
#     task = asyncio.create_task(task1())
#     print("Main function is running")

#     await task

#     print("Main function is completed")

# asyncio.run(main())



import asyncio
import httpx

async def fetch(url):
    async with httpx.AsyncClient() as client:
        response = await client.get(url)
        return response.json()

async def main():
    task1 = asyncio.create_task(
        fetch("https://jsonplaceholder.typicode.com/posts/1")
    )

    task2 = asyncio.create_task(
        fetch("https://jsonplaceholder.typicode.com/users/1")
    )

    post = await task1
    user = await task2

    print(post)
    print(user)

# Guarded so importing this module from main.py doesn't fire network calls.
if __name__ == "__main__":
    asyncio.run(main())
