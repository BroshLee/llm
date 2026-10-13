"""Async HTTP client that fetches a user's dashboard from the API."""

import asyncio

import httpx

API_BASE_URL = "http://127.0.0.1:8000"


async def fetch_data(client: httpx.AsyncClient, path: str) -> dict:
    response = await client.get(path)
    response.raise_for_status()
    return response.json()


async def get_user(client: httpx.AsyncClient, user_id: int) -> dict:
    return await fetch_data(client, f"/user/{user_id}")


async def get_payment(client: httpx.AsyncClient, user_id: int) -> dict:
    return await fetch_data(client, f"/user/{user_id}/payments")


async def get_product(client: httpx.AsyncClient, product_id: int = 1) -> dict:
    return await fetch_data(client, f"/product/{product_id}")


async def update_order_price(order_id: int, price: float) -> dict:
    async with httpx.AsyncClient(base_url=API_BASE_URL, timeout=10.0) as client:
        response = await client.put(
            f"/orders/{order_id}/price",
            json={"price": price},
        )
        response.raise_for_status()
        return response.json()


async def get_dashboard(user_id: int) -> dict:
    async with httpx.AsyncClient(base_url=API_BASE_URL, timeout=10.0) as client:
        user, payment, product = await asyncio.gather(
            get_user(client, user_id),
            get_payment(client, user_id),
            get_product(client),
        )
    return {"user": user, "payment": payment, "product": product}


async def main() -> None:
    try:
        order_id = int(input("Enter the order ID (try 7): "))
        price = float(input("Enter the new price: "))
        result = await update_order_price(order_id, price)
        print(result)
    except ValueError:
        print("Enter a valid order ID and numeric price.")
    except httpx.HTTPError as exc:
        print(f"Could not fetch the dashboard: {exc}")


if __name__ == "__main__":
    asyncio.run(main())
