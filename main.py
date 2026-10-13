"""FastAPI routes for the demo API. Start with: uvicorn main:app --reload"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from functions import (
    get_orders,
    get_payments,
    get_product,
    get_user,
    update_order_price,
)

app = FastAPI(title="Demo API")


class PriceUpdate(BaseModel):
    price: float = Field(gt=0)


@app.get("/")
async def read_root():
    return {"message": "Demo API is running"}


@app.get("/user/{user_id}")
async def read_user(user_id: int):
    return await get_user(user_id)


@app.get("/user/{user_id}/payments")
async def read_payments(user_id: int):
    return await get_payments(user_id)


@app.get("/user/{user_id}/orders")
async def read_orders(user_id: int):
    return await get_orders(user_id)


@app.put("/orders/{order_id}/price")
async def change_order_price(order_id: int, update: PriceUpdate):
    try:
        order = await update_order_price(order_id, update.price)
    except LookupError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return {"message": "Price updated", "order": order}


@app.get("/product/{product_id}")
async def read_product(product_id: int):
    return await get_product(product_id)
