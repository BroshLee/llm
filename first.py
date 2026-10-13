from typing import Literal
from fastapi import FastAPI


app = FastAPI()

SortOption = Literal["local", "remote"]

@app.get("/get_file")
async def get_products(
    sort: SortOption = "local"
):
    return {
        "sort": sort
    }
