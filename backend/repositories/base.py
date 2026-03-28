from typing import Any, Optional


class BaseRepository:
    def __init__(self, db: Any, collection_name: str) -> None:
        self._db = db
        self._collection = getattr(db, collection_name)

    async def find_one(
        self,
        filter_dict: dict,
        sort: Optional[list[tuple[str, int]]] = None,
    ) -> Optional[dict]:
        return await self._collection.find_one(
            filter_dict, {"_id": 0}, sort=sort
        )

    async def find_many(
        self,
        filter_dict: dict,
        limit: int = 100,
        sort: Optional[list[tuple[str, int]]] = None,
    ) -> list[dict]:
        cursor = self._collection.find(filter_dict, {"_id": 0})
        if sort:
            cursor = cursor.sort(sort)
        return await cursor.to_list(limit)

    async def insert_one(self, document: dict) -> Any:
        return await self._collection.insert_one(document)

    async def update_one(
        self,
        filter_dict: dict,
        update_dict: dict,
        upsert: bool = False,
    ) -> Any:
        return await self._collection.update_one(
            filter_dict, update_dict, upsert=upsert
        )

    async def delete_one(self, filter_dict: dict) -> Any:
        return await self._collection.delete_one(filter_dict)

    async def delete_many(self, filter_dict: dict) -> Any:
        return await self._collection.delete_many(filter_dict)

    async def count(self, filter_dict: Optional[dict] = None) -> int:
        return await self._collection.count_documents(filter_dict or {})
