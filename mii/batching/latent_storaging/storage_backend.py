import lmdb
import torch
from safetensors.torch import load, save
from typing import Union
import hashlib
import redis.asyncio as aioredis
import asyncio
import redis

class StorageBackend:
    def __init__(self, max_storage_size, path, chunk_size):
        self.chunk_size = chunk_size
        # self.env = lmdb.open(path, map_size=max_storage_size)
        # self.redis = redis.Redis(host="localhost", port=6380, db=0)
        self.r_sync = redis.Redis(host='localhost', port=6381, db=0)

    def initialize_async(self, redis):
        # self.redis = redis.Redis(host="localhost", port=6380, db=0)
        self.r_async = redis

    # def close_async(self):
    #     self.loop.close()
        
    def _serialize(self, latent: torch.Tensor):
        return save({"latent_bytes": latent.contiguous()})

    def _deserialize(self, b: Union[bytearray, bytes]):
        return load(bytes(b))["latent_bytes"]

    def _hashing(self, input_token_chunk: torch.Tensor):
        # assert input_token_chunk.shape[0]
        return hashlib.sha256(input_token_chunk.cpu().numpy().tobytes()).hexdigest()
    
    def put(self, key: torch.Tensor, value: torch.Tensor):
        hashed_key = self._hashing(key)
        serialized_value = self._serialize(value)
        # with self.env.begin(write=True) as txn:
        #     txn.put(hashed_key.encode(), serialized_value)
        self.redis.set(hashed_key, serialized_value)

    def get(self, key: torch.Tensor):
        hashed_key = self._hashing(key)
        # with self.env.begin() as txn:
        #     value = txn.get(hashed_key.encode())
        #     if value:
        #         return self._deserialize(value)
        #     else:
        #         return None
        data = self.r_sync.get(hashed_key)
        return self._deserialize(data) if data else None
            

    async def batch_put(self, keys: list[torch.Tensor], values: list[torch.Tensor]):
        # with self.env.begin(write=True) as txn:
        async with self.r_async.pipeline(transaction=False) as pipe:
            for key, value in zip(keys, values):
                hashed_key = self._hashing(key)
                serialized_value = self._serialize(value)
                pipe.set(hashed_key, serialized_value)
            await pipe.execute()