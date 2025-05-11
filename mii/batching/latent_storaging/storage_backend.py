import lmdb
import torch
from safetensors.torch import load, save
from typing import Union
import hashlib

class StorageBackend:
    def __init__(self, max_storage_size, path, chunk_size):
        self.chunk_size = chunk_size
        # self.env = lmdb.open(path, map_size=max_storage_size)
        self.env = {}
        
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
        self.env[hashed_key] = serialized_value

    def get(self, key: torch.Tensor):
        hashed_key = self._hashing(key)
        # with self.env.begin() as txn:
        value = self.env.get(hashed_key)
        if value:
            return self._deserialize(value)
        else:
            return None
            

    def batch_put(self, keys: list[torch.Tensor], values: list[torch.Tensor]):
        # with self.env.begin(write=True) as txn:
        for key, value in zip(keys, values):
            hashed_key = self._hashing(key)
            serialized_value = self._serialize(value)
            self.env[hashed_key] = serialized_value