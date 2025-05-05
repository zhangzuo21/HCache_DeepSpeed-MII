from .storage_backend import StorageBackend
import torch

class LatentStoragingEngie:
    def __init__(self, chunk_size):
        self.chunk_size = chunk_size
        self.backend = StorageBackend(160000000, '/home/zzy/storage_path', chunk_size)

    def retrive(self, seq: torch.Tensor):
        retrived_list = []
        # print(f"seq len: {seq.shape[0]}")
        for i in range(seq.shape[0] // self.chunk_size):
            chunked_seq = seq[0: (i + 1) * self.chunk_size]
            stored = self.backend.get(chunked_seq)
            if not stored == None:
                retrived_list.append(stored)
        if len(retrived_list):
            retrived = torch.concat(retrived_list, dim=1)
            return retrived
        else:
            return None
    
    def store_seq(self, seq: torch.Tensor, value: torch.Tensor, restored_offset: int):
        seq_len = seq.shape[0]
        chunk_num = seq_len // self.chunk_size
        chunk_offset = restored_offset // self.chunk_size
        chunked_seq = []
        chunked_value = []
        for i in range(chunk_num - chunk_offset):
            chunked_seq.append(seq[0: (i + chunk_offset + 1) * self.chunk_size])
            chunked_value.append(value[:, i * self.chunk_size: (i + 1) * self.chunk_size])
        if len(chunked_seq):
            self.backend.batch_put(chunked_seq, chunked_value)
