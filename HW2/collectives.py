import torch
import torch.distributed as dist


def ring_reduce_scatter(tensor):
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    chunks = [chunk.clone() for chunk in tensor.chunk(world_size)]

    right = (rank + 1) % world_size
    left = (rank - 1) % world_size

    for step in range(world_size - 1):
        send_index = (rank - 1 - step) % world_size
        receive_index = (rank - 2 - step) % world_size

        incoming = torch.empty_like(chunks[receive_index])

        request = dist.isend(chunks[send_index], dst=right)
        dist.recv(incoming, src=left)
        request.wait()

        chunks[receive_index].add_(incoming)

    return chunks[rank]


def ring_all_gather(shard):
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    chunks = [torch.empty_like(shard) for _ in range(world_size)]
    chunks[rank].copy_(shard)

    right = (rank + 1) % world_size
    left = (rank - 1) % world_size

    for step in range(world_size - 1):
        send_index = (rank - step) % world_size
        receive_index = (rank - step - 1) % world_size

        incoming = torch.empty_like(shard)

        request = dist.isend(chunks[send_index], dst=right)
        dist.recv(incoming, src=left)
        request.wait()

        chunks[receive_index].copy_(incoming)

    return torch.cat(chunks)


def ring_all_reduce(tensor):
    shard = ring_reduce_scatter(tensor)
    result = ring_all_gather(shard)
    return result
