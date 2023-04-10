import torch
import torch.distributed as dist
from torch.distributed._tensor import DeviceMesh, Shard, distribute_tensor, DTensor
import torch.multiprocessing as mp
import os


def test_mesh(rank, world_size, tensor, mesh, placements, tag=""):
    dt = distribute_tensor(tensor, mesh, placements)
    print(f"\n[{rank}/{world_size}]--> {dt}")


def test_mesh_from_local(rank, world_size, tensor, mesh, placements, tag=""):
    dt = DTensor.from_local(tensor, mesh, placements)
    print(f"\n[{rank}/{world_size}]--> {dt.shape} --> {dt}")


def run(rank, world_size):

    linear_mesh = DeviceMesh("cpu", list(range(world_size)))
    grid_mesh = DeviceMesh("cpu", mesh=[[0, 1, 2, 3],
                                        [4, 5, 6, 7]])
    threeD_mesh = DeviceMesh("cpu", mesh=[[[0, 1],
                                           [2, 3]],
                                          [[4, 5],
                                           [6, 7]]])

    linear_tensor = torch.tensor(range(0, 32))
    grid_tensor = torch.tensor(range(0, 32)).reshape((4, 8))

    # Examples to create DTensor from global tensor

    test_mesh(rank, world_size, grid_tensor,
              linear_mesh, [Shard(dim=0)], "linear")

    test_mesh(rank, world_size, grid_tensor,
              grid_mesh, [Shard(dim=0), Shard(dim=1)], "grid")

    # Error case: Placements have length 1 but device mesh has 2 dimensions
    # test_mesh(rank, world_size, grid_tensor, grid_mesh, [Shard(dim=0)])

    # The following has an equivalent form utilizing a linear mesh, see the example following it.
    test_mesh(rank, world_size, grid_tensor,
              grid_mesh, [Shard(dim=1), Shard(dim=1)], "grid-to-grid(dim=1, dim=1)")
    # This is equivalent to the above example, due to sharding over the same dimension.
    test_mesh(rank, world_size, grid_tensor,
              linear_mesh, [Shard(dim=1)], "grid-to-linear (dim=1)")

    # grid_tensor split into 3D mesh
    test_mesh(rank, world_size, grid_tensor,
              threeD_mesh, [Shard(dim=0), Shard(dim=0), Shard(dim=1)])

    test_mesh(rank, world_size, grid_tensor,
              threeD_mesh, [Shard(dim=0), Shard(dim=1), Shard(dim=0)])

    test_mesh(rank, world_size, grid_tensor,
              threeD_mesh, [Shard(dim=0), Shard(dim=1), Shard(dim=1)])

    # Trying to shard linear tensor over 2d mesh
    test_mesh(rank, world_size, linear_tensor,
              grid_mesh, [Shard(dim=0), Shard(dim=0)])

    # Same as above but with reshaped tensor to 1x32 (not necessary)
    test_mesh(rank, world_size, linear_tensor.reshape((1, linear_tensor.shape[0])),
              grid_mesh, [Shard(dim=0), Shard(dim=0)])

    # Error case: Same as above but with reshaped tensor to 1x32
    test_mesh(rank, world_size, linear_tensor.reshape((1, linear_tensor.shape[0])),
              grid_mesh, [Shard(dim=0), Shard(dim=0)])

    # Examples to create DTensor from local tensor
    local_tensor = torch.randn((3, 2), requires_grad=True)
    test_mesh_from_local(rank, world_size, local_tensor,
                         linear_mesh, [Shard(dim=0)])

    test_mesh_from_local(rank, world_size, local_tensor,
                         grid_mesh, [Shard(dim=0), Shard(dim=1)])

    test_mesh_from_local(rank, world_size, local_tensor,
                         grid_mesh, [Shard(dim=0), Shard(dim=0)])

    # This should be the same as the above
    test_mesh_from_local(rank, world_size, local_tensor,
                         linear_mesh, [Shard(dim=0)])

    # DTensor operations
    dt = distribute_tensor(linear_tensor, linear_mesh, [Shard(dim=0)])
    print(f"\n[{rank}/{world_size}]--> {dt}")
    sum = dt.sum()
    print(f"\n[{rank}/{world_size}]--> {sum}")


def init_process(rank, size, fn, backend='gloo'):
    """ Initialize the distributed environment. """
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '29500'
    dist.init_process_group(backend, rank=rank, world_size=size)
    run(rank, size)


if __name__ == "__main__":
    size = 8
    processes = []
    # mp.set_start_method("spawn")
    for rank in range(size):
        p = mp.Process(target=init_process, args=(rank, size, run))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()
