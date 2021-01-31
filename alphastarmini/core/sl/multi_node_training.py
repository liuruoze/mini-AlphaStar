import torch.distributed as dist

parser.add_argument("--local_rank", type=int, default=0) 


def synchronize():
    """
    Helper function to synchronize (barrier) among all processes when
    using distributed training
    """
    if not dist.is_available():
        return
    if not dist.is_initialized():
        return
    world_size = dist.get_world_size()
    if world_size == 1:
        return
    dist.barrier()


num_gpus = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1

is_distributed = num_gpus > 1

if is_distributed:
    torch.cuda.set_device(args.local_rank)  
    torch.distributed.init_process_group(
        backend="nccl", init_method="env://"
    )
    synchronize()


if is_distributed:
    model = torch.nn.parallel.DistributedDataParallel(
        model, device_ids=[args.local_rank], output_device=args.local_rank,
        # this should be removed if we update BatchNorm stats
        broadcast_buffers=True,
    )
