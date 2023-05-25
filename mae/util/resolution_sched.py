from lib.scheduler import ConstantResolutionScheduler, RandomResolutionScheduler
import numpy as np
import re


def get_target_size_scheduler(args):
    if args.target_size_scheduler == "random":
        target_size_scheduler = RandomResolutionScheduler(args.target_size)
    elif args.target_size_scheduler == "constant":
        target_size_scheduler = ConstantResolutionScheduler(args.target_size)
    else:
        match = re.compile("random:([0-9])").findall(args.target_size_scheduler)
        if match:
            target_size_scheduler = RandomResolutionScheduler(
                args.target_size, int(match[0])
            )
        else:
            raise NotImplementedError

    return target_size_scheduler


def get_source_size_scheduler(args):
    raise NotImplementedError
    # if args.source_size_scheduler == "random":
    #    source_size_scheduler = RandomResolutionScheduler(args.source_size)
    # elif args.source_size_scheduler == "constant":
    #    source_size_scheduler = ConstantResolutionScheduler(args.source_size)
    # else:
    #    raise NotImplementedError

    # return source_size_scheduler
