from lib.scheduler import ConstantResolutionScheduler, RandomResolutionScheduler
import numpy as np
import re


def get_output_size_scheduler(args):
    if args.fixed_output_size_min or args.fixed_output_size_max:
        assert (
            args.fixed_output_size_min > 0
            and args.fixed_output_size_max > 0
            and args.fixed_output_size_max >= args.fixed_output_size_min
        )
        output_size_scheduler = RandomResolutionScheduler(
            target_size=np.arange(
                args.fixed_output_size_min, args.fixed_output_size_max + 1, 16
            )
        )
    else:
        output_size_scheduler = ConstantResolutionScheduler(target_size=0)

    return output_size_scheduler


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
    if args.source_size_scheduler == "random":
        source_size_scheduler = RandomResolutionScheduler(args.source_size)
    elif args.source_size_scheduler == "constant":
        source_size_scheduler = ConstantResolutionScheduler(args.source_size)
    else:
        raise NotImplementedError

    return source_size_scheduler
