import imp
import os
import time

import torch
import torch.distributed
import util.misc as misc
import wandb
from torch.distributed import all_reduce
from torch.nn.functional import adaptive_avg_pool2d
from tqdm.cli import tqdm
from util.dist_utils import gather_from_all


# utils
@torch.no_grad()
def concat_all_gather(tensor):
    """
    Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    tensors_gather = [
        torch.ones_like(tensor) for _ in range(torch.distributed.get_world_size())
    ]
    torch.distributed.all_gather(tensors_gather, tensor, async_op=False)

    output = torch.cat(tensors_gather, dim=1)
    return output


def get_knn_iter(x, gpu):
    if gpu == 0:
        return tqdm(x)
    else:
        return x


@torch.no_grad()
def kNN(
    cmd_args=None,
    net=None,
    trainloader=None,
    testloader=None,
    sigma=0.07,
    feat_dim=768,
    eval_scale=256,
    eval_base_resolution=1.0,
    gsd_embed=False,
):
    is_dist = misc.is_dist_avail_and_initialized()
    net.eval()
    print(f"Starting KNN evaluation with K={cmd_args.knn}")
    gsd_ratio = eval_base_resolution
    if gsd_embed:
        gsd_ratio = gsd_ratio * (224 / eval_scale)

    st_time = time.time()
    trainFeatures = torch.zeros(
        [feat_dim + 1, len(trainloader) * trainloader.batch_size]
    )
    if not hasattr(cmd_args, "gpu"):
        cmd_args.gpu = None

    if cmd_args.gpu is not None:
        trainFeatures = trainFeatures.cuda(cmd_args.gpu)
    else:
        trainFeatures = trainFeatures.cuda()

    for batch_idx, (inputs, targets) in get_knn_iter(
        enumerate(trainloader), cmd_args.gpu
    ):
        # print mean and std as a debugging sanity check
        if batch_idx == 0:
            print("Eval data mean (should be near 0):", inputs.mean())
            print("Eval data std (should be near 1):", inputs.std())

        # targets = targets.cuda(async=True)
        batchSize = inputs.size(0)
        if cmd_args.gpu is not None:
            inputs = inputs.cuda(cmd_args.gpu)
        else:
            inputs = inputs.cuda()
        inputs = torch.nn.functional.interpolate(
            inputs, (eval_scale, eval_scale), mode="area"
        )
        features = net(
            inputs,
            input_res=torch.ones(len(inputs)).float().to(inputs.device) * gsd_ratio,
            knn_feats=True,
        )
        # breakpoint()
        trainFeatures[
            :-1, batch_idx * batchSize : batch_idx * batchSize + batchSize
        ] = features.T
        trainFeatures[
            -1, batch_idx * batchSize : batch_idx * batchSize + batchSize
        ] = targets

    if is_dist:
        print(f"distributed world size: {torch.distributed.get_world_size()}")
        trainFeatures = gather_from_all(
            trainFeatures.permute(1, 0).contiguous()
        ).permute(1, 0)

    if not hasattr(cmd_args, "gpu") or cmd_args.gpu is None:
        trainLabels = torch.flatten(trainFeatures[-1, :]).cuda()
        trainFeatures = trainFeatures[:-1, :].cuda()
    else:
        trainLabels = torch.flatten(trainFeatures[-1, :]).cuda(cmd_args.gpu)
        trainFeatures = trainFeatures[:-1, :].cuda(cmd_args.gpu)

    trainFeatures = torch.nn.functional.normalize(trainFeatures, dim=0)

    print(
        f"Grabbing all kNN training features took {(time.time() - st_time): .1f} seconds"
    )
    print(f"Shape of final train features {trainFeatures.shape}")
    top1 = torch.FloatTensor([0.0])
    total = torch.FloatTensor([0.0])
    if cmd_args.gpu is not None:
        top1 = top1.cuda(cmd_args.gpu)
        total = total.cuda(cmd_args.gpu)
    else:
        top1 = top1.cuda()
        total = total.cuda()
    C = int(trainLabels.max() + 1)
    st_time = time.time()
    with torch.no_grad():
        retrieval_one_hot = torch.zeros(cmd_args.knn, C).cuda()
        for batch_idx, (inputs, targets) in get_knn_iter(
            enumerate(testloader), cmd_args.gpu
        ):

            # targets = targets.cuda(async=True)
            batchSize = inputs.size(0)
            if cmd_args.gpu is not None:
                inputs = inputs.cuda(cmd_args.gpu)
                targets = targets.cuda(cmd_args.gpu)
            else:
                inputs = inputs.cuda()
                targets = targets.cuda()
            inputs = torch.nn.functional.interpolate(
                inputs, (eval_scale, eval_scale), mode="area"
            )

            features = net(
                inputs,
                input_res=torch.ones(len(inputs)).float().to(inputs.device) * gsd_ratio,
                knn_feats=True,
            )
            features = torch.nn.functional.normalize(features, dim=1)
            dist = torch.mm(features, trainFeatures)
            # if misc.is_main_process():
            #     breakpoint()
            yd, yi = dist.topk(cmd_args.knn, dim=1, largest=True, sorted=True)
            candidates = trainLabels.view(1, -1).expand(batchSize, -1)
            retrieval = torch.gather(candidates, 1, yi).long()

            retrieval_one_hot.resize_(batchSize * cmd_args.knn, C).zero_()
            retrieval_one_hot.scatter_(1, retrieval.view(-1, 1), 1)
            yd_transform = yd.clone().div_(sigma).exp_()
            probs = torch.sum(
                torch.mul(
                    retrieval_one_hot.view(batchSize, -1, C),
                    yd_transform.view(batchSize, -1, 1),
                ),
                1,
            )
            _, predictions = probs.sort(1, True)

            # Find which predictions match the target
            correct = predictions.eq(targets.data.view(-1, 1))

            top1 = top1 + correct.narrow(1, 0, 1).sum().item()

            total += targets.size(0)

    if is_dist:
        all_reduce(top1)
        all_reduce(total)
    top1 = top1.detach().cpu().numpy().item()  # sum
    total = total.detach().cpu().numpy().item()  # sum

    return top1 / total
