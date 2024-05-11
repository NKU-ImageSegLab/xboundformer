import os
import sys
from os.path import join

import torch
import os
import sys

import torch
import imageio
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from lib.metrics import get_binary_metrics, MetricsResult
from base import get_cfg


def test(model, loader, config):
    exp_name = config.dataset

    base_path = join('logs', exp_name).__str__()
    if parse_config.trained_model_path is not None:
        base_path = join(parse_config.trained_model_path, exp_name).__str__()

    result_path = base_path

    if parse_config.result_path is not None:
        result_path = join(parse_config.result_path, exp_name).__str__()

    model_path = join(base_path, 'model')
    image_path = join(result_path, 'image')
    os.makedirs(base_path, exist_ok=True)
    os.makedirs(model_path, exist_ok=True)
    os.makedirs(image_path, exist_ok=True)
    best_path = join(model_path, 'best.pkl')
    metrics = get_binary_metrics()
    best_weight = torch.load(best_path, map_location=torch.device('cpu'))
    model.load_state_dict(best_weight)
    model.eval()
    for batch_idx, batch_data in tqdm(
            iterable=enumerate(loader),
            desc=f"{config.dataset} Test",
            total=len(loader)
    ):
        data = batch_data['image'].cuda().float()
        label = batch_data['label'].cuda().float()
        origin_image_names = batch_data['origin_image_name']
        with torch.no_grad():
            if config.arch == 'transfuse':
                _, _, output = model(data)
            elif config.arch == 'xboundformer':
                output, point_maps_pre, point_maps_pre1, point_maps_pre2 = model(
                    data)
                metrics.update(output, label.int())

            output = output.cpu() > 0.5
            output = output * 255
        total_length = len(origin_image_names)
        for i in range(total_length):
            name = origin_image_names[i] + '.png'

            predict_image = output[i][0]
            predict_image = predict_image.numpy().astype('uint8')
            imageio.imwrite(join(image_path, name), predict_image)
    result = MetricsResult(metrics.compute())
    params, flops = result.cal_params_flops(model, 256)
    result.to_result_csv(
        os.path.join(image_path, 'result.csv'),
        'xboundformer',
        flops=flops,
        params=params
    )


if __name__ == '__main__':
    # -------------------------- get args --------------------------#
    parse_config = get_cfg()

    EPOCHS = parse_config.n_epochs
    os.environ['CUDA_VISIBLE_DEVICES'] = parse_config.gpu
    device_ids = range(torch.cuda.device_count())

    # -------------------------- build dataloaders --------------------------#
    if 'isic' in parse_config.dataset:
        from public.isic_dataset import ISICDataset

        dataset = ISICDataset(
            parse_config,
            train=True,
            aug=parse_config.aug
        )
        dataset2 = ISICDataset(
            parse_config, train=False, aug=False
        )
    else:
        raise NotImplementedError
    val_loader = DataLoader(
        dataset2,
        batch_size=1,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
        drop_last=False
    )

    if parse_config.arch == 'xboundformer':
        from lib.xboundformer import _segm_pvtv2

        model = _segm_pvtv2(1, parse_config.im_num, parse_config.ex_num,
                            parse_config.xbound, parse_config.image_size).cuda()
    elif parse_config.arch == 'transfuse':
        from lib.TransFuse.TransFuse import TransFuse_S

        model = TransFuse_S(pretrained=True).cuda()
    else:
        exit(1)
    test(model, val_loader, parse_config)

