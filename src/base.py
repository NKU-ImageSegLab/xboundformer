import argparse

import yaml

from utils.name_config import NameConfig


def get_cfg():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='./config/isic2016.yaml')
    parser.add_argument('--dataset_path', type=str)
    parser.add_argument('--arch', type=str)
    parser.add_argument('--gpu', type=str)
    parser.add_argument('--net_layer', type=int)
    parser.add_argument('--dataset', type=str)
    parser.add_argument('--exp_name', type=str)
    parser.add_argument('--fold', type=str)
    parser.add_argument('--lr_seg', type=float)  # 0.0003
    parser.add_argument('--n_epochs', type=int)  # 100
    parser.add_argument('--bt_size', type=int)  # 36
    parser.add_argument('--seg_loss', type=int)
    parser.add_argument('--aug', type=int)
    parser.add_argument('--patience', type=int)  # 50

    # transformer
    parser.add_argument('--filter', type=int)
    parser.add_argument('--im_num', type=int)
    parser.add_argument('--ex_num', type=int)
    parser.add_argument('--xbound', type=int)
    parser.add_argument('--point_w', type=float)

    # log_dir name
    parser.add_argument('--folder_name', type=str)

    parse_config = parser.parse_args()
    with open(parse_config.config, "r") as yaml_file:
        # 使用PyYAML加载YAML数据
        config = yaml.safe_load(yaml_file)

    def merge_dicts(dict1, dict2):
        for key, value in dict2.items():
            if value is not None:
                dict1[key] = value
        return dict1

    args_dict = vars(parse_config)
    config = merge_dicts(config, args_dict)
    return NameConfig(**config)