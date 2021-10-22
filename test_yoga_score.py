import argparse

import numpy as np
import torch, sys
import torch.nn.functional as F
from tqdm import tqdm
import data_loader.data_loaders as module_data
import model.loss as module_loss
import model.metric as module_metric
import model.model as module_arch
from parse_config import ConfigParser
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
import logging


def main(config):
    logger = config.get_logger('test')
    fh = logging.FileHandler('test_yoga_score.log')
    fh.setLevel(logging.INFO)
    logger.addHandler(fh)

    # setup data_loader instances
    module_args = dict(config["data_loader"]['args'])
    module_args.update({
        "shuffle": False,
        "validation_split": 0.0,
        "training": False
    })
    data_loader = getattr(module_data, config['data_loader']['type'])(**module_args)

    # data_loader = getattr(module_data, config['data_loader']['type'])(
    #     config['data_loader']['args']['data_dir'],
    #     batch_size=512,
    #     shuffle=False,
    #     validation_split=0.0,
    #     training=False,
    #     num_workers=2
    # )

    # build model architecture
    model = config.init_obj('arch', module_arch)
    logger.info(model)

    # get function handles of loss and metrics
    loss_fn = getattr(module_loss, config['loss'])
    metric_fns = [getattr(module_metric, met) for met in config['metrics']]

    logger.info('Loading checkpoint: {} ...'.format(config.resume))
    checkpoint = torch.load(config.resume)
    state_dict = checkpoint['state_dict']
    if config['n_gpu'] > 1:
        model = torch.nn.DataParallel(model)
    model.load_state_dict(state_dict)

    # prepare model for testing
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()

    total_loss = 0.0
    total_metrics = torch.zeros(len(metric_fns))

    pos_amount, neg_amount = 0, 0
    with torch.no_grad():
        for i, data_items in enumerate(tqdm(data_loader)):
            categories = ["base", "same", "diff"]
            encoded_features = {}
            for cat in categories:
                items = data_items[cat]

                data = items["skeleton"].to(device)

                target = items["class"].to(device)

                features = model.encode(data)
                encoded_features[cat] = features
            # calculate the cosine between base and same:
            distance_bs = torch.mm(encoded_features["base"], encoded_features["same"].T)

            # calculate the cosine between base and diff:
            distance_bd = torch.mm(encoded_features["base"], encoded_features["diff"].T)

            distance_diff = distance_bs - distance_bd
            # if distance_diff < 0: correct
            pos = torch.sum(torch.where(distance_diff < 0, 1, 0))
            neg = torch.sum(torch.where(distance_diff > 0, 1, 0))
            pos_amount += pos
            neg_amount += neg
    print(pos_amount, neg_amount)
    print("precision: ", pos_amount * 1.0 / (pos_amount + neg_amount))

    log = {"precision: ": pos_amount * 1.0 / (pos_amount + neg_amount)}

    np.set_printoptions(threshold=sys.maxsize)
    logger.info(log)


if __name__ == '__main__':
    args = argparse.ArgumentParser(description='PyTorch Template')
    args.add_argument('-c', '--config', default=None, type=str,
                      help='config file path (default: None)')
    args.add_argument('-r', '--resume', default=None, type=str,
                      help='path to latest checkpoint (default: None)')
    args.add_argument('-d', '--device', default=None, type=str,
                      help='indices of GPUs to enable (default: all)')

    config = ConfigParser.from_args(args)
    main(config)
