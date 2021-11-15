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
    fh = logging.FileHandler('test_yoga_score_3item.log')
    fh.setLevel(logging.INFO)
    logger.addHandler(fh)

    # setup data_loader instances
    module_args = dict(config["data_loader"]['args'])
    module_args.update({
        "score_csv": "data_loader/score_data_raw.csv",
        "raw_anno": True,
        "batch_size": 512,
        "shuffle": False,
        "validation_split": 0.0,
        "training": False
    })
    data_loader = getattr(module_data, "Score3ItemDataLoader")(**module_args)

    # build model architecture
    model = config.init_obj('arch', module_arch)
    logger.info(model)

    cfg_trainer = config['trainer']
    use_keypoints = cfg_trainer.get('use_keypoints', False)

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

    pos_amount = 0
    class_pos_amount = 0
    total_amount = 0
    with torch.no_grad():
        for i, data_items in enumerate(tqdm(data_loader)):
            features_list = []
            predict_class = []
            for key in range(3):
                keypoints = data_items[key].to(device)
                features_list.append(model.encode(keypoints))

                output = model(keypoints)
                output = F.softmax(output, dim=1)
                predict = torch.argmax(output, dim=1)
                predict_class.append(predict)

            distance_hm = torch.diagonal(torch.mm(features_list[0], features_list[1].T), 0)
            distance_ml = torch.diagonal(torch.mm(features_list[1], features_list[2].T), 0)
            distance_hl = torch.diagonal(torch.mm(features_list[0], features_list[2].T), 0)

            pos = torch.sum(torch.where((distance_hl < distance_hm) & (distance_hl < distance_ml), 1, 0))
            pos_amount += pos

            pos = torch.sum(
                torch.where((predict_class[0] == predict_class[1]) & (predict_class[1] == predict_class[2]), 1, 0))
            class_pos_amount += pos
            total_amount += distance_hl.shape[0]

    print(pos_amount)
    print("precision: ", pos_amount * 1.0 / total_amount)

    log = {"precision: ": pos_amount * 1.0 / total_amount,
           "class precision": class_pos_amount * 1.0 / total_amount}

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
