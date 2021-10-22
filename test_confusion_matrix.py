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
    fh = logging.FileHandler('test_confusion_matrix.log')
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

    targets, predicts = [], []
    with torch.no_grad():
        for i, data_items in enumerate(tqdm(data_loader)):
            data = data_items["skeleton"].to(device)

            target = data_items["class"].to(device)

            output = model(data)

            #
            # save sample images, or do something with output here
            #
            output = F.softmax(output, dim=1)
            predict = torch.argmax(output, dim=1)
            targets.append(target.cpu().numpy())
            predicts.append(predict.cpu().numpy())

            # computing loss, metrics on test set
            loss = loss_fn(output, target)
            batch_size = data.shape[0]
            total_loss += loss.item() * batch_size
            for i, metric in enumerate(metric_fns):
                total_metrics[i] += metric(output, target) * batch_size

    target = np.concatenate(targets)
    predict = np.concatenate(predicts)
    confusion_m = confusion_matrix(target, predict)
    report = classification_report(target, predict)

    n_samples = len(data_loader.sampler)
    log = {'loss': total_loss / n_samples}
    log.update({
        met.__name__: total_metrics[i].item() / n_samples for i, met in enumerate(metric_fns)
    })
    log["confusion_matrix"] = confusion_m
    log["classification_report"] = report
    np.set_printoptions(threshold=sys.maxsize)
    logger.info(log)

    print(confusion_m)
    print(report)


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
