from argparse import ArgumentParser
from torch.utils.data import DataLoader
from torch.optim import Adam
from IQADataset.IQADataset import IQADataset
from IQAnetwork.mainNetwork import *
import numpy as np
from scipy import stats
import os, yaml

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

try:
    from tensorboardX import SummaryWriter
except ImportError:
    raise RuntimeError("No tensorboardX package is found. Please install with the command: \npip install tensorboardX")

from ignite.engine import Events, create_supervised_trainer, create_supervised_evaluator
from ignite.metrics.metric import Metric


def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def loss_fn(y_pred, y):
    return F.l1_loss(y_pred, y[0])


class IQAPerformance(Metric):
    """
    Evaluation of IQA methods using SROCC, KROCC, PLCC, RMSE,      MAE, OR.

    `update` must receive output of the form (y_pred, y).
    """

    def reset(self):
        self._y_pred = []
        self._y = []
        self._y_std = []

    def update(self, output):
        y_pred, y = output

        y_pred1 = y_pred.cpu()
        y_pred1 = y_pred1.numpy()
        suuy = sum(sum(y_pred1))
        suuy = torch.tensor(suuy)
        suuy1 = suuy.cuda()

        self._y.append(y[0].item())
        self._y_std.append(y[1].item())
        self._y_pred.append(torch.mean(y_pred).item())

    def compute(self):
        sq = np.reshape(np.asarray(self._y), (-1,))
        sq_std = np.reshape(np.asarray(self._y_std), (-1,))
        q = np.reshape(np.asarray(self._y_pred), (-1,))
        srocc = stats.spearmanr(sq, q)[0]
        krocc = stats.stats.kendalltau(sq, q)[0]
        plcc = stats.pearsonr(sq, q)[0]
        rmse = np.sqrt(((sq - q) ** 2).mean())
        mae = np.abs((sq - q)).mean()
        outlier_ratio = (np.abs(sq - q) > 2 * sq_std).mean()
        return srocc, krocc, plcc, rmse, mae, outlier_ratio


def get_data_loaders(config, train_batch_size, exp_id=0):
    train_dataset = IQADataset(config, exp_id, 'train')
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=train_batch_size,
                                               shuffle=True,
                                               num_workers=0)

    val_dataset = IQADataset(config, exp_id, 'val')
    val_loader = torch.utils.data.DataLoader(val_dataset)

    if config['test_ratio']:
        test_dataset = IQADataset(config, exp_id, 'test')
        test_loader = torch.utils.data.DataLoader(test_dataset)

        return train_loader, val_loader, test_loader

    return train_loader, val_loader


def create_summary_writer(model, data_loader, log_dir='tensorboard_logs'):
    writer = SummaryWriter(log_dir=log_dir)
    data_loader_iter = iter(data_loader)
    x, y = next(data_loader_iter)
    try:
        writer.add_graph(model, x)
    except Exception as e:
        print("Failed to save model graph: {}".format(e))
    return writer


def run(train_batch_size, epochs, lr, weight_decay, config, exp_id, log_dir, trained_model_file, save_result_file,
        disable_gpu=False):
    if config['test_ratio']:
        train_loader, val_loader, test_loader = get_data_loaders(config, train_batch_size, exp_id)
    else:
        train_loader, val_loader = get_data_loaders(config, train_batch_size, exp_id)

    device = torch.device("cuda" if not disable_gpu and torch.cuda.is_available() else "cpu")
    print('cnn device:'+str(device))
    model = CNNIQAnet()
    writer = create_summary_writer(model, train_loader, log_dir)
    model = model.to(device)
    optimizer = Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    global best_criterion
    best_criterion = -1
    trainer = create_supervised_trainer(model, optimizer, loss_fn,
                                        device=device)  
    evaluator = create_supervised_evaluator(model,
                                            metrics={'IQA_performance': IQAPerformance()},
                                            device=device)

    @trainer.on(Events.ITERATION_COMPLETED)
    def log_training_loss(engine):
        writer.add_scalar("training/loss", engine.state.output, engine.state.iteration)

    @trainer.on(Events.EPOCH_COMPLETED)
    def log_validation_results(engine):
        evaluator.run(val_loader)
        metrics = evaluator.state.metrics
        SROCC, KROCC, PLCC, RMSE, MAE, OR = metrics['IQA_performance']
        print(
            "Validation Results"
            " - Epoch: {} SROCC: {:.4f} KROCC: {:.4f} PLCC: {:.4f} RMSE: {:.4f} MAE: {:.4f} OR: {:.2f}%".format(engine.state.epoch, SROCC, KROCC, PLCC, RMSE, MAE, 100 * OR))
        writer.add_scalar("validation/SROCC", SROCC, engine.state.epoch)
        writer.add_scalar("validation/KROCC", KROCC, engine.state.epoch)
        writer.add_scalar("validation/PLCC", PLCC, engine.state.epoch)
        writer.add_scalar("validation/RMSE", RMSE, engine.state.epoch)
        writer.add_scalar("validation/MAE", MAE, engine.state.epoch)
        writer.add_scalar("validation/OR", OR, engine.state.epoch)
        global best_criterion
        global best_epoch
        if SROCC > best_criterion and engine.state.epoch > 100:
            best_criterion = SROCC
            best_epoch = engine.state.epoch
            torch.save(model.state_dict(), trained_model_file)  

    @trainer.on(Events.EPOCH_COMPLETED)
    def log_testing_results(engine):
        if config["test_ratio"] > 0 and config['test_during_training']:
            evaluator.run(test_loader)
            metrics = evaluator.state.metrics
            SROCC, KROCC, PLCC, RMSE, MAE, OR = metrics['IQA_performance']
            print(
                "Testing Results   "
                " - Epoch: {} SROCC: {:.4f} KROCC: {:.4f} PLCC: {:.4f} RMSE: {:.4f} MAE: {:.4f} OR: {:.2f}%".format(engine.state.epoch, SROCC, KROCC, PLCC, RMSE, MAE, 100 * OR))
            writer.add_scalar("testing/SROCC", SROCC, engine.state.epoch)
            writer.add_scalar("testing/KROCC", KROCC, engine.state.epoch)
            writer.add_scalar("testing/PLCC", PLCC, engine.state.epoch)
            writer.add_scalar("testing/RMSE", RMSE, engine.state.epoch)
            writer.add_scalar("testing/MAE", MAE, engine.state.epoch)
            writer.add_scalar("testing/OR", OR, engine.state.epoch)

    @trainer.on(Events.COMPLETED)
    def final_testing_results(engine):
        if config["test_ratio"] > 0:
            model.load_state_dict(torch.load(trained_model_file))
            evaluator.run(test_loader)
            metrics = evaluator.state.metrics
            SROCC, KROCC, PLCC, RMSE, MAE, OR = metrics['IQA_performance']
            global best_epoch
            print(
                "Final Test Results - Epoch: {} SROCC: {:.4f} KROCC: {:.4f} PLCC: {:.4f} RMSE: {:.4f} MAE: {:.4f} OR: {:.2f}%"
                    .format(best_epoch, SROCC, KROCC, PLCC, RMSE, MAE, 100 * OR))
            np.save(save_result_file, (SROCC, KROCC, PLCC, RMSE, MAE, OR))
    trainer.run(train_loader, max_epochs=epochs)
    writer.close()


if __name__ == "__main__":
    cuda = torch.cuda.is_available()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    parser = ArgumentParser(description='PyTorch CNNIQA')
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--weight_decay', type=float, default=0.0)
    parser.add_argument('--config', default='config.yaml', type=str)
    parser.add_argument('--exp_id', default='256', type=str)
    parser.add_argument('--database', default='LIVE', type=str)
    parser.add_argument('--model', default='MB-CNN', type=str)
    parser.add_argument("--log_dir", type=str, default="tensorboard_logs")
    parser.add_argument('--disable_gpu', default=False)
    args = parser.parse_args()
    with open(args.config) as f:
        config = yaml.load(f)

    log_dir = args.log_dir + '/EXP{}-{}-{}-lr={}-train'.format(args.exp_id, args.database, args.model, args.lr)
    ensure_dir('checkpoints')

    trained_model_file = 'checkpoints/{}-{}-EXP{}-lr={}-6'.format(args.model, args.database, args.exp_id, args.lr)
    ensure_dir('results')

    save_result_file = 'results/{}-{}-EXP{}-lr={}'.format(args.model, args.database, args.exp_id, args.lr)

    run(args.batch_size, args.epochs, args.lr, args.weight_decay, config, args.exp_id,
      log_dir, trained_model_file, save_result_file, args.disable_gpu)
























































