import options
import logging
import logger
import data
import models
import torch
import numpy as np


def train_loop(args):
    """
    Main training loop. Try to keep as clean as possible.
    :param args: command line arguments
    :return:
    """
    my_logger = logger.Logger(args)
    args.is_train = True
    train_dataset = data.MyDataLoader(args)
    args.is_train = False
    val_dataset = data.MyDataLoader(args)
    model = models.MyModel(args)
    model.optimizer.zero_grad()
    for epoch in range(args.number_of_epochs):
        train_loss_total = []
        for batch_index, (input, target) in enumerate(train_dataset):
            output = model.network(input)
            train_loss = model.network.loss_fn(output, target)
            train_loss.backward()
            model.optimizer.step()
            model.optimizer.zero_grad()
            train_loss_np = train_loss.cpu().detach().numpy()
            my_logger.write_scaler("batch", "train_loss_total", train_loss_np, epoch*(len(train_dataset) // args.batch_size) + batch_index)
            logging.info("train: epoch: {}, batch {} / {}, loss: {}".format(epoch,
                                                                            batch_index,
                                                                            len(train_dataset) // args.batch_size,
                                                                            train_loss_np))
            train_loss_total.append(train_loss_np)
        train_loss_total = np.mean(np.array(train_loss_total))
        my_logger.write_scaler("epoch", "train_loss_total", train_loss_total, epoch)
        logging.info("train: epoch: {}, training loss: {}".format(epoch,
                                                                  train_loss_total))
        model.save_network(which_epoch=str(epoch))
        model.save_network(which_epoch="latest")
        with torch.no_grad():
            val_loss_total = []
            for input, target in val_dataset:
                model.optimizer.zero_grad()
                output = model.network(input)
                val_loss = model.network.loss_fn(output, target)
                val_loss_total.append(val_loss.cpu().detach().numpy())
            val_loss_total = np.mean(np.array(val_loss_total))
            my_logger.write_scaler("epoch", "val_loss_total", val_loss_total, epoch)
            logging.info("validation: epoch: {}, loss: {}".format(epoch, val_loss_total))
        my_logger.write_scaler("epoch", "learning rate", model.optimizer.param_groups[0]['lr'], epoch)
        logging.info("lr: {}".format(model.optimizer.param_groups[0]['lr']))
        if args.lr_policy == "plateau":
            model.scheduler.step(val_loss_total)
        else:
            model.scheduler.step()


if __name__ == "__main__":
    args = options.parse_arguments()
    if args.log:
        args.log.parent.mkdir(parents=True, exist_ok=True)
        logging.basicConfig(filename=args.log, filemode='w', level=args.verbosity.upper())
    else:
        logging.basicConfig(level=args.verbosity.upper())
    logging.info("Entering training loop")
    train_loop(args)
    logging.info("Finished training loop")
