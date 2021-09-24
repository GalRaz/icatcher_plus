import options
import logging
import logger
import data
import models
import torch
import numpy as np
from pathlib import Path


def train_loop(args):
    """
    Main training loop. Try to keep as clean as possible.
    :param args: command line arguments
    :return:
    """
    my_logger = logger.Logger(args)
    args.phase = "train"
    train_dataset = data.MyDataLoader(args)
    args.phase = "val"
    val_dataset = data.MyDataLoader(args)
    model = models.MyModel(args)
    for epoch in range(args.number_of_epochs):
        running_loss = 0.0
        running_corrects = 0
        num_datapoints = 0
        for batch_index, batch in enumerate(train_dataset):
            model.optimizer.zero_grad()
            output = model.network(batch)
            train_loss = model.network.loss_fn(output, batch["label"])
            _, predictions = torch.max(output, 1)
            train_loss.backward()
            model.optimizer.step()
            train_loss_np = train_loss.cpu().detach().numpy()
            my_logger.write_scaler("batch", "train_loss", train_loss_np, epoch*(len(train_dataset) // args.batch_size) + batch_index)
            logging.info("train: epoch: {}, batch {} / {}, loss: {}".format(epoch,
                                                                            batch_index,
                                                                            len(train_dataset) // args.batch_size,
                                                                            train_loss_np))
            num_datapoints += batch["label"].shape[0]
            running_loss += train_loss.item() * batch["label"].shape[0]
            running_corrects += torch.sum(torch.eq(predictions, batch["label"])).item()
        epoch_loss = running_loss / num_datapoints
        epoch_acc = running_corrects / num_datapoints
        my_logger.write_scaler("epoch", "train_loss", epoch_loss, epoch)
        my_logger.write_scaler("epoch", "train_acc", epoch_acc, epoch)
        logging.info("train: epoch: {}, training loss: {}".format(epoch,
                                                                  epoch_loss))
        logging.info("train: epoch: {}, training acc: {}".format(epoch,
                                                                 epoch_acc))
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
            my_logger.write_scaler("epoch", "val_loss", val_loss_total, epoch)
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
        logging_file = Path(args.experiment_path, "log")
        logging.basicConfig(filename=str(logging_file), filemode='w', level=args.verbosity.upper())
    else:
        logging.basicConfig(level=args.verbosity.upper())
    logging.info("Entering training loop")
    train_loop(args)
    logging.info("Finished training loop")
