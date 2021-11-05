import options
import logging
import logger
import data
import models
import torch
from pathlib import Path
from visualize import calculate_confusion_matrix


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
            train_loss = model.loss_fn(output, batch["label"])
            _, predictions = torch.max(output, 1)
            train_loss.backward()
            model.optimizer.step()
            train_loss_np = train_loss.cpu().detach().numpy()
            correct = torch.sum(torch.eq(predictions, batch["label"])).item()
            my_logger.write_scaler("batch", "train_loss", train_loss_np, epoch*(len(train_dataset) // args.batch_size) + batch_index)
            logging.info("train: epoch: {}, batch {} / {}, loss: {}, acc: {}".format(epoch,
                                                                                     batch_index,
                                                                                     len(train_dataset) // args.batch_size,
                                                                                     train_loss_np,
                                                                                     (correct / batch["label"].shape[0]) * 100))
            num_datapoints += batch["label"].shape[0]
            running_loss += train_loss.item() * batch["label"].shape[0]
            running_corrects += torch.sum(torch.eq(predictions, batch["label"])).item()
        epoch_loss = running_loss / num_datapoints
        epoch_acc = (running_corrects / num_datapoints) * 100
        my_logger.write_scaler("epoch", "train_loss", epoch_loss, epoch)
        my_logger.write_scaler("epoch", "train_acc", epoch_acc, epoch)
        logging.info("train: epoch: {}, training loss: {}".format(epoch,
                                                                  epoch_loss))
        logging.info("train: epoch: {}, training acc: {}".format(epoch,
                                                                 epoch_acc))
        model.save_network(which_epoch=str(epoch))
        model.save_network(which_epoch="latest")
        model.network.train(mode=False)
        with torch.no_grad():
            running_loss = 0.0
            running_corrects = 0
            num_datapoints = 0
            for batch in val_dataset:
                output = model.network(batch)
                val_loss = model.loss_fn(output, batch["label"])
                _, predictions = torch.max(output, 1)
                num_datapoints += batch["label"].shape[0]
                running_loss += val_loss.item() * batch["label"].shape[0]
                running_corrects += torch.sum(torch.eq(predictions, batch["label"])).item()
        model.network.train(mode=True)
        val_loss_total = running_loss / num_datapoints
        val_acc_total = (running_corrects / num_datapoints) * 100
        my_logger.write_scaler("epoch", "val_loss", val_loss_total, epoch)
        my_logger.write_scaler("epoch", "val_acc", val_acc_total, epoch)
        logging.info("validation: epoch: {}, loss: {}".format(epoch, val_loss_total))
        logging.info("validation: epoch: {}, acc: {}".format(epoch, val_acc_total))
        my_logger.write_scaler("epoch", "learning rate", model.optimizer.param_groups[0]['lr'], epoch)
        logging.info("lr: {}".format(model.optimizer.param_groups[0]['lr']))
        if args.lr_policy == "plateau":
            model.scheduler.step(val_loss_total)
        else:
            model.scheduler.step()


def predict_on_preprocessed(args):
    args.phase = "val"
    val_dataset = data.MyDataLoader(args)
    model = models.MyModel(args)
    confusion_matrix = torch.zeros(3, 3)
    with torch.no_grad():
        # running_loss = 0.0
        running_corrects = 0
        num_datapoints = 0
        for i, batch in enumerate(val_dataset):
            logging.info("batch: {} / {}".format(i, len(val_dataset) // args.batch_size))
            output = model.network(batch)
            # val_loss = model.loss_fn(output, batch["label"])
            _, predictions = torch.max(output, 1)
            for t, p in zip(batch["label"].cpu().view(-1), predictions.cpu().view(-1)):
                confusion_matrix[t.long(), p.long()] += 1
            num_datapoints += batch["label"].shape[0]
            # running_loss += val_loss.item() * batch["label"].shape[0]
            running_corrects += torch.sum(torch.eq(predictions, batch["label"])).item()
    mat, total_acc = calculate_confusion_matrix(None, None,
                                                Path(args.experiment_path, "conf_{}.png".format(args.use_disjoint)),
                                                confusion_matrix.numpy())
    # val_loss_total = running_loss / num_datapoints
    val_acc_total = (running_corrects / num_datapoints) * 100
    # logging.info("validation loss: {}".format(val_loss_total))
    logging.info("validation acc: {}".format(val_acc_total))
    logging.info("per class acc: {}".format(mat.diagonal()))
    logging.info("confusion matrix (normalized): {}".format(mat))


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
