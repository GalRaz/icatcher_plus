import os
import options
import logging
import logger
import data
import models
import torch
import torch.multiprocessing as mp
import torch.distributed as dist
import numpy as np
from pathlib import Path
from visualize import calculate_confusion_matrix


def train_loop(rank, args):
    """
    Main training loop
    :param rank: id for distributed training. 0 is master.
    :param args: command line arguments
    :return:
    """
    # initialize
    args.rank = rank
    if args.gpu_id == "-1":
        args.device = "cpu"
    else:
        args.device = "cuda:{}".format(args.rank)
    setup(args)
    my_logger = logger.Logger(args)
    args.phase = "train"
    train_dataloader = data.MyDataLoader(args)
    args.phase = "val"
    val_dataloader = data.MyDataLoader(args)
    model = models.MyModel(args)
    # actual epoch loop
    for epoch in range(args.number_of_epochs):
        if args.distributed:
            dist.barrier()
            train_dataloader.dataloader.sampler.set_epoch(epoch)
        running_loss = 0.0
        running_corrects = 0
        num_datapoints = 0
        for batch_index, batch in enumerate(train_dataloader.dataloader):
            model.optimizer.zero_grad()
            output = model.network(batch)
            train_loss = model.loss_fn(output, batch["label"])
            _, predictions = torch.max(output, 1)
            train_loss.backward()
            model.optimizer.step()
            train_loss_np = train_loss.cpu().detach().numpy()
            correct = torch.sum(torch.eq(predictions, batch["label"])).item()
            my_logger.write_scaler("batch", "train_loss", train_loss_np, epoch*(len(train_dataloader.dataloader)) + batch_index)
            logging.info("train: epoch: {}, batch {} / {}, loss: {}, acc: {}".format(epoch,
                                                                                     batch_index,
                                                                                     len(train_dataloader.dataloader),
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
        val_loss_total = 0.
        if args.rank == 0:
            model.save_network(which_epoch=str(epoch))
            model.save_network(which_epoch="latest")
            model.network.train(mode=False)
            with torch.no_grad():
                running_loss = 0.0
                running_corrects = 0
                num_datapoints = 0
                for batch_index, batch in enumerate(val_dataloader.dataloader):
                    if args.distributed:  # might be unnecessary
                        output = model.network.module(batch)
                    else:
                        output = model.network(batch)
                    val_loss = model.loss_fn(output, batch["label"])
                    _, predictions = torch.max(output, 1)
                    num_datapoints += batch["label"].shape[0]
                    running_loss += val_loss.item() * batch["label"].shape[0]
                    running_corrects += torch.sum(torch.eq(predictions, batch["label"])).item()
                    logging.info("val: batch {} / {}".format(batch_index, len(val_dataloader.dataloader)))
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
            if args.distributed:
                val_loss_total = torch.Tensor([val_loss_total]).to(args.rank)
                dist.broadcast(tensor=val_loss_total, src=0)
                model.scheduler.step(val_loss_total.item())
            else:
                model.scheduler.step(val_loss_total)
        else:
            model.scheduler.step()


def setup(args):
    if args.rank == 0:  # setup logging
        if args.log:
            logging_file = Path(args.experiment_path, "log")
            logging.basicConfig(filename=str(logging_file), filemode='w', level=args.verbosity.upper())
        else:
            logging.basicConfig(level=args.verbosity.upper())
    if args.distributed:
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = args.port
        # initialize the process group
        dist.init_process_group("nccl", rank=args.rank, world_size=args.world_size)  # gloo / nccl
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)


def cleanup():
    dist.destroy_process_group()


def predict_on_preprocessed(args):
    """
    uses a simple "foward" on the validation set to estimate performance (prints confusion matrix)
    todo: move to test.py
    :param args:
    :return:
    """
    args.phase = "val"
    val_dataset = data.MyDataLoader(args)
    model = models.MyModel(args)
    model.network.train(mode=False)
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
    norm_mat, mat, total_acc = calculate_confusion_matrix(None, None,
                                                          Path(args.experiment_path, "confusion_matrix_{}.png".format(args.use_disjoint)),
                                                          confusion_matrix.numpy())
    # val_loss_total = running_loss / num_datapoints
    val_acc_total = (running_corrects / num_datapoints) * 100
    # logging.info("validation loss: {}".format(val_loss_total))
    logging.info("validation acc: {}%".format(val_acc_total))
    logging.info("per class acc: {}".format(norm_mat.diagonal()))
    logging.info("confusion matrix (normalized): {}".format(norm_mat))

    ##############################################################################################################


def get_data_ready_to_use(args, paths):
    img_files_seg, box_files_seg, class_seg = paths
    imgs = []
    for img_file in img_files_seg:
        img = Image.open(args.opt.dataset_folder / "faces" / img_file).convert('RGB')
        img = args.img_processor(img)
        imgs.append(img)
    imgs = torch.stack(imgs)

    boxs = []
    for box_file in box_files_seg:
        box = np.load(args.opt.dataset_folder / "faces" / box_file, allow_pickle=True).item()
        box = torch.tensor([box['face_size'], box['face_ver'], box['face_hor'], box['face_height'], box['face_width']])
        boxs.append(box)
    boxs = torch.stack(boxs)
    boxs = boxs.float()
    imgs = imgs.to(args.opt.device)
    boxs = boxs.to(args.opt.device)
    class_seg = torch.as_tensor(class_seg).to(args.opt.device)
    return {
        'imgs': imgs,  # n x 3 x 100 x 100
        'boxs': boxs,  # n x 5
        'label': class_seg,  # n x 1
        'path': img_files_seg[2]  # n x 1
    }


##################################
def sample(path, k, l):
    calibration_path = path[:k]
    validation_path = path[k:k + l]
    return calibration_path, validation_path


############################

def get_new_metamodel_weights(meta_model, temp_model, validation_set):
    meta_model.optimizer.zero_grad()
    output = temp_model.network(validation_set)
    train_loss = temp_model.innerScheduler(output, validation_set["label"])
    train_loss.backward()
    meta_model.optimizer.step(train_loss)


############################
def innerLoop(model, batch):
    #    running_loss = 0.0
    #    running_corrects = 0
    #    num_datapoints = 0
    model.optimizer.zero_grad()
    output = model.network(batch)  ### does this returns a list of results acording to the network?
    train_loss = model.innerScheduler(output, batch["label"])
    _, predictions = torch.max(output, 1)
    train_loss.backward()
    model.innerOptimizer.step()


#    train_loss_np = train_loss.cpu().detach().numpy()
#    correct = torch.sum(torch.eq(predictions, batch["label"])).item()
#   my_logger.write_scaler("batch", "train_loss", train_loss_np,
#                          epoch * (len(train_dataloader.dataloader)) + batch_index)
#   logging.info("train: epoch: {}, batch {} / {}, loss: {}, acc: {}".format(epoch,
#                                                                            batch_index,
#                                                                            len(train_dataloader.dataloader),
#                                                                            train_loss_np,
#                                                                            (correct / batch["label"].shape[0]) * 100))
#    num_datapoints = batch["label"].shape[0]
#    running_loss = train_loss.item() * batch["label"].shape[0]
#    running_corrects = torch.sum(torch.eq(predictions, batch["label"])).item()
#    return running_loss, running_corrects, num_datapoints

##############################
def MAMLtrain(rank, args):
    args.rank = rank
    if args.gpu_id == "-1":
        args.device = "cpu"
    else:
        args.device = "cuda:{}".format(args.rank)
    setup(args)
    my_logger = logger.Logger(args)

    ## might be good to set args.phase to one of :
    # {train_calibration , train_validation , test_calibration , test_calibration}

    args.phase = "train"
    train_dataloader = data.MyDataLoader(args)
    args.phase = "test"
    test_dataloader = data.MyDataLoader(args)
    meta_model = models.MAMLmodel(args)

    # model and optimizer for outer loop
    outer_optim = meta_model.optimizer

    # optimizer for inner loop
    test_model = meta_model.clone()
    inner_optim = meta_model.innerOptimizer
    for i in range(args.number_of_epochs):  ## number of trials
        for task_index, task in enumerate(train_dataloader.dataloader):  # Ti in p(T)
            task_model = test_model.clone()
            train_samples, test_samples = sample(task, test_model.K, test_model.L)
            calibration_samples = []
            validation_samples = []
            for train_sample in train_samples:
                calibration_samples.append(get_data_ready_to_use(args, train_sample))
            for test_sample in test_samples:
                validation_samples.append(get_data_ready_to_use(args, test_sample))

            innerLoop(task_model, calibration_samples)  # updating test model weights
            get_new_metamodel_weights(meta_model, task_model, validation_samples)


###################################################################################################################


if __name__ == "__main__":
    args = options.parse_arguments_for_training()
    if args.distributed:
        if args.train_type == "MAML":
            mp.spawn(MAMLtrain,
                     args=(args,),
                     nprocs=args.world_size,
                     join=True)
        mp.spawn(train_loop,
                 args=(args,),
                 nprocs=args.world_size,
                 join=True)

    else:
        if args.train_type == "MAML":
            MAMLtrain(0,args)
        train_loop(0, args)