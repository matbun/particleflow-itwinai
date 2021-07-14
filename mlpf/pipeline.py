import sys
import os
import yaml
import json
import datetime
import glob
import random
import platform
import numpy as np
from pathlib import Path
import click
from tqdm import tqdm

import tensorflow as tf
from tensorflow.keras import mixed_precision
import tensorflow_addons as tfa

from tfmodel.data import Dataset
from tfmodel.datasets import CMSDatasetFactory, DelphesDatasetFactory
from tfmodel.model_setup import (
    make_model,
    configure_model_weights,
    LearningRateLoggingCallback,
    prepare_callbacks,
    FlattenedCategoricalAccuracy,
    eval_model,
    freeze_model,
)

from tfmodel.utils import (
    get_lr_schedule,
    create_experiment_dir,
    get_strategy,
    make_weight_function,
    load_config,
    compute_weights_invsqrt,
    compute_weights_none,
    get_train_val_datasets,
    targets_multi_output,
    get_dataset_def,
    prepare_val_data,
    set_config_loss,
    get_loss_dict,
    parse_config,
    get_best_checkpoint,
    delete_all_but_best_checkpoint,
    get_heptfds_dataset,
)

from tfmodel.onecycle_scheduler import OneCycleScheduler, MomentumOneCycleScheduler
from tfmodel.lr_finder import LRFinder


@click.group()
@click.help_option("-h", "--help")
def main():
    pass


@main.command()
@click.help_option("-h", "--help")
@click.option("-c", "--config", help="configuration file", type=click.Path())
@click.option("-w", "--weights", default=None, help="trained weights to load", type=click.Path())
@click.option("--ntrain", default=None, help="override the number of training events", type=int)
@click.option("--ntest", default=None, help="override the number of testing events", type=int)
@click.option("-r", "--recreate", help="force creation of new experiment dir", is_flag=True)
@click.option("-p", "--prefix", default="", help="prefix to put at beginning of training dir name", type=str)
def train(config, weights, ntrain, ntest, recreate, prefix):
    """Train a model defined by config"""
    config_file = config
    config, config_file_stem, global_batch_size, n_train, n_test, n_epochs, weights = parse_config(
        config, ntrain, ntest, weights
    )

    dataset_def = get_dataset_def(config)
    X_val, ygen_val, ycand_val = prepare_val_data(config, dataset_def, single_file=True)

    ds_train, ds_test = get_heptfds_dataset(config)
    ds_train_r = ds_train.repeat(config["setup"]["num_epochs"])
    ds_test_r = ds_test.repeat(config["setup"]["num_epochs"])

    if recreate or (weights is None):
        outdir = create_experiment_dir(prefix=prefix + config_file_stem + "_", suffix=platform.node())
    else:
        outdir = str(Path(weights).parent)

    # Decide tf.distribute.strategy depending on number of available GPUs
    strategy, maybe_global_batch_size = get_strategy(global_batch_size)
    # If using more than 1 GPU, we scale the batch size by the number of GPUs
    if maybe_global_batch_size is not None:
        global_batch_size = maybe_global_batch_size
    total_steps = n_epochs * n_train // global_batch_size
    lr = float(config["setup"]["lr"])

    with strategy.scope():
        lr_schedule, optim_callbacks = get_lr_schedule(config, lr=lr, steps=total_steps)
        opt = tf.keras.optimizers.Adam(learning_rate=lr_schedule)

        if config["setup"]["dtype"] == "float16":
            model_dtype = tf.dtypes.float16
            policy = mixed_precision.Policy("mixed_float16")
            mixed_precision.set_global_policy(policy)
            opt = mixed_precision.LossScaleOptimizer(opt)
        else:
            model_dtype = tf.dtypes.float32

        model = make_model(config, model_dtype)

        # Run model once to build the layers
        model(tf.cast(X_val[:1], model_dtype))

        initial_epoch = 0
        if weights:
            # We need to load the weights in the same trainable configuration as the model was set up
            configure_model_weights(model, config["setup"].get("weights_config", "all"))
            model.load_weights(weights, by_name=True)
            initial_epoch = int(weights.split("/")[-1].split("-")[1])
        model(tf.cast(X_val[:1], model_dtype))

        config = set_config_loss(config, config["setup"]["trainable"])
        configure_model_weights(model, config["setup"]["trainable"])
        model(tf.cast(X_val[:1], model_dtype))

        loss_dict, loss_weights = get_loss_dict(config)
        model.compile(
            loss=loss_dict,
            optimizer=opt,
            sample_weight_mode="temporal",
            loss_weights=loss_weights,
            metrics={
                "cls": [
                    FlattenedCategoricalAccuracy(name="acc_unweighted", dtype=tf.float64),
                    FlattenedCategoricalAccuracy(use_weights=True, name="acc_weighted", dtype=tf.float64),
                ]
            },
        )
        model.summary()

        callbacks = prepare_callbacks(
            model,
            outdir,
            X_val[: config["setup"]["batch_size"]],
            ycand_val[: config["setup"]["batch_size"]],
            targets_multi_output(config['dataset']['num_output_classes']),
            config["dataset"]["num_output_classes"],
        )
        callbacks.append(optim_callbacks)

        fit_result = model.fit(
            ds_train_r,
            validation_data=ds_test_r,
            epochs=initial_epoch + n_epochs,
            callbacks=callbacks,
            steps_per_epoch=n_train // global_batch_size,
            validation_steps=n_test // global_batch_size,
            initial_epoch=initial_epoch,
        )
        history_path = Path(outdir) / "history"
        history_path = str(history_path)
        with open("{}/history.json".format(history_path), "w") as fi:
            json.dump(fit_result.history, fi)
        model.save(outdir + "/model_full", save_format="tf")

        print("Training done.")

    print("Starting evaluation...")
    eval_dir = Path(outdir) / "evaluation"
    eval_dir.mkdir()
    eval_dir = str(eval_dir)
    # eval_model(X_val, ygen_val, ycand_val, model, config, eval_dir, global_batch_size)
    create_pred_files(config_file, outdir)
    print("Evaluation done.")
    freeze_model(model, config, outdir)


@main.command()
@click.help_option("-h", "--help")
@click.option("-t", "--train_dir", required=True, help="directory containing a completed training", type=click.Path())
@click.option("-c", "--config", help="configuration file", type=click.Path())
@click.option("-w", "--weights", default=None, help="trained weights to load", type=click.Path())
@click.option("-e", "--evaluation_dir", help="optionally specify evaluation output dir", type=click.Path())
def evaluate(config, train_dir, weights=None, evaluation_dir=None):
    create_pred_files(config, train_dir, weights, evaluation_dir)

def create_pred_files(config, train_dir, weights=None, evaluation_dir=None):
    """Evaluate the trained model in train_dir"""
    config, _, global_batch_size, _, _, _, weights = parse_config(config, weights=weights)
    # Switch off multi-output for the evaluation for backwards compatibility
    config["setup"]["multi_output"] = False

    if evaluation_dir is None:
        eval_dir = str(Path(train_dir) / "evaluation")
    else:
        eval_dir = evaluation_dir
    Path(eval_dir).mkdir(parents=True, exist_ok=True)

    if config["setup"]["dtype"] == "float16":
        model_dtype = tf.dtypes.float16
        policy = mixed_precision.Policy("mixed_float16")
        mixed_precision.set_global_policy(policy)
        opt = mixed_precision.LossScaleOptimizer(opt)
    else:
        model_dtype = tf.dtypes.float32

    dataset_def = get_dataset_def(config)
    X_val, ygen_val, ycand_val = prepare_val_data(config, dataset_def, single_file=False)

    strategy, maybe_global_batch_size = get_strategy(global_batch_size)
    if maybe_global_batch_size is not None:
        global_batch_size = maybe_global_batch_size

    with strategy.scope():
        model = make_model(config, model_dtype)

        # Evaluate model once to build the layers
        model(tf.cast(X_val[:1], model_dtype))

        if weights:
            # need to load the weights in the same trainable configuration as the model was set up
            configure_model_weights(model, config["setup"].get("weights_config", "all"))
            model.load_weights(weights, by_name=True)
        else:
            weights = get_best_checkpoint(train_dir)
            print("Loading best weights that could be found from {}".format(weights))
            model.load_weights(weights, by_name=True)
        model(tf.cast(X_val[:1], model_dtype))

        model.compile()
        eval_model(X_val, ygen_val, ycand_val, model, config, eval_dir, global_batch_size)
        freeze_model(model, config, train_dir)


@main.command()
@click.help_option("-h", "--help")
@click.option("-c", "--config", help="configuration file", type=click.Path())
@click.option("-o", "--outdir", help="output directory", type=click.Path(), default=".")
@click.option("-n", "--figname", help="name of saved figure", type=click.Path(), default="lr_finder.jpg")
@click.option("-l", "--logscale", help="use log scale on y-axis in figure", default=False, is_flag=True)
def find_lr(config, outdir, figname, logscale):
    """Run the Learning Rate Finder to produce a batch loss vs. LR plot from
    which an appropriate LR-range can be determined"""
    config, _, global_batch_size, n_train, _, _, _ = parse_config(config)
    ds_train, _ = get_heptfds_dataset(config)

    # Decide tf.distribute.strategy depending on number of available GPUs
    strategy, maybe_global_batch_size = get_strategy(global_batch_size)

    # If using more than 1 GPU, we scale the batch size by the number of GPUs
    if maybe_global_batch_size is not None:
        global_batch_size = maybe_global_batch_size

    dataset_def = get_dataset_def(config)
    X_val, _, _ = prepare_val_data(config, dataset_def, single_file=True)

    with strategy.scope():
        opt = tf.keras.optimizers.Adam(learning_rate=1e-7)  # This learning rate will be changed by the lr_finder
        if config["setup"]["dtype"] == "float16":
            model_dtype = tf.dtypes.float16
            policy = mixed_precision.Policy("mixed_float16")
            mixed_precision.set_global_policy(policy)
            opt = mixed_precision.LossScaleOptimizer(opt)
        else:
            model_dtype = tf.dtypes.float32

        model = make_model(config, model_dtype)
        config = set_config_loss(config, config["setup"]["trainable"])

        # Run model once to build the layers
        model(tf.cast(X_val[:1], model_dtype))

        configure_model_weights(model, config["setup"]["trainable"])

        loss_dict, loss_weights = get_loss_dict(config)
        model.compile(
            loss=loss_dict,
            optimizer=opt,
            sample_weight_mode="temporal",
            loss_weights=loss_weights,
            metrics={
                "cls": [
                    FlattenedCategoricalAccuracy(name="acc_unweighted", dtype=tf.float64),
                    FlattenedCategoricalAccuracy(use_weights=True, name="acc_weighted", dtype=tf.float64),
                ]
            },
        )
        model.summary()

        max_steps = 200
        lr_finder = LRFinder(max_steps=max_steps)
        callbacks = [lr_finder]

        model.fit(
            ds_train,
            epochs=max_steps,
            callbacks=callbacks,
            steps_per_epoch=1,
        )

        lr_finder.plot(save_dir=outdir, figname=figname, log_scale=logscale)


@main.command()
@click.help_option("-h", "--help")
@click.option("-t", "--train_dir", help="training directory", type=click.Path())
@click.option("-d", "--dry_run", help="do not delete anything", is_flag=True, default=False)
def delete_all_but_best_ckpt(train_dir, dry_run):
    """Delete all checkpoint weights in <train_dir>/weights/ except the one with lowest loss in its filename."""
    delete_all_but_best_checkpoint(train_dir, dry_run)


@main.command()
@click.help_option("-h", "--help")
@click.option("-c", "--config", help="configuration file", type=click.Path())
@click.option("--ntrain", default=None, help="override the number of training events", type=int)
@click.option("--ntest", default=None, help="override the number of testing events", type=int)
def debug_data(config, ntrain, ntest):
    """Train a model defined by config"""
    config, config_file_stem, global_batch_size, n_train, n_test, n_epochs, weights = parse_config(
        config, ntrain, ntest, weights=None,
    )

    dataset_def = get_dataset_def(config)
    ds_train, ds_test, dataset_transform = get_train_val_datasets(config, global_batch_size=1, n_train=n_train, n_test=n_test)

    # cand_counts = np.zeros(8)
    # for data_item in tqdm(ds_train, desc="Counting"):
    #     import pdb; pdb.set_trace()
    #     cand_vals, cand_count = np.unique(np.argmax(data_item[1]['cls'], axis=2), return_counts=True)
    #     cand_counts[cand_vals.astype("int32")] += cand_count
    # print("cand_counts: ", cand_counts)

    dsf = CMSDatasetFactory(config)
    ds_train, _ = dsf.get_dataset(split="train")
    ds_test, _ = dsf.get_dataset(split="test")
    for data_item in tqdm(ds_train, desc="Counting"):
        import pdb; pdb.set_trace()


if __name__ == "__main__":
    main()
