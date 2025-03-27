import yaml
import torch
from accelerate import Accelerator

from models import get_model
from loaders import get_loaders
from criterion import get_criterion
from optimizer import get_optimizer, get_scheduler

from utils.train_utils import setup_distributed

def main(config):
    setup_distributed()

    accelerator = Accelerator()

    train_loader, valid_loader = get_loaders(config)


    # Initialize model, criterion, optimizer, scheduler
    model = get_model(config)
    criterion = get_criterion(config)
    optimizer = get_optimizer(model.parameters(), config)
    scheduler = get_scheduler(optimizer, config, train_loader)

    accelerator.print(f"----- Model -----")
    accelerator.print(model)
    accelerator.print(f"-----------------")

    accelerator.print(f"----- Criterion, Optimizer, Scheduler -----")
    accelerator.print(criterion)
    accelerator.print(optimizer)
    accelerator.print(scheduler)
    accelerator.print(f"-------------------------------------------")

    # Prepare everything with accelerator
    model, optimizer, scheduler, criterion, train_loader, valid_loader = accelerator.prepare(
        model, optimizer, scheduler, criterion, train_loader, valid_loader
    )

    for epoch in range(config.train.epochs):
        train_one_epoch(model, train_loader, valid_loader, criterion, optimizer, scheduler, accelerator, epoch, config)
# 