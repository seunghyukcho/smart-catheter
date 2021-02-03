import json

import torch
import argparse
import importlib
import numpy as np
import torch.nn as nn
from pathlib import Path
from torch.optim import Adam
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from arguments import get_task_parser, add_train_args


if __name__ == "__main__":
    # Read task argument first, and determine the other arguments
    task_parser = get_task_parser()

    task_name = task_parser.parse_known_args()[0].task
    task_module = importlib.import_module(f'tasks.{task_name}')
    task_dataset = getattr(task_module, 'Dataset')
    task_model = getattr(task_module, 'Model')

    parser = argparse.ArgumentParser()
    add_train_args(parser)
    getattr(task_module, 'add_task_args')(parser)
    args = parser.parse_args()

    # Seed settings
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    print('Loading train dataset...')
    train_dataset = task_dataset(args, args.train_data)
    train_loader = DataLoader(dataset=train_dataset, batch_size=args.batch_size, shuffle=True)

    print('Loading validation dataset...')
    valid_dataset = task_dataset(args, args.valid_data)
    valid_loader = DataLoader(dataset=valid_dataset, batch_size=args.batch_size)

    print('Building model...')
    model = task_model(args)
    model = model.to(args.device)

    criterion = nn.SmoothL1Loss()
    optimizer = Adam(model.parameters(), lr=args.lr)

    print('Start training!')
    best_loss = 0
    writer = SummaryWriter(args.log_dir)
    for epoch in range(args.epochs):
        train_loss = 0.0
        train_correct = 0
        model.train()
        for x, y in train_loader:
            model.zero_grad()

            # Move the parameters to device given by argument
            x, y = x.to(args.device), y.to(args.device)
            pred = model(x)

            # Calculate loss of annotators' labeling
            loss = criterion(pred, y)

            # Update model weight using gradient descent
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        # Validation
        with torch.no_grad():
            valid_loss = 0
            model.eval()
            for x, y in valid_loader:
                x, y = x.to(args.device), y.to(args.device)
                pred = model(x)
                valid_loss += criterion(pred, y).item()
        print(
            f'Epoch: {epoch + 1} | '
            f'Train Loss: {train_loss} | '
            f'Validation Loss: {valid_loss}'
        )

        # Save tensorboard log
        if epoch % args.log_interval == 0:
            writer.add_scalar('train_loss', train_loss, epoch)
            writer.add_scalar('valid_loss', valid_loss, epoch)

        # Save the model with highest accuracy on validation set
        if best_loss == 0 or best_loss > valid_loss:
            best_loss = valid_loss
            checkpoint_dir = Path(args.save_dir)
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            torch.save({
                'model': model.state_dict()
            }, checkpoint_dir / 'best_model.pth')

            with open(checkpoint_dir / 'args.json', 'w') as f:
                json.dump(args.__dict__, f, indent=2)
