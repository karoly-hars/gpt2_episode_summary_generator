import torch


def make_train_state(save_path, early_stopping_patience):
    """Create training state dictionary to keep track of improvements during training."""
    return {'stop_early': False,
            'early_stopping_step': 0,
            'early_stopping_patience': early_stopping_patience,
            'steps': 0,
            'min_val_loss': float('Inf'),
            'train_loss': [],
            'val_loss': [],
            'save_path': save_path}


def update_train_state(model, train_state, steps, train_loss, val_loss):
    """
    Update training state:
    - update losses
    - save the model in case of an improvement
    - check for early stopping
    - return update training state

    :param model: Model to train
    :param train_state: A dictionary representing the training state values
    :param steps: Training steps so far
    :param train_loss: Current training loss
    :param val_loss: Current validation loss
    :return: A new train_state
    """
    # update train state
    train_state['steps'] = steps
    train_state['train_loss'].append(train_loss)
    train_state['val_loss'].append(val_loss)
    loss_t = train_state['val_loss'][-1]

    # If loss increased
    if loss_t >= train_state['min_val_loss']:
        # Update step
        train_state['early_stopping_step'] += 1

    # Loss decreased
    else:
        # Save the best model
        train_state['min_val_loss'] = loss_t
        torch.save(model.state_dict(), train_state['save_path'])

        # Reset early stopping step
        train_state['early_stopping_step'] = 0

    # Update early stopping
    train_state['stop_early'] = train_state['early_stopping_step'] >= train_state['early_stopping_patience']

    return train_state
