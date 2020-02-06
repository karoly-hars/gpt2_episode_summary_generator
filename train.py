import argparse
import torch
from torch.utils.data import DataLoader
from pytorch_transformers import GPT2LMHeadModel, AdamW, WarmupLinearSchedule
from utils.data import EpisodeSummaryTokenizer, create_datasets_from_jsons
from utils.gen_utils import set_random_seeds, generate_sequence


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
    - return the updated training state

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


def initialize_training(args, device):
    """Initialize the tokenizer, the data loaders, the model and other components for the optimization process."""
    # Create tokenizer, datasets and loaders
    tokenizer = EpisodeSummaryTokenizer.from_pretrained(
        args.gpt2_size, max_num_words=args.max_num_words, size_variance_handling=args.size_var_handling
    )
    train_dataset, val_dataset = create_datasets_from_jsons(args.json_paths, tokenizer, args.val_split)

    dataloaders = {
        'train': DataLoader(train_dataset,
                            shuffle=True,
                            batch_size=args.batch_size,
                            collate_fn=tokenizer.pad_batch_to_same_size),
        'val': DataLoader(val_dataset,
                          shuffle=False,
                          batch_size=args.batch_size,
                          collate_fn=tokenizer.pad_batch_to_same_size)
    }

    # Load pre-trained network weights
    model = GPT2LMHeadModel.from_pretrained(args.gpt2_size)
    model = model.to(device)

    # Prepare optimizer and scheduler
    no_decay = ['bias', 'LayerNorm.weight']  # no decay for biases and layer norm
    optimizer_grouped_parameters = [
        {
            'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            'weight_decay': args.weight_decay
        },
        {
            'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            'weight_decay': 0.0
        }
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    scheduler = WarmupLinearSchedule(optimizer, warmup_steps=0, t_total=args.max_steps)
    model.zero_grad()

    train_state = make_train_state(save_path=args.model_save_path, early_stopping_patience=args.early_stopping_patience)

    return tokenizer, dataloaders, model, optimizer, scheduler, train_state


def forward_batch(model, batch, device):
    """Run a batch of data through a network/model."""
    inputs, labels = (batch, batch)
    inputs, labels = inputs.to(device), labels.to(device)

    outputs = model(inputs, labels=labels)

    return outputs[:2]


def run_training(args):
    """Run training process."""
    # Set seed
    set_random_seeds(args.random_seed)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print('Device: {}'.format(str(device)))

    # Initialize training
    tokenizer, dataloaders, model, optimizer, scheduler, train_state = initialize_training(args, device)

    # Run training process
    steps = 0
    model.train()
    print('\nRunning training:')

    while steps < args.max_steps and not train_state['stop_early']:
        model.train()

        running_train_loss = 0
        num_train_samples = 0
        running_val_loss = 0
        num_val_samples = 0

        for train_batch in dataloaders['train']:
            optimizer.zero_grad()

            loss, logits = forward_batch(model, train_batch, device)

            loss.backward()
            optimizer.step()
            scheduler.step()
            model.zero_grad()

            running_train_loss += loss.item()
            num_train_samples += train_batch.size()[0]

            steps += 1

            # Checkpoint
            if steps > 0 and steps % args.checkpoint_steps == 0:
                model.eval()

                for val_batch in dataloaders['val']:
                    loss, logits = forward_batch(model, val_batch, device)

                    running_val_loss += loss.item()
                    num_val_samples += val_batch.size()[0]

                train_state = update_train_state(model, train_state, steps,
                                                 running_train_loss / num_train_samples,
                                                 running_val_loss / num_val_samples)

                print('\n============== {} / {} =============='.format(steps, args.max_steps))
                print('train loss: {:.4f} | val loss: {:.4f}'.format(train_state['train_loss'][-1],
                                                                     train_state['val_loss'][-1]))
                # Generate some samples
                generated = generate_sequence(
                    model, tokenizer,
                    max_length=args.max_gen_len,
                    num_samples=args.num_samples,
                    top_k=args.sampling_top_k,
                    device=device
                )
                print('-' * 41)
                print(*generated, sep='\n')
                print('-' * 41)

                # Check for early stopping
                if train_state['stop_early']:
                    print('\nTraining finished with early stopping.')
                    print('best loss: {:.4f}'.format(train_state['min_val_loss']))
                    break

                # Reset sums and set model back to train
                running_train_loss = 0
                num_train_samples = 0
                running_val_loss = 0
                num_val_samples = 0
                model.train()


def get_arguments():
    """Collect command line arguments."""
    parser = argparse.ArgumentParser(
        description='GPT-2 model training for text generation.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    # Args related to the data
    parser.add_argument('-s', '--random_seed', type=int, required=False, default=0, help='Random seed.')
    parser.add_argument('-v', '--val_split', type=float, required=False, default=0.1,
                        help='Ratio of the validation subset size compared to all available data.')
    parser.add_argument('-m', '--max_num_words', type=int, required=False, default=80,
                        help='Maximum number of words per summary in the training set.')
    parser.add_argument('-sv', '--size_var_handling', type=str, required=False,
                        default='chop_at_sentence_end', choices=['chop_at_sentence_end', 'chop', 'ignore'],
                        help='Describes how to handle training sequences with different lengths. Options:'
                             ' -"chop_at_sentence_end": Chop long texts to make sure '
                             'that they contain <= words than max_num_words, but only chop at the end of a sentence.'
                             ' If that is not possible, drop data instance, and do not include it in the dataset.'
                             ' -"chop": Chop long texts to make sure that they contain <= words than max_num_words. '
                             'It is okay to chop after any word.'
                             ' -"ignore": Ignore size variance and tokenize all text without chopping.'
                             'In this case, max_num_words has no effect.')
    parser.add_argument('-j', '--json_paths', nargs='*', required=False,
                        default=['wiki_episode_summaries.json', 'imdb_episode_summaries.json'],
                        help='Path to the JSON files which contain the episode data (the outputs of the spiders).')

    # Training and optimization args
    parser.add_argument('-b', '--batch_size', type=int, required=False, default=8, help='Batch size.')
    parser.add_argument('-w', '--weight_decay', type=float, required=False, default=0.01, help='Weight decay.')
    parser.add_argument('-lr', '--learning_rate', type=float, required=False, default=5e-5,
                        help='Initial learning rate.')
    parser.add_argument('-a', '--adam_epsilon', type=float, required=False, default=1e-8,
                        help='Epsilon param of the Adam optimizer.')
    parser.add_argument('-ms', '--max_steps', type=int, required=False, default=4000,
                        help='Maximum number of training steps.')
    parser.add_argument('-cs', '--checkpoint_steps', type=int, required=False, default=50,
                        help='Checkpoint frequency during the training process.')
    parser.add_argument('-g', '--gpt2_size', type=str, required=False, default='gpt2',
                        choices=['gpt2', 'gpt2-medium', 'gpt2-large'],
                        help='Which GPT-2 architecture to use from pytorch-transformers.')
    parser.add_argument('-e', '--early_stopping_patience', type=int, required=False, default=3,
                        help='Patience before initiating early stopping.')
    parser.add_argument('-mp', '--model_save_path', type=str, required=False, default='ep_summary_gen_model.pth',
                        help='Save path for the trained model or checkpoints during training.')

    # sampling args
    parser.add_argument('-ns', '--num_samples', type=int, required=False, default=8,
                        help='Number of samples generated and displayed at every checkpoint.')
    parser.add_argument('-mg', '--max_gen_len', type=int, required=False, default=135,
                        help='Max length of the generated samples.')
    parser.add_argument('-tk', '--sampling_top_k', type=int, required=False, default=20,
                        help='The number of highest probability vocabulary tokens to keep during top-k-filtering '
                             'in the sample generation. Should be between 1 and inf.')

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = get_arguments()
    run_training(args)
