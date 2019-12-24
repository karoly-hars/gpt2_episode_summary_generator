import argparse
import torch
from dataset import EpisodeSummaryTokenizer, create_datasets_from_jsons
from torch.utils.data import DataLoader


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


def run_training(args):
    # create tokenizer, datasets and loaders
    tokenizer = EpisodeSummaryTokenizer.from_pretrained(
        args.gpt2_version, max_num_words=args.max_num_words, size_variance_handling=args.size_var_handling
    )
    train_dataset, val_dataset = create_datasets_from_jsons(args.json_paths, tokenizer, args.val_split)

    train_dataloader = DataLoader(
        train_dataset, shuffle=True, batch_size=args.batch_size, collate_fn=tokenizer.pad_batch_to_same_size
    )
    val_dataloader = DataLoader(val_dataset, batch_size=1, collate_fn=tokenizer.pad_batch_to_same_size)


def get_arguments():
    """Collect command line arguments."""
    parser = argparse.ArgumentParser(
        description="GPT-2 model traning for text generation.",
        formatter_class=lambda prog: argparse.RawTextHelpFormatter(prog, width=75)
    )
    # args related to the data
    parser.add_argument("-s", "--seed", type=int, required=False, default=0, help="Random seed.")
    parser.add_argument("-v", "--val_split", type=float, required=False, default=0.1,
                        help="Ratio of the validation subset size compared to all available data.")
    parser.add_argument("-m", "--max_num_words", type=int, required=False, default=96,
                        help="Maximum number of words per summary in the training set.")
    parser.add_argument("-sv", "--size_var_handling", type=str, required=False,
                        default="chop_at_sentence_end", choices=["chop_at_sentence_end", "chop", "ignore"],
                        help="Describes how to handle training sequences with different lengths.\nOptions:"
                             "\n-'chop_at_sentence_end': Chop long texts to make sure "
                             "that they contain <= words than max_num_words, but only chop at the end of a sentence."
                             " If that is not possible, return None instead of vectorizing the text."
                             "\n-'chop': Chop long texts to make sure that they contain <= words than max_num_words. "
                             "It is okay to chop after any word."
                             "\n-'ignore': Ignore size variance and tokenize all text without chopping."
                             "In this case, max_num_words has no effect.")
    parser.add_argument('-j', '--json_paths', nargs='*', required=False,
                        default=["wiki_episode_summaries.json", "imdb_episode_summaries.json"],
                        help="Path to the JSON files which contain the episode data (the outputs of the spiders).")

    # training and optimization args
    parser.add_argument("-b", "--batch_size", type=int, required=False, default=4, help="Batch size.")
    parser.add_argument("-w", "--weight_decay", type=float, required=False, default=0.01, help="Weight decay.")
    parser.add_argument("-lr", "--learning_rate", type=float, required=False, default=5e-5,
                        help="Initial learning rate.")
    parser.add_argument("-a", "--adam_epsilon", type=float, required=False, default=1e-8,
                        help="Epsilon param of the Adam optimizer.")
    parser.add_argument("-ms", "--max_steps", type=int, required=False, default=1e5,
                        help="Maximum number of training steps.")
    parser.add_argument("-cs", "--checkpoint_steps", type=int, required=False, default=100,
                        help="Checkpoint frequency during the training process.")
    parser.add_argument("-g", "--gpt2_version", type=str, required=False, default="gpt2-medium",
                        choices=["gpt2, gpt2-medium", "gpt2-large"],
                        help="Which GPT2 version to use from pytorch-transformers.")
    parser.add_argument("-e", "--early_stopping_patience", type=int, required=False, default=3,
                        help="Patience before initiating early stopping.")
    parser.add_argument("-sp", "--model_save_path", type=str, required=False, default="ep_summary_gen_model.pth",
                        help="Save path for the trained model or checkpoints during training.")

    # sampling args
    parser.add_argument("-ns", "--num_samples", type=int, required=False, default=5,
                        help="Number of samples generated and displayed at every checkpoint.")
    parser.add_argument("-mg", "--max_gen_length", type=int, required=False, default=192,
                        help="Max length of the generated samples.")
    parser.add_argument("-tk", "--sampling_top_k", type=int, required=False, default=20,
                        help="The number of highest probability vocabulary tokens to keep during top-k-filtering "
                             "in the sample generation. Should be between 1 and inf.")

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = get_arguments()
    run_training(args)
