import argparse
import torch
from utils import set_random_seeds, generate_sequence
from pytorch_transformers import GPT2Config, GPT2Tokenizer, GPT2LMHeadModel


def generate_samples(args):
    # Set seed
    set_random_seeds(args.random_seed)

    # Initialize training
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Load pre-trained network weights
    print("Loading pre-trained model...")
    config = GPT2Config.from_pretrained(args.gpt2_version)
    model = GPT2LMHeadModel(config)
    model.load_state_dict(torch.load(args.model_load_path))
    model = model.to(device)
    model.eval()
    print("Done.\n")

    # Create tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained(args.gpt2_version)

    # Generate some samples
    print("Generating...")
    generated = generate_sequence(model, tokenizer,
                                  max_length=args.max_gen_len,
                                  num_samples=args.num_samples,
                                  top_k=args.sampling_top_k,
                                  device=device)
    print("Generated samples:")
    print("-" * 41)
    print(*generated, sep="\n\n")
    print("-" * 41)


def get_arguments():
    """Collect command line arguments."""
    parser = argparse.ArgumentParser(
        description="Generate TV show episode summaries with a GPT-2 model pretrained for this purpose.",
        formatter_class=lambda prog: argparse.RawTextHelpFormatter(prog, width=75)
    )

    parser.add_argument("-s", "--random_seed", type=int, required=False, default=99, help="Random seed.")
    parser.add_argument("-g", "--gpt2_version", type=str, required=False, default="gpt2-medium",
                        choices=["gpt2, gpt2-medium", "gpt2-large"],
                        help="Which GPT2 version to use from pytorch-transformers.")
    parser.add_argument("-sp", "--model_load_path", type=str, required=False, default="ep_summary_gen_model.pth",
                        help="Save path for the trained model or checkpoints during training.")
    parser.add_argument("-ns", "--num_samples", type=int, required=False, default=8,
                        help="Number of samples generated and displayed at every checkpoint.")
    parser.add_argument("-mg", "--max_gen_len", type=int, required=False, default=192,
                        help="Max length of the generated samples.")
    parser.add_argument("-tk", "--sampling_top_k", type=int, required=False, default=20,
                        help="The number of highest probability vocabulary tokens to keep during top-k-filtering "
                             "in the sample generation. Should be between 1 and inf.")

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = get_arguments()
    generate_samples(args)
