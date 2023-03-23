import argparse
from flask import Flask, jsonify, request
import torchvision.transforms as transforms
from PIL import Image

from utils import load_reversed_token_map, str2bool
from model.Model import MSTS


app = Flask(__name__)

# utils
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
test_dir = "./data/test"

# config

parser = argparse.ArgumentParser()

parser.add_argument(
    "--work_type",
    type=str,
    default="one_input_pred",
    help="choose work type which test",
)
parser.add_argument(
    "--encoder_type",
    type=str,
    default="efficientnetB0",
    help="choose encoder model type 'efficientnetB2', wide_res', 'res', and 'resnext' ",
)
parser.add_argument("--seed", type=int, default=1, help="choose seed number")
parser.add_argument(
    "--tf_encoder", type=int, default=6, help="the number of transformer layers"
)
parser.add_argument(
    "--tf_decoder", type=int, default=6, help="the number of transformer decoder layers"
)
parser.add_argument(
    "--decode_length", type=int, default=100, help="length of decoded SMILES sequence"
)
parser.add_argument(
    "--emb_dim", type=int, default=512, help="dimension of word embeddings"
)
parser.add_argument(
    "--attention_dim",
    type=int,
    default=512,
    help="dimension of attention linear layers",
)
parser.add_argument(
    "--decoder_dim", type=int, default=512, help="dimension of decoder RNN"
)
parser.add_argument("--dropout", type=float, default=0.5, help="drop out rate")
parser.add_argument(
    "--device",
    type=str,
    default="cuda",
    help="sets device for model and PyTorch tensors",
)
parser.add_argument(
    "--gpu_non_block", type=str2bool, default=True, help="GPU non blocking flag"
)
parser.add_argument(
    "--fp16",
    type=str2bool,
    default=True,
    help="Use half-precision/mixed precision training",
)
parser.add_argument(
    "--cudnn_benchmark",
    type=str2bool,
    default=True,
    help="set to true only if inputs to model are fixed size; otherwise lot of computational overhead",
)


parser.add_argument(
    "--epochs", type=int, default=60, help="number of epochs to train for"
)
parser.add_argument("--batch_size", type=int, default=256, help="batch size")
parser.add_argument(
    "--checkpointing_cnn",
    type=int,
    default=0,
    help="Checkpoint  the cnn to save memory",
)
parser.add_argument(
    "--workers",
    type=int,
    default=8,
    help="for data-loading; right now, only 1 works with h5py",
)
parser.add_argument(
    "--encoder_lr",
    type=float,
    default=1e-4,
    help="learning rate for encoder if fine-tuning",
)
parser.add_argument(
    "--decoder_lr", type=float, default=4e-4, help="learning rate for decoder"
)
parser.add_argument(
    "--grad_clip",
    type=float,
    default=5.0,
    help="clip gradients at an absolute value of",
)
parser.add_argument(
    "--fine_tune_encoder", type=str2bool, default=True, help="fine-tune encoder"
)

parser.add_argument(
    "--model_save_path", type=str, default="graph_save", help="model save path"
)
parser.add_argument(
    "--model_load_path", type=str, default="./src/model_path", help="model load path"
)
parser.add_argument(
    "--model_load_num", type=int, default=11, help="epoch number of saved model"
)
parser.add_argument(
    "--test_file_path", type=str, default=test_dir, help="test file path"
)
parser.add_argument(
    "--grayscale", type=str2bool, default=True, help="gray scale images"
)
parser.add_argument(
    "--reversed_token_map_dir)", type=str, default=True, help="gray scale images"
)


config = parser.parse_args()


if config.grayscale:
    transform = transforms.Compose(
        [transforms.Compose([normalize]), transforms.Grayscale(3)]
    )
else:
    transform = transforms.Compose([normalize])

# load model
model = MSTS(config)
reversed_token_map_dir = "./model/reversed_token_map/REVERSED_TOKENMAP_5.json"
reversed_token_map = load_reversed_token_map(reversed_token_map_dir)
model.model_load()


@app.route("/health")
def health():
    return "OK"


@app.route("/predict", methods=["POST"])
def predict():
    if request.method == "POST":
        # get posted file
        img = request.files["file"]
        # predict
        predicted_smiles = model.one_test(
            Image.open(img), reversed_token_map, transform
        )
        return jsonify({"smiles": predicted_smiles})


if __name__ == "__main__":
    app.run()
