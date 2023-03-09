import os
import argparse
import torch
import torchvision.transforms as transforms
import pandas as pd
import time
import ray
import yaml
import asyncio
import time
from itertools import combinations
import datetime
import numpy as np


from model.Model import MSTS
from src.datasets import SmilesDataset, PNGSmileDataset
from src.config import input_data_dir, base_file_name, test_dir
from utils import logger, make_directory, load_reversed_token_map, smiles_name_print, str2bool, convert_smiles


def main():
    start_time = time.time()

    #smiles_name_print()


    parser = argparse.ArgumentParser()

    parser.add_argument('--work_type', type=str, default='train', help="choose work type 'train' or 'test'")
    parser.add_argument('--dataset', type=str, default='image', help='choose which dataset, image or hdf5')
    parser.add_argument('--encoder_type', type=str, default='efficientnetB2',
                        help="choose encoder model type 'efficientnetB2', wide_res', 'res', and 'resnext' ")
    parser.add_argument('--seed', type=int, default=1, help="choose seed number")
    parser.add_argument('--tf_encoder', type=int, default=0, help="the number of transformer layers")
    parser.add_argument('--tf_decoder', type=int, default=0, help="the number of transformer decoder layers")
    parser.add_argument('--decode_length', type=int, default=100, help='length of decoded SMILES sequence')
    parser.add_argument('--emb_dim', type=int, default=512, help='dimension of word embeddings')
    parser.add_argument('--attention_dim', type=int, default=512, help='dimension of attention linear layers')
    parser.add_argument('--decoder_dim', type=int, default=512, help='dimension of decoder RNN')
    parser.add_argument('--dropout', type=float, default=0.5, help='drop out rate')
    parser.add_argument('--device', type=str, default='cuda', help='sets device for model and PyTorch tensors')
    parser.add_argument('--gpu_non_block', type=str2bool, default=True, help='GPU non blocking flag')
    parser.add_argument('--fp16', type=str2bool, default=True, help='Use half-precision/mixed precision training')
    parser.add_argument('--cudnn_benchmark', type=str2bool, default=True,
                        help='set to true only if inputs to model are fixed size; otherwise lot of computational overhead')

    parser.add_argument('--epochs', type=int, default=60, help='number of epochs to train for')
    parser.add_argument('--batch_size', type=int, default=256, help='batch size')
    parser.add_argument('--checkpointing_cnn', type=int, default=0, help='Checkpoint  the cnn to save memory')
    parser.add_argument('--workers', type=int, default=8, help='for data-loading; right now, only 1 works with h5py')
    parser.add_argument('--encoder_lr', type=float, default=1e-4, help='learning rate for encoder if fine-tuning')
    parser.add_argument('--decoder_lr', type=float, default=4e-4, help='learning rate for decoder')
    parser.add_argument('--grad_clip', type=float, default=5., help='clip gradients at an absolute value of')
    parser.add_argument('--fine_tune_encoder', type=str2bool, default=True, help='fine-tune encoder')

    parser.add_argument('--model_save_path', type=str, default='graph_save', help='model save path')
    parser.add_argument('--model_load_path', type=str, default=None, help='model load path')
    parser.add_argument('--model_load_num', type=int, default=None, help='epoch number of saved model')
    parser.add_argument('--test_file_path', type=str, default=test_dir, help='test file path')
    parser.add_argument('--grayscale', type=str2bool, default=True, help='gray scale images ')


    config = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    config.device = 'cpu' if device == 'cpu' else config.device
    print('torch work_type:', config.device)
    print("batch size:", config.batch_size)


    # Custom dataloaders
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    if config.work_type != 'ensemble_test':
        model = MSTS(config) #create one instance of the model

    if config.work_type == 'train':
        # from utils import make_directory
        # make_directory(config.model_save_path)
        if not (config.model_load_path == None) and not (config.model_load_num == None):
            model.model_load()
            print('model loaded')
        else:
            print('start from initial')

        if config.dataset == 'hdf5':
            train_loader = torch.utils.data.DataLoader(
                SmilesDataset(input_data_dir, base_file_name, 'TRAIN',
                              transform=transforms.Compose([normalize])),
                batch_size=config.batch_size, shuffle=True,
                num_workers=config.workers, pin_memory=True)

            val_loader = torch.utils.data.DataLoader(
                SmilesDataset(input_data_dir, base_file_name, 'VAL',
                              transform=transforms.Compose([normalize])),
                batch_size=config.batch_size, shuffle=True,
                num_workers=config.workers, pin_memory=True)

        elif config.dataset == 'image':
            train_loader = torch.utils.data.DataLoader(
                PNGSmileDataset(input_data_dir, base_file_name, 'TRAIN',
                                transform=transforms.Compose([normalize]),
                                grayscale=config.grayscale),
                batch_size=config.batch_size, shuffle=True,
                num_workers=config.workers, pin_memory=True)

            val_loader = torch.utils.data.DataLoader(
                PNGSmileDataset(input_data_dir, base_file_name, 'VAL',
                                transform=transforms.Compose([normalize]),
                                grayscale=config.grayscale),
                batch_size=config.batch_size, shuffle=True,
                num_workers=config.workers, pin_memory=True)

        else:
            print("Incorrect input file type")

        log_index = ['t_loss', 't_accr', 'v_loss', 'v_accr']

        logger(log_index)
        # logger(log_index, data_dir)
        for itr in range(config.epochs):
            print('epoch:', itr)

            t_l, t_a = model.train(train_loader)
            v_l, v_a = model.validation(val_loader)
            print('t_loss: %s     t_accr: %s     v_loss: %s    v_accr: %s' % (t_l, t_a, v_l, v_a))
            model.model_save(save_num=itr)
            logger([t_l, t_a, v_l, v_a])

    elif config.work_type == 'single_test':
        from src.config import sample_submission_dir, generate_submission_dir, reversed_token_map_dir

        if not config.test_file_path == None:

            submission = pd.read_csv(sample_submission_dir)
            reversed_token_map = load_reversed_token_map(reversed_token_map_dir)
            data_list = os.listdir(config.test_file_path)

            #transform = transforms.Compose([normalize])
            if config.grayscale is not False:
                transform = transforms.Compose([transforms.Compose([normalize]),
                                                transforms.Grayscale(3)])

            else:
                transform = transforms.Compose([normalize])

            model.model_load()
            print('model loaded')
            submission = model.model_test(submission, data_list, reversed_token_map, transform)
            submission.to_csv(generate_submission_dir, index=False)

        else:
            print('the test file path is none')

    elif config.work_type == 'ensemble_test':

        #for experiments esamble test, easy for caculating tanimoto
        #for application esamble prediction, evaluate and generate more information from platfrom PubChem for users.
        from src.config import sample_submission_dir, generate_submission_dir
        #from src.config import sample_submission_dir, generate_submission_dir, reversed_token_map_dir
        from PIL import Image

        from rdkit import Chem
        from rdkit.DataStructs import FingerprintSimilarity as FPS
        from rdkit.Chem import MolFromSmiles,RDKFingerprint
        from utils import make_directory, decode_predicted_sequences
        ray.init()

        from copy import deepcopy

        def clone_config(config, p_config):
            new_config = deepcopy(config)

            new_config.emb_dim = p_config['emb_dim']
            new_config.attention_dim = p_config['attention_dim']
            new_config.decoder_dim = p_config['decoder_dim']
            new_config.encoder_type = p_config['encoder_type']
            new_config.tf_encoder = p_config['tf_encoder']
            new_config.tf_decoder = p_config['tf_decoder']
            # new_config.load_model_path = p_config['load_model_path']
            # new_config.load_model_num = p_config['load_model_num']
            new_config.model_load_path = p_config['load_model_path']
            new_config.model_load_num = p_config['load_model_num']
            new_config.reverse_token_map = p_config['reverse_token_map']
            return new_config

        # now we create models:
        with open('/org/temp/anon/data/model/model/prediction_models.yaml') as f:
            p_configs = yaml.load(f)

        model_configs = []
        models = []
        reverse_token_maps = []
        for key in p_configs:
            p_config = p_configs[key]
            print(p_config, key)
            new_config = clone_config(config, p_config)
            model_configs.append(new_config)

        for model_config in model_configs:
            model_ = MSTS(model_config)
            model_.model_load()
            print('model loaded')
            model_._encoder.eval()
            model_._decoder.eval()
            models.append(model_)

            reverse_token_map = load_reversed_token_map(model_config.reverse_token_map)
            reverse_token_maps.append(reverse_token_map)



        if not config.test_file_path == None:

            submission = pd.read_csv(sample_submission_dir)
            # reversed_token_map = load_reversed_token_map(reversed_token_map_dir)
            data_list = os.listdir(config.test_file_path)

            if config.grayscale is not False:
                transform = transforms.Compose([transforms.Compose([normalize]),
                                                transforms.Grayscale(3)])

            else:
                transform = transforms.Compose([normalize])

            conf_len = len(p_configs)  # configure length == number of model to use
            fault_counter = 0
            #sequence = None
            model_contribution = np.zeros(conf_len)

            def png_to_tensor(img: Image):
                """
                convert png format image to torch tensor with resizing and value rescaling
                :param img: .png file
                :return: tensor data of float type
                """
                img = img.resize((256,256))
                img = np.array(img)

                # what is the dimension of img?
                # (
                print(img.ndim)
                if img.ndim == 3:
                    img = np.moveaxis(img, 2, 0) # this function moves the final axis to the first
                    # it means that the img can be [256, 256, 3]  -> [3, 256, 256] #[N C H W] is right
                    # or [3, 256, 256] -> [256, 256, 3]] # [N H W C]
                else:
                    # now with only grayscale your image is [256, 256] ->
                    img = np.stack([img, img, img], 0)

                return torch.FloatTensor(img) / 255.

            def _decode(models, _input):
                outputs = list()
                for model in models:
                    encoded_imgs = model._encoder(_input.unsqueeze(0))
                    print("USING NEW PREDICTION CODE ...")
                    predictions = model._decoder(encoded_imgs, decode_lengths=model._decode_length, mode='generation')
                    outputs.append(predictions)

                return outputs

            def calculate_similarity(combination_of_smiles, combination_index):
                return {idx: FPS(comb[0], comb[1]) for comb, idx in zip(combination_of_smiles, combination_index)}

            for i, dat in enumerate(data_list):
                imgs = Image.open(config.test_file_path + dat)
                imgs = png_to_tensor(imgs)
                imgs = transform(imgs).cuda()

                # predict SMILES sequence form each predictors
                preds_raw = _decode(models, imgs)

                preds=[]
                for p, r_v_m in zip(preds_raw, reverse_token_maps):
                    # predicted sequence token value
                    SMILES_predicted_sequence = list(torch.argmax(p.detach().cpu(), -1).numpy())[0]
                    # converts prediction to readable format from sequence token value
                    decoded_sequences = decode_predicted_sequences(SMILES_predicted_sequence, r_v_m)
                    preds.append(decoded_sequences)
                del(preds_raw)
                print(preds)

                # fault check: whether the prediction satisfies the SMILES format or not
                ms = {}

                for idx, p in enumerate(preds):
                    m = MolFromSmiles(p)
                    if m != None:
                        ms.update({idx:m})

                if len(ms) == 0: # there is no decoded sequence that matches to SMILES format
                    print('decode fail')
                    fault_counter += 1
                    sequence = preds[0]

                elif len(ms) == 1: # there is only one decoded sequence that matches to SMILES format
                    sequence = preds[list(ms.keys())[0]]

                else: # there is more than two decoded sequence that matches to SMILES format
                    # result ensemble
                    ms_to_fingerprint = [RDKFingerprint(x) for x in ms.values()]
                    combination_of_smiles = list(combinations(ms_to_fingerprint, 2))
                    # [1 2 3 4 5
                    ms_to_index = [x for x in ms]
                    combination_index = list(combinations(ms_to_index, 2))

                    # calculate similarity score
                    smiles_dict = calculate_similarity(combination_of_smiles, combination_index)
                    # sort the pairs by similarity score
                    smiles_dict = sorted(smiles_dict.items(), key=(lambda x: x[1]), reverse=True)

                    if smiles_dict[0][1] == 1.0: # if a similar score is 1 we assume to those predictions are correct.
                        sequence = preds[smiles_dict[0][0][0]]
                    else:
                        score_board = np.zeros(conf_len)
                        for j, (idx, value) in enumerate(smiles_dict):
                            score_board[list(idx)] += conf_len-j

                        pick = int(np.argmax(score_board)) # choose the index that has the highest score
                        sequence = preds[pick]  # pick the decoded sequence
                        model_contribution[pick] += 1 # logging witch model used
                        #sequence = preds[np.argmax(score_board)]

                print('{} sequence:, {}'.format(i, sequence))
                # print('decode_time:', time.time() - start_time)

                submission.loc[submission['file_name'] == dat, 'SMILES'] = sequence
                del(preds)

            # loop.close()
            print('total fault:', fault_counter)
            print('model contribution:', model_contribution)

            # submission = model.ensemble_test(submission, data_list, reversed_token_map, transform)
            submission.to_csv(generate_submission_dir, index=False)

        else:
            print('the test file path is none')


    elif config.work_type == 'one_input_pred':
        #TODO shows all possible smiles
        #input one sample for application
        from src.config import reversed_token_map_dir
        #from .utils import convert_smiles
        #ray.init()
        image = '122.png'
        reversed_token_map = load_reversed_token_map(reversed_token_map_dir)

        if config.grayscale is not None:
            transform = transforms.Compose([transforms.Compose([normalize]),
                                            transforms.Grayscale(3)])

        else:
            transform = transforms.Compose([normalize])

        model.model_load()

        smiles = model.one_test(image, reversed_token_map, transform)
        print('Generated SMILES:', smiles)
        convert_smiles(smiles)



    else:
        print('incorrect work type received.')

    print('process time:', time.time() - start_time)


    #TODO add one_input test for tool
    # add generating other representations
    # not print model
    # add PubChem to evaluate.
    # elif one_sample_test():
    #     # predict fot image
    #     # read image
if __name__ == '__main__':
    main()
