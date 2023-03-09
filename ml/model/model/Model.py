import datetime

import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence
from utils import logger

from PIL import Image

from rdkit import Chem
from rdkit.DataStructs import FingerprintSimilarity as FPS
from rdkit.Chem import MolFromSmiles,RDKFingerprint

from model.Network import Encoder, DecoderWithAttention
#from .Predictor import Predict
from utils import make_directory, decode_predicted_sequences

import ray
import random
import numpy as np
import yaml
import asyncio
import time
from itertools import combinations
import datetime



class MSTS:
    """
    Molecule Structure To SMILES
    this class has big three feature that 'train', 'validation', and 'test'
    """
    def __init__(self, config):

        self._work_type = config.work_type
        self._seed = config.seed

        self._vocab_size = 70
        self._decode_length = config.decode_length
        self._emb_dim = config.emb_dim
        self._attention_dim = config.attention_dim
        self._decoder_dim = config.decoder_dim
        self._dropout = config.dropout
        self._device = config.device
        self._gpu_non_block = config.gpu_non_block
        self.tf_encoder = config.tf_encoder
        self.tf_decoder = config.tf_decoder
        self._cudnn_benchmark = config.cudnn_benchmark

        self._epochs = config.epochs
        self._batch_size = config.batch_size
        self._workers = config.workers
        self._encoder_lr = config.encoder_lr
        self._decoder_lr = config.decoder_lr
        self._grad_clip = config.grad_clip
        self._fine_tune_encoder = config.fine_tune_encoder
        self._checkpointing_cnn = config.checkpointing_cnn

        self._model_save_path = config.model_save_path
        self._model_load_path = config.model_load_path
        self._model_load_num = config.model_load_num
        self._test_file_path = config.test_file_path

        self._model_name = self._model_name_maker()

        self._seed_everything(self._seed)
        self.fp16 = config.fp16

        if self.tf_decoder > 0:
            from .Network import TransformerDecoder
            print("Create Transformer Decoder")
            self._decoder = TransformerDecoder(self._emb_dim, self._decoder_dim, self._vocab_size,
                                               self._device, dropout=self._dropout, n_layers=self.tf_decoder)
        else:
            self._decoder = DecoderWithAttention(attention_dim=self._attention_dim,
                                                 embed_dim=self._emb_dim,
                                                 decoder_dim=self._decoder_dim,
                                                 vocab_size=self._vocab_size,
                                                 dropout=self._dropout,
                                                 device=self._device)

        # define different decoder by work type
        if self._work_type == 'train':
            make_directory(self._model_save_path + '/' + self._model_name)
            #self._decoder.to(self._device, non_blocking=self._gpu_non_block)
            self._decoder_optimizer = torch.optim.Adam(params=filter(lambda p: p.requires_grad,
                                                                     self._decoder.parameters()),
                                                       lr=self._decoder_lr)
        # elif self._work_type == 'single_test':
        #     # self._decoder = PredictiveDecoder(attention_dim=self._attention_dim,
        #     #                                   embed_dim=self._emb_dim,
        #     #                                   decoder_dim=self._decoder_dim,
        #     #                                   vocab_size=self._vocab_size,
        #     #
        #     #                                   device=self._device)
        #     self._decoder.to(self._device, non_blocking=self._gpu_non_block)

        self._decoder.to(self._device, non_blocking=self._gpu_non_block)

        self._encoder = Encoder(model_type=config.encoder_type,
                                tf_encoder=config.tf_encoder,
                                embed_dim=self._emb_dim,
                                checkpointing_cnn=self._checkpointing_cnn)
        self._encoder.to(self._device, non_blocking=self._gpu_non_block)
        self._encoder.fine_tune(self._fine_tune_encoder)
        print(self._encoder) #print model structure
        print(self._decoder) #print model structure
        #print(len(self._encoder.resnet[0]))
        #
        self._encoder_optimizer = torch.optim.Adam(self._encoder.parameters(),
                                                   lr=self._encoder_lr) if self._fine_tune_encoder else None
        if torch.cuda.device_count() > 1 and self._device != 'cpu':
            print("Let's use", torch.cuda.device_count(), "GPUs!")
            self._encoder = nn.DataParallel(self._encoder) # model will become model.module -> all keys will have module in there
            # solution: use 2 gpus in testing or convert the name back to no module.
        self._criterion = nn.CrossEntropyLoss().to(self._device, non_blocking=self._gpu_non_block)

        if self.fp16:
            self.grad_scaler = torch.cuda.amp.GradScaler()
        else:
            self.grad_scaler = None

    def _clip_gradient(self, optimizer, grad_clip):
        for group in optimizer.param_groups:
            for param in group['params']:
                if param.grad is not None:
                    param.grad.data.clamp_(-grad_clip, grad_clip)

    def train(self, train_loader):
        self._encoder.train()
        self._decoder.train()

        mean_loss = 0
        mean_accuracy = 0

        total_image_processed = 0
        start_time = time.time()  # record the starting moment when we start training
        log_step = 100  # we will print out the speed every 200 training steps
        total_batches = len(train_loader)

        #measuring training speed
        # The unit that we can use to measure speed is images/second or tokens/second
        for i, (imgs, sequence, sequence_lens) in enumerate(train_loader):
            imgs = imgs.to(self._device)
            sequence = sequence.to(self._device)
            sequence_lens = sequence_lens.to(self._device)

            # the forward pass starts from here
            # normally the data  type of imgs and sequence is fp32 (32 bit for float representation)
            # for modern GPUS we can represent float with fp16 (16 bit for float representation)
            # so we can have 2x more meory and computation is theoretically 2x faster
            with torch.cuda.amp.autocast(enabled=self.fp16):
                imgs = self._encoder(imgs)
                predictions, caps_sorted, decode_lengths, alphas, sort_ind = self._decoder(imgs, sequence, sequence_lens)

                targets = caps_sorted[:, 1:]

                # Calculate accuracy
                accr = self._accuracy_calcluator(predictions.detach().cpu().numpy(),
                                                 targets.detach().cpu().numpy())
                mean_accuracy = mean_accuracy + (accr - mean_accuracy) / (i + 1)

                predictions = pack_padded_sequence(predictions, decode_lengths, batch_first=True).data
                targets = pack_padded_sequence(targets, decode_lengths, batch_first=True).data

                # Calculate loss
                loss = self._criterion(predictions, targets)
                mean_loss = mean_loss + (loss.detach().item() - mean_loss) / (i + 1)

            # Back prop.
            self._decoder_optimizer.zero_grad()
            self._encoder_optimizer.zero_grad()

            if self.fp16:
                self.grad_scaler.scale(loss).backward()
            else:
                loss.backward()

            # Clip gradients
            if self._grad_clip is not None:
                self._clip_gradient(self._decoder_optimizer, self._grad_clip)
                self._clip_gradient(self._encoder_optimizer, self._grad_clip)

            # Update weights
            if self.fp16:
                self.grad_scaler.step(self._decoder_optimizer)
                self.grad_scaler.step(self._encoder_optimizer)
                self.grad_scaler.update()
            else:
                self._decoder_optimizer.step()
                self._encoder_optimizer.step()

            # update the number of images procssed
            total_image_processed = total_image_processed  + imgs.size(0)

            if (i + 1) % log_step == 0:
                end_time = time.time()
                elapse = end_time - start_time
                img_per_sec = total_image_processed / elapse

                training_time = str(datetime.timedelta(seconds=int(elapse)))
                print("Iteration %d/%d ; mean_loss %.6f ; mean_accuracy %.5f ; img_per_sec %6.2f ; time: %s"
                      % (i, total_batches, mean_loss, mean_accuracy, img_per_sec, training_time), flush=True)
                #logger([mean_loss, mean_accuracy, img_per_sec, training_time])

        # after finishing training the epoch
        end_time = time.time()
        elapse = end_time - start_time
        training_time = str(datetime.timedelta(seconds=int(elapse)))
        print("mean_loss %.6f ; mean_accuracy %.5f ; img_per_sec %6.2f; time: %s"
              % (mean_loss, mean_accuracy, img_per_sec, training_time))
        #logger([mean_loss, mean_accuracy, img_per_sec, training_time])

        return mean_loss, mean_accuracy

    def validation(self, val_loader):
        self._encoder.eval()
        self._decoder.eval()

        mean_loss = 0
        mean_accuracy = 0

        with torch.no_grad():

            for i, (imgs, sequence, sequence_lens) in enumerate(val_loader):
                imgs = imgs.to(self._device)
                sequence = sequence.to(self._device)
                sequence_lens = sequence_lens.to(self._device)

                imgs = self._encoder(imgs)
                predictions, caps_sorted, decode_lengths, _, _ = self._decoder(imgs, sequence, sequence_lens)
                targets = caps_sorted[:, 1:]

                accr = self._accuracy_calcluator(predictions.detach().cpu().numpy(),
                                                 targets.detach().cpu().numpy())

                mean_accuracy = mean_accuracy + (accr - mean_accuracy) / (i + 1)

                predictions = pack_padded_sequence(predictions, decode_lengths, batch_first=True).data
                targets = pack_padded_sequence(targets, decode_lengths, batch_first=True).data

                loss = self._criterion(predictions, targets)
                mean_loss = mean_loss + (loss.detach().item() - mean_loss) / (i + 1)
                del (loss, predictions, caps_sorted, decode_lengths, targets)

        return mean_loss, mean_accuracy

    def model_test(self, submission, data_list, reversed_token_map, transform, labels=None):
        """
        single model test function
        :param submission: submission file
        :param data_list: list of test data path
        :param reversed_token_map: converts prediction to readable format
        :param transform: normalize function
        """
        self._encoder.eval()
        self._decoder.eval()

        fault_counter = 0
        for i, dat in enumerate(data_list):
            start_time = time.time()
            imgs = Image.open(self._test_file_path + dat)
            imgs = self.png_to_tensor(imgs)
            #print(imgs.ndim)
            imgs = transform(imgs).to(self._device)

            encoded_imgs = self._encoder(imgs.unsqueeze(0))
            print("USING NEW PREDICTION CODE ...")
            predictions = self._decoder(encoded_imgs, decode_lengths=self._decode_length, mode='generation')

            SMILES_predicted_sequence = list(torch.argmax(predictions.detach().cpu(), -1).numpy())[0]
            decoded_sequences = decode_predicted_sequences(SMILES_predicted_sequence, reversed_token_map)
            if self.is_smiles(decoded_sequences):
                fault_counter += 1

            print('{} sequence:, {}'.format(i, decoded_sequences))
            print('decode_time:', time.time() - start_time)


            if labels is not None:
                label = labels[i]
                SMILES_label = self.is_smiles(label)
                # compute the Tanimoto sim between decoded sequences and label

            submission.loc[submission['file_name']== dat, 'SMILES'] = decoded_sequences
            del (predictions)

        print('total fault:', fault_counter)
        return submission

    # def ensemble_test(self, submission, data_list, reversed_token_map, transform):
    #     """
    #     ensemble test function
    #     :param submission: submission file
    #     :param data_list: list of test data path
    #     :param reversed_token_map: converts prediction to readable format
    #     :param transform: normalize function
    #     """
    #     # load .yaml file that contains information about each model
    #     with open('/org/temp/anon/data/model/model/prediction_models.yaml') as f:
    #         p_configs = yaml.load(f)
    #
    #     predictors = []
    #     # inefficient and wrong
    #     # 1. model in Predict is outdated compared to model in Model.py
    #     # 2. the current "model" object already contains a model -> you create 1 extra model for nothing
    #     for conf in p_configs.values():
    #         predictors.append(Predict.remote(conf, self._device,
    #                                          self._gpu_non_block,
    #                                          self._decode_length, self._model_load_path))
    #     # 1. go to main.py and load yaml
    #     # 2. for each p_config : make a new config with values from p_config overwritten over the default values from main.py
    #     # 3. create a new model = MSTS(p_config)
    #
    #     loop = asyncio.get_event_loop()
    #     async def process_async_calculate_similarity(combination_of_smiles, combination_index):
    #         return {idx: await self.async_fps(comb[0], comb[1]) for comb, idx in zip(combination_of_smiles, combination_index)}
    #
    #     def process_calculate_similarity(smiles, index):
    #         pass
    #
    #     # for each model:
    #     # create a prediction (something similar to single_test)
    #     def ray_prediction(imgs):
    #         return ray.get([p.decode.remote(imgs) for p in predictors])
    #
    #
    #     conf_len = len(p_configs)  # configure length == number of model to use
    #     fault_counter = 0
    #     #sequence = None
    #     model_contribution = np.zeros(conf_len)
    #     for i, dat in enumerate(data_list):
    #         imgs = Image.open(self._test_file_path + dat)
    #         imgs = self.png_to_tensor(imgs)
    #         imgs = transform(imgs).pin_memory().cuda()
    #
    #         # predict SMILES sequence form each predictors
    #         preds_raw = ray_prediction(imgs)
    #
    #         preds=[]
    #         for p in preds_raw:
    #             # predicted sequence token value
    #             SMILES_predicted_sequence = list(torch.argmax(p.detach().cpu(), -1).numpy())[0]
    #             # converts prediction to readable format from sequence token value
    #             decoded_sequences = decode_predicted_sequences(SMILES_predicted_sequence, reversed_token_map)
    #             preds.append(decoded_sequences)
    #         del(preds_raw)
    #
    #         # fault check: whether the prediction satisfies the SMILES format or not
    #         ms = {}
    #         for idx, p in enumerate(preds):
    #             m = MolFromSmiles(p)
    #             if m != None:
    #                 ms.update({idx:m})
    #
    #         if len(ms) == 0: # there is no decoded sequence that matches to SMILES format
    #             print('decode fail')
    #             fault_counter += 1
    #             sequence = preds[0]
    #
    #         elif len(ms) == 1: # there is only one decoded sequence that matches to SMILES format
    #             sequence = preds[list(ms.keys())[0]]
    #
    #         else: # there is more than two decoded sequence that matches to SMILES format
    #             # result ensemble
    #             ms_to_fingerprint = [RDKFingerprint(x) for x in ms.values()]
    #             combination_of_smiles = list(combinations(ms_to_fingerprint, 2))
    #             # [1 2 3 4 5]
    #             # [[1, 2], [1,3 ], [1, 4], [1, 5], [2, 3] ... [4 5]]
    #             ms_to_index = [x for x in ms]
    #             combination_index = list(combinations(ms_to_index, 2))
    #
    #             # calculate similarity score
    #             smiles_dict = loop.run_until_complete(process_async_calculate_similarity(combination_of_smiles, combination_index))
    #
    #             # sort the pairs by similarity score
    #             smiles_dict = sorted(smiles_dict.items(), key=(lambda x: x[1]), reverse=True)
    #
    #             if smiles_dict[0][1] == 1.0: # if a similar score is 1 we assume to those predictions are correct.
    #                 sequence = preds[smiles_dict[0][0][0]]
    #             else:
    #                 score_board = np.zeros(conf_len)
    #                 for i, (idx, value) in enumerate(smiles_dict):
    #                     score_board[list(idx)] += conf_len-i
    #
    #                 pick = int(np.argmax(score_board)) # choose the index that has the highest score
    #                 sequence = preds[pick]  # pick the decoded sequence
    #                 model_contribution[pick] += 1 # logging witch model used
    #                 sequence = preds[np.argmax(score_board)]
    #
    #         print('{} sequence:, {}'.format(i, sequence))
    #         # print('decode_time:', time.time() - start_time)
    #
    #         submission.loc[submission['file_name'] == dat, 'SMILES'] = sequence
    #         del(preds)
    #
    #     loop.close()
    #     print('total fault:', fault_counter)
    #     print('model contribution:', model_contribution)
    #     return submission

    #TODO
    def one_test(self, image, reversed_token_map, transform):
        """
        input is one sample for model test function
        :param reversed_token_map: converts prediction to readable format
        :param transform: normalize function
        """
        self._encoder.eval()
        self._decoder.eval()


        imgs = Image.open(image)
        print("Input image mode:", imgs.mode)
        print("Input image size", imgs.size)
        # if image channels is 4, img mode is RGBA convert RGBA into RGB
        if len(imgs.mode) == 4:
            x = np.array(imgs)
            r, g, b, a = np.rollaxis(x, axis = -1)
            r[a == 0] = 255
            g[a == 0] = 255
            b[a == 0] = 255
            x = np.dstack([r, g, b])
            imgs = Image.fromarray(x, 'RGB')
        #imgs = cv2.cvtColor(imgs, cv2.COLOR_RGBA2RGB)
        else:
            pass

        #print(imgs.mode)
        imgs = self.png_to_tensor(imgs)
        #print(imgs.shape)


        imgs = transform(imgs).to(self._device)
        encoded_imgs = self._encoder(imgs.unsqueeze(0))
        print("USING NEW PREDICTION CODE ...")
        predictions = self._decoder(encoded_imgs, decode_lengths=self._decode_length, mode='generation')

        SMILES_predicted_sequence = list(torch.argmax(predictions.detach().cpu(), -1).numpy())[0]
        decoded_sequences = decode_predicted_sequences(SMILES_predicted_sequence, reversed_token_map)


        del (predictions)

        return decoded_sequences


    def ensemble_test(self, image, reversed_token_map, transform):
        """
        input is one sample for model test function
        :param reversed_token_map: converts prediction to readable format
        :param transform: normalize function
        """
        self._encoder.eval()
        self._decoder.eval()


        imgs = Image.open(image)
        print("Input image mode:", imgs.mode)
        print("Input image size", imgs.size)
        # if image channels is 4, img mode is RGBA convert RGBA into RGB
        if len(imgs.mode) == 4:
            x = np.array(imgs)
            r, g, b, a = np.rollaxis(x, axis = -1)
            r[a == 0] = 255
            g[a == 0] = 255
            b[a == 0] = 255
            x = np.dstack([r, g, b])
            imgs = Image.fromarray(x, 'RGB')
        #imgs = cv2.cvtColor(imgs, cv2.COLOR_RGBA2RGB)
        else:
            pass




        #print(imgs.mode)
        imgs = self.png_to_tensor(imgs)
        #print(imgs.shape)


        imgs = transform(imgs).to(self._device)
        encoded_imgs = self._encoder(imgs.unsqueeze(0))
        print("USING NEW PREDICTION CODE ...")
        predictions = self._decoder(encoded_imgs, decode_lengths=self._decode_length, mode='generation')

        SMILES_predicted_sequence = list(torch.argmax(predictions.detach().cpu(), -1).numpy())[0]
        decoded_sequences = decode_predicted_sequences(SMILES_predicted_sequence, reversed_token_map)


        del (predictions)

        return decoded_sequences



    def png_to_tensor(self, img: Image):
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

    def is_smiles(self, sequence):
        """
        check the sequence matches with the SMILES format
        :param sequence: decoded sequence
        :return: True or False
        """
        m = Chem.MolFromSmiles(sequence)
        return False if m == None else True

    def model_save(self, save_num):
        torch.save(
            self._decoder.state_dict(),
            '{}/'.format(self._model_save_path) + self._model_name + '/decoder{}.pkl'.format(
                str(save_num))
        )
        torch.save(
            self._encoder.state_dict(),
            '{}/'.format(self._model_save_path) + self._model_name + '/encoder{}.pkl'.format(
                str(save_num))
        )

    def model_load(self):

        # decoder_state = self._decoder.state_dict()
        # if you want to not load embedding:

        # trained_state = torch.load(....)
        #

        decoder_checkpoint = '{}/decoder{}.pkl'.format(self._model_load_path, str(self._model_load_num))
        #print(decoder_checkpoint)
        #lg
        #pubchem
        self._decoder.load_state_dict(
            torch.load('{}/decoder{}.pkl'.format(self._model_load_path, str(self._model_load_num)),
                       map_location=self._device)
        )

        #print(('{}/decoder{}.pkl'.format(self._model_load_path, str(self._model_load_num))

        try:
            self._encoder.load_state_dict(
                torch.load('{}/encoder{}.pkl'.format(self._model_load_path, str(self._model_load_num)),
                           map_location=self._device)
            )
        except RuntimeError as e:
            self._encoder.module.load_state_dict(
                torch.load('{}/encoder{}.pkl'.format(self._model_load_path, str(self._model_load_num)),
                           map_location=self._device)
            )

    def _model_name_maker(self):
        name = 'model-emb_dim_{}-attention_dim_{}-decoder_dim_{}-dropout_{}-batch_size_{}'.format(
            self._emb_dim, self._attention_dim, self._decoder_dim, self._dropout, self._batch_size)
        return name

    def _accuracy_calcluator(self, prediction: np.array, target: np.array):
        prediction = np.argmax(prediction, 2)
        l_p = prediction.shape[1]
        l_t = target.shape[1]
        dist = abs(l_p - l_t)

        if l_p > l_t:
            accr = np.array(prediction[:, :-dist] == target, dtype=np.int).mean()
        elif l_p < l_t:
            accr = np.array(prediction == target[:, :-dist], dtype=np.int).mean()
        else:
            accr = np.array(prediction == target, dtype=np.int).mean()

        return accr

    def _seed_everything(self, seed):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = True

    async def async_fps(self, m1, m2):
        return FPS(m1, m2)
