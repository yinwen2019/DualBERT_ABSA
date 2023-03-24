import os
import sys

from PTNet_bert1 import PTNetBert1Classifier

os.environ['CUDA_VISIBLE_DEVICES'] = "3"
import copy
import random
import logging
import argparse
import torch
import torch.nn as nn
import numpy as np
from sklearn import metrics
from time import strftime, localtime
from torch.utils.data import DataLoader
from transformers import BertModel, AdamW, RobertaModel, DebertaModel, BertForMaskedLM, RobertaForMaskedLM, RobertaConfig

from PTNet_bert import PTNetBertClassifier
from PTNet_roberta import PTNetRoBertaClassifier
from data_utils import Tokenizer4Bert, Tokenizer4RoBerta, Tokenizer4DeBerta, ABSAPROData, ABSARoBertaData

logger = logging.getLogger()
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler(sys.stdout))


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


class Instructor:
    ''' Model training and evaluation '''

    def __init__(self, opt):
        self.opt = opt
        trainset = None
        testset = None
        if opt.model_name == 'PTNet_bert':
            tokenizer = Tokenizer4Bert(opt.max_length, opt.pretrained_bert_name)
            bert = BertModel.from_pretrained(opt.pretrained_bert_name)
            bertmask = BertForMaskedLM.from_pretrained(opt.pretrained_bert_name)
            self.model = opt.model_class((bert, bertmask), opt).to(opt.device)
            trainset = ABSAPROData(opt.dataset_file['train'], tokenizer, opt=opt)
            testset = ABSAPROData(opt.dataset_file['test'], tokenizer, opt=opt)
        if opt.model_name == 'PTNet_bert1':
            tokenizer = Tokenizer4Bert(opt.max_length, opt.pretrained_bert_name)
            bert = BertModel.from_pretrained(opt.pretrained_bert_name)
            self.model = opt.model_class(bert, opt).to(opt.device)
            trainset = ABSAPROData(opt.dataset_file['train'], tokenizer, opt=opt)
            testset = ABSAPROData(opt.dataset_file['test'], tokenizer, opt=opt)
        elif opt.model_name == 'PTNet_roberta':
            tokenizer = Tokenizer4RoBerta(opt.max_length, opt.pretrained_bert_name)
            roberta = RobertaModel.from_pretrained(opt.pretrained_bert_name)
            robertamask = RobertaForMaskedLM.from_pretrained(opt.pretrained_bert_name)
            self.model = opt.model_class((roberta, robertamask), opt).to(opt.device)
            trainset = ABSARoBertaData(opt.dataset_file['train'], tokenizer, opt=opt)
            testset = ABSARoBertaData(opt.dataset_file['test'], tokenizer, opt=opt)

        self.train_dataloader = DataLoader(dataset=trainset, batch_size=opt.batch_size, shuffle=True)
        self.test_dataloader = DataLoader(dataset=testset, batch_size=opt.batch_size)

        if opt.device.type == 'cuda':
            logger.info('cuda memory allocated: {}'.format(torch.cuda.memory_allocated(self.opt.device.index)))
        self._print_args()

    def _print_args(self):
        n_trainable_params, n_nontrainable_params = 0, 0
        for p in self.model.parameters():
            n_params = torch.prod(torch.tensor(p.shape))
            if p.requires_grad:
                n_trainable_params += n_params
            else:
                n_nontrainable_params += n_params

        logger.info('n_trainable_params: {0}, n_nontrainable_params: {1}'.format(n_trainable_params, n_nontrainable_params))
        logger.info('training arguments:')

        for arg in vars(self.opt):
            logger.info('>>> {0}: {1}'.format(arg, getattr(self.opt, arg)))

    def _cloze_params(self):
        if self.opt.model_name == 'PTNet_bert':
            for name, para in self.model.named_parameters():
                if 'pro_model.bert' in name:
                    para.requires_grad = False

    def get_bert_optimizer(self, model):
        # Prepare optimizer and schedule (linear warmup and decay)
        no_decay = ['bias', 'LayerNorm.weight']

        logger.info("bert learning rate on")
        _params = filter(lambda n, p: p.requires_grad, model.named_parameters())
        optimizer_grouped_parameters = [
            {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
             'weight_decay': self.opt.weight_decay},
            {'params': [p for n, p in model.named_parameters() if any(
                nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=self.opt.bert_lr, eps=self.opt.adam_epsilon)

        return optimizer

    def _train(self, criterion, optimizer, max_test_acc_overall=0):
        max_test_acc = 0
        max_f1 = 0
        global_step = 0
        model_path = ''
        for epoch in range(self.opt.num_epoch):
            logger.info('>' * 60)
            logger.info('epoch: {}'.format(epoch))
            n_correct, n_total = 0, 0
            for i_batch, sample_batched in enumerate(self.train_dataloader):
                global_step += 1
                # switch model to training mode, clear gradient accumulators
                self.model.train()
                optimizer.zero_grad()
                inputs = [sample_batched[col].to(self.opt.device) for col in self.opt.inputs_cols]
                outputs, outputs1, outputs2, loss = self.model(inputs)
                targets = sample_batched['polarity'].to(self.opt.device)
                if self.opt.losstype is not None:
                    # loss1 = criterion(outputs1, targets)
                    # loss2 = criterion(outputs2, targets)
                    losss = criterion(outputs, targets)
                    # loss = (loss1 + loss2) + 1 * loss
                    loss = losss + 10 * loss
                    # outputs = torch.add(outputs1, outputs2)
                else:
                    loss = criterion(outputs1, targets)
                    outputs = outputs1

                loss.backward()
                optimizer.step()

                if global_step % self.opt.log_step == 0:
                    n_correct += (torch.argmax(outputs, -1) == targets).sum().item()
                    n_total += len(outputs)
                    train_acc = n_correct / n_total
                    test_acc, f1 = self._evaluate()
                    if test_acc > max_test_acc:
                        max_test_acc = test_acc
                        if test_acc > max_test_acc_overall:
                            if not os.path.exists('./state_dict'):
                                os.mkdir('./state_dict')
                            model_path = './state_dict/{}_{}_acc_{:.4f}_f1_{:.4f}'.format(self.opt.model_name, self.opt.dataset,
                                                                                          test_acc, f1)
                            self.best_model = copy.deepcopy(self.model)
                            logger.info('>> saved: {}'.format(model_path))
                    if f1 > max_f1:
                        max_f1 = f1
                    logger.info(
                        'loss: {:.4f}, train_acc: {:.4f}, test_acc: {:.4f}, f1: {:.4f}'.format(loss.item(), train_acc, test_acc,
                                                                                               f1))
        return max_test_acc, max_f1, model_path

    def _evaluate(self, show_results=False):
        # switch model to evaluation mode
        self.model.eval()
        n_test_correct, n_test_total = 0, 0
        targets_all, outputs_all = None, None
        with torch.no_grad():
            for batch, sample_batched in enumerate(self.test_dataloader):
                inputs = [sample_batched[col].to(self.opt.device) for col in self.opt.inputs_cols]
                targets = sample_batched['polarity'].to(self.opt.device)
                outputs, outputs1, outputs2, loss = self.model(inputs)
                # outputs = torch.add(outputs1, outputs2)
                n_test_correct += (torch.argmax(outputs, -1) == targets).sum().item()
                n_test_total += len(outputs)
                targets_all = torch.cat((targets_all, targets), dim=0) if targets_all is not None else targets
                outputs_all = torch.cat((outputs_all, outputs), dim=0) if outputs_all is not None else outputs
        test_acc = n_test_correct / n_test_total
        f1 = metrics.f1_score(targets_all.cpu(), torch.argmax(outputs_all, -1).cpu(), labels=[0, 1, 2], average='macro')

        labels = targets_all.data.cpu()
        predic = torch.argmax(outputs_all, -1).cpu()
        if show_results:
            report = metrics.classification_report(labels, predic, digits=4)
            confusion = metrics.confusion_matrix(labels, predic)
            return report, confusion, test_acc, f1

        return test_acc, f1

    def _test(self):
        self.model = self.best_model
        self.model.eval()
        test_report, test_confusion, acc, f1 = self._evaluate(show_results=True)
        logger.info("Precision, Recall and F1-Score...")
        logger.info(test_report)
        logger.info("Confusion Matrix...")
        logger.info(test_confusion)

    def run(self):
        criterion = nn.CrossEntropyLoss()
        # if self.opt.model_name == 'PTNet_bert':
        #     self._cloze_params()
        if 'bert' not in self.opt.model_name:
            _params = filter(lambda p: p.requires_grad, self.model.parameters())
            optimizer = self.opt.optimizer(_params, lr=self.opt.learning_rate, weight_decay=self.opt.l2reg)
        elif self.opt.model_name == 'PTNet_bert':
            optimizer = self.get_bert_optimizer(self.model)
        else:
            optimizer = self.get_bert_optimizer(self.model)
        max_test_acc_overall = 0
        max_f1_overall = 0
        max_test_acc, max_f1, model_path = self._train(criterion, optimizer, max_test_acc_overall)
        logger.info('max_test_acc: {0}, max_f1: {1}'.format(max_test_acc, max_f1))
        max_test_acc_overall = max(max_test_acc, max_test_acc_overall)
        max_f1_overall = max(max_f1, max_f1_overall)
        # torch.save(self.best_model.state_dict(), model_path)
        logger.info('>> saved: {}'.format(model_path))
        logger.info('#' * 60)
        logger.info('max_test_acc_overall:{}'.format(max_test_acc_overall))
        logger.info('max_f1_overall:{}'.format(max_f1_overall))
        self._test()


def main():
    model_classes = {
        'PTNet_bert': PTNetBertClassifier,
        'PTNet_bert1': PTNetBert1Classifier,
        'PTNet_roberta': PTNetRoBertaClassifier
    }

    dataset_files = {
        'restaurant': {
            'train': './dataset/Restaurants_corenlp/train.json',
            'test': './dataset/Restaurants_corenlp/test.json',
        },
        'laptop': {
            'train': './dataset/Laptops_corenlp/train.json',
            'test': './dataset/Laptops_corenlp/test.json'
        },
        'twitter': {
            'train': './dataset/Tweets_corenlp/train.json',
            'test': './dataset/Tweets_corenlp/test.json',
        }
    }

    input_colses = {
        'PTNet_bert': ['input_ids', 'token_type_ids', 'attention_mask', 'label_ids', 'asp_start', 'asp_end', 'mask_index', 'aspect_mask'],
        'PTNet_bert1': ['input_ids', 'token_type_ids', 'attention_mask', 'label_ids', 'asp_start', 'asp_end', 'mask_index', 'aspect_mask'],
        'PTNet_roberta': ['input_ids', 'token_type_ids', 'attention_mask', 'label_ids', 'asp_start', 'asp_end', 'mask_index', 'aspect_mask']
    }

    initializers = {
        'xavier_uniform_': torch.nn.init.xavier_uniform_,
        'xavier_normal_': torch.nn.init.xavier_normal_,
        'orthogonal_': torch.nn.init.orthogonal_,
    }

    optimizers = {
        'adadelta': torch.optim.Adadelta,
        'adagrad': torch.optim.Adagrad,
        'adam': torch.optim.Adam,
        'adamax': torch.optim.Adamax,
        'asgd': torch.optim.ASGD,
        'rmsprop': torch.optim.RMSprop,
        'sgd': torch.optim.SGD,
    }

    pretrain = {
        'bert-base-uncased': '/home/Like/flc/wxl/PreModel/bert-base-uncased',
        'bert-large-uncased': '/home/Like/flc/wxl/PreModel/bert-large-uncased',
        'roberta-base': '/home/Like/flc/wxl/PreModel/roberta-base',
        'deberta-v3-base': 'E:/PreModel/deberta-v3-base',
        'deberta-base': 'E:/PreModel/deberta-base'
    }

    # Hyperparameters
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', default='PTNet_bert', type=str, help=', '.join(model_classes.keys()))
    parser.add_argument('--dataset', default='laptop', type=str, help=', '.join(dataset_files.keys()))
    parser.add_argument('--optimizer', default='adam', type=str, help=', '.join(optimizers.keys()))
    parser.add_argument('--initializer', default='xavier_uniform_', type=str, help=', '.join(initializers.keys()))
    parser.add_argument('--learning_rate', default=0.002, type=float)
    parser.add_argument('--l2reg', default=1e-4, type=float)
    parser.add_argument('--num_epoch', default=20, type=int)
    parser.add_argument('--batch_size', default=16, type=int)
    parser.add_argument('--log_step', default=5, type=int)
    parser.add_argument('--polarities_dim', default=3, type=int, help='3')
    parser.add_argument('--max_length', default=100, type=int, help='input token max length')

    # * PTNet
    parser.add_argument('--attention_heads', default=5, type=int, help='number of multi-attention heads')
    parser.add_argument('--device', default=None, type=str, help='cpu, cuda')
    parser.add_argument('--seed', default=1000, type=int)
    parser.add_argument("--weight_decay", default=0.0, type=float, help="Weight decay if we apply some.")

    parser.add_argument('--losstype', default=None, type=str, help="['doubleloss']")
    parser.add_argument('--alpha', default=0.25, type=float)
    parser.add_argument('--beta', default=0.25, type=float)

    # * bert
    parser.add_argument('--pretrained_bert_name', default='bert-base-uncased', type=str)
    parser.add_argument("--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam optimizer.")
    parser.add_argument('--bert_dim', type=int, default=768)
    parser.add_argument('--bert_dropout', type=float, default=0.3, help='BERT dropout rate.')
    parser.add_argument('--bert_lr', default=2e-5, type=float)
    opt = parser.parse_args()

    opt.model_class = model_classes[opt.model_name]
    opt.dataset_file = dataset_files[opt.dataset]
    opt.inputs_cols = input_colses[opt.model_name]
    opt.initializer = initializers[opt.initializer]
    opt.optimizer = optimizers[opt.optimizer]
    opt.pretrained_bert_name = pretrain[opt.pretrained_bert_name]
    opt.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') if opt.device is None else torch.device(
        opt.device)

    # set random seed
    setup_seed(opt.seed)

    # if not os.path.exists('./log'):
    #     os.makedirs('./log', mode=0o777)
    # log_file = '{}-{}-{}.log'.format(opt.model_name, opt.dataset, strftime("%Y-%m-%d_%H-%M-%S", localtime()))
    # logger.addHandler(logging.FileHandler(os.path.join('log', log_file)))

    ins = Instructor(opt)
    ins.run()


if __name__ == '__main__':
    main()
