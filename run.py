import numpy as np
import torch
import pickle
import hashlib
import logging
from tqdm.auto import tqdm
import torch.optim as optim
from pathlib import Path
import utils
import os
import sys
import logging
from dataloader import DataLoaderTrain, DataLoaderTest
from torch.utils.data import Dataset, DataLoader
from preprocess import read_news, read_news_bert, get_doc_input, get_doc_input_bert
from model_bert import ModelBert
from parameters import parse_args

from transformers import AutoTokenizer, AutoModel, AutoConfig

finetuneset={
'encoder.layer.6.attention.self.query.weight',
'encoder.layer.6.attention.self.query.bias',
'encoder.layer.6.attention.self.key.weight',
'encoder.layer.6.attention.self.key.bias',
'encoder.layer.6.attention.self.value.weight',
'encoder.layer.6.attention.self.value.bias',
'encoder.layer.6.attention.output.dense.weight',
'encoder.layer.6.attention.output.dense.bias',
'encoder.layer.6.attention.output.LayerNorm.weight',
'encoder.layer.6.attention.output.LayerNorm.bias',
'encoder.layer.6.intermediate.dense.weight',
'encoder.layer.6.intermediate.dense.bias',
'encoder.layer.6.output.dense.weight',
'encoder.layer.6.output.dense.bias',
'encoder.layer.6.output.LayerNorm.weight',
'encoder.layer.6.output.LayerNorm.bias',
'encoder.layer.7.attention.self.query.weight',
'encoder.layer.7.attention.self.query.bias',
'encoder.layer.7.attention.self.key.weight',
'encoder.layer.7.attention.self.key.bias',
'encoder.layer.7.attention.self.value.weight',
'encoder.layer.7.attention.self.value.bias',
'encoder.layer.7.attention.output.dense.weight',
'encoder.layer.7.attention.output.dense.bias',
'encoder.layer.7.attention.output.LayerNorm.weight',
'encoder.layer.7.attention.output.LayerNorm.bias',
'encoder.layer.7.intermediate.dense.weight',
'encoder.layer.7.intermediate.dense.bias',
'encoder.layer.7.output.dense.weight',
'encoder.layer.7.output.dense.bias',
'encoder.layer.7.output.LayerNorm.weight',
'encoder.layer.7.output.LayerNorm.bias',
'pooler.dense.weight',
'pooler.dense.bias',
'rel_pos_bias.weight',
'classifier.weight',
'classifier.bias'}
def train(args):
    # Only support title Turing now
    assert args.enable_hvd  # TODO
    if args.enable_hvd:
        import horovod.torch as hvd

    if args.load_ckpt_name is not None:
        #TODO: choose ckpt_path
        ckpt_path = utils.get_checkpoint(args.model_dir, args.load_ckpt_name)
    else:
        ckpt_path = utils.latest_checkpoint(args.model_dir)


    hvd_size, hvd_rank, hvd_local_rank = utils.init_hvd_cuda(
        args.enable_hvd, args.enable_gpu)

    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    config = AutoConfig.from_pretrained("bert-base-uncased", output_hidden_states=True)
    bert_model = AutoModel.from_pretrained("bert-base-uncased",config=config)
    
    #bert_model.load_state_dict(torch.load('../bert_encoder_part.pkl'))
    # freeze parameters
    for name,param in bert_model.named_parameters():
        if name not in finetuneset:
            param.requires_grad = False
    
    news, news_index, category_dict, domain_dict, subcategory_dict = read_news_bert(
        os.path.join(args.root_data_dir,
                    f'{args.dataset}/{args.train_dir}/news.tsv'), 
        args,
        tokenizer
    )
    
    news_title, news_title_type, news_title_attmask, \
    news_abstract, news_abstract_type, news_abstract_attmask, \
    news_body, news_body_type, news_body_attmask, \
    news_category, news_domain, news_subcategory = get_doc_input_bert(
        news, news_index, category_dict, domain_dict, subcategory_dict, args)
    
    news_combined = np.concatenate([
        x for x in
        [news_title, news_title_type, news_title_attmask, \
            news_abstract, news_abstract_type, news_abstract_attmask, \
            news_body, news_body_type, news_body_attmask, \
            news_category, news_domain, news_subcategory]
        if x is not None], axis=1)
    
    model = ModelBert(args, bert_model, len(category_dict), len(domain_dict), len(subcategory_dict))
    word_dict = None
    
    if args.enable_gpu:
        model = model.cuda()

    lr_scaler = hvd.local_size()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    if args.enable_hvd:
        hvd.broadcast_parameters(model.state_dict(), root_rank=0)
        hvd.broadcast_optimizer_state(optimizer, root_rank=0)
        compression = hvd.Compression.none
        optimizer = hvd.DistributedOptimizer(
            optimizer,
            named_parameters=model.named_parameters(),
            compression=compression,
            op=hvd.Average)


    dataloader = DataLoaderTrain(
        news_index=news_index,
        news_combined=news_combined,
        word_dict=word_dict,
        data_dir=os.path.join(args.root_data_dir,
                            f'{args.market}/{args.train_dir}'),
        filename_pat=args.filename_pat,
        args=args,
        world_size=hvd_size,
        worker_rank=hvd_rank,
        cuda_device_idx=hvd_local_rank,
        enable_prefetch=True,
        enable_shuffle=True,
        enable_gpu=args.enable_gpu,
    )

    logging.info('Training...')
    for ep in range(args.epochs):
        loss = 0.0
        accuary = 0.0
        for cnt, (log_ids, log_mask, input_ids, targets) in enumerate(dataloader):
            if cnt > args.max_steps_per_epoch:
                break

            if args.enable_gpu:
                log_ids = log_ids.cuda(non_blocking=True)
                log_mask = log_mask.cuda(non_blocking=True)
                input_ids = input_ids.cuda(non_blocking=True)
                targets = targets.cuda(non_blocking=True)

            bz_loss, y_hat = model(input_ids, log_ids, log_mask, targets)
            loss += bz_loss.data.float()
            accuary += utils.acc(targets, y_hat)
            optimizer.zero_grad()
            bz_loss.backward()
            optimizer.step()

            if cnt % args.log_steps == 0:
                logging.info(
                    '[{}] Ed: {}, train_loss: {:.5f}, acc: {:.5f}'.format(
                        hvd_rank, cnt * args.batch_size, loss.data / cnt,
                        accuary / cnt))
            
            # save model minibatch
            print(hvd_rank,cnt,args.save_steps,cnt%args.save_steps)
            if hvd_rank == 0 and cnt % args.save_steps == 0:
                ckpt_path = os.path.join(args.model_dir, f'epoch-{ep+1}-{cnt}.pt')
                torch.save(
                    {
                        'model_state_dict': model.state_dict(),
                        'category_dict': category_dict,
                        'word_dict': word_dict,
                        'domain_dict': domain_dict,
                        'subcategory_dict': subcategory_dict
                    }, ckpt_path)
                logging.info(f"Model saved to {ckpt_path}")

        loss /= cnt
        print(ep + 1, loss)

        # save model last of epoch
        if hvd_rank == 0:
            ckpt_path = os.path.join(args.model_dir, f'epoch-{ep+1}.pt')
            torch.save(
                {
                    'model_state_dict': model.state_dict(),
                    'category_dict': category_dict,
                    'word_dict': word_dict,
                    'domain_dict': domain_dict,
                    'subcategory_dict': subcategory_dict
                }, ckpt_path)
            logging.info(f"Model saved to {ckpt_path}")

    dataloader.join()


def test(args):

    if args.enable_hvd:
        import horovod.torch as hvd

    hvd_size, hvd_rank, hvd_local_rank = utils.init_hvd_cuda(
        args.enable_hvd, args.enable_gpu)

    if args.load_ckpt_name is not None:
        #TODO: choose ckpt_path
        ckpt_path = utils.get_checkpoint(args.model_dir, args.load_ckpt_name)
    else:
        ckpt_path = utils.latest_checkpoint(args.model_dir)

    assert ckpt_path is not None, 'No ckpt found'
    checkpoint = torch.load(ckpt_path)

    if 'subcategory_dict' in checkpoint:
        subcategory_dict = checkpoint['subcategory_dict']
    else:
        subcategory_dict = {}

    category_dict = checkpoint['category_dict']
    word_dict = checkpoint['word_dict']
    domain_dict = checkpoint['domain_dict']
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    config = AutoConfig.from_pretrained("bert-base-uncased", output_hidden_states=True)
    bert_model = AutoModel.from_pretrained("bert-base-uncased",config=config)
    model = ModelBert(args, bert_model, len(category_dict), len(domain_dict), len(subcategory_dict))
    
    if args.enable_gpu:
        model.cuda()

    model.load_state_dict(checkpoint['model_state_dict'])
    logging.info(f"Model loaded from {ckpt_path}")

    if args.enable_hvd:
        hvd.broadcast_parameters(model.state_dict(), root_rank=0)

    model.eval()
    torch.set_grad_enabled(False)

    news, news_index, category_dict, domain_dict, subcategory_dict = read_news_bert(
        os.path.join(args.root_data_dir,
                    f'{args.market}/{args.test_dir}/news.tsv'), 
        args,
        tokenizer
    )

    news_title, news_title_type, news_title_attmask, \
    news_abstract, news_abstract_type, news_abstract_attmask, \
    news_body, news_body_type, news_body_attmask, \
    news_category, news_domain, news_subcategory = get_doc_input_bert(
        news, news_index, category_dict, domain_dict, subcategory_dict, args)

    news_combined = np.concatenate([
    x for x in
    [news_title, news_title_type, news_title_attmask, \
    news_abstract, news_abstract_type, news_abstract_attmask, \
    news_body, news_body_type, news_body_attmask, \
    news_category, news_domain, news_subcategory]
    if x is not None], axis=1)


    class NewsDataset(Dataset):
        def __init__(self, data):
            self.data = data

        def __getitem__(self, idx):
            return self.data[idx]

        def __len__(self):
            return self.data.shape[0]

    def news_collate_fn(arr):
        arr = torch.LongTensor(arr)
        return arr

    news_dataset = NewsDataset(news_combined)
    news_dataloader = DataLoader(news_dataset,
                                batch_size=args.batch_size * 4,
                                num_workers=args.num_workers,
                                collate_fn=news_collate_fn)

    news_scoring = []
    with torch.no_grad():
        for input_ids in tqdm(news_dataloader):
            input_ids = input_ids.cuda()
            news_vec = model.news_encoder(input_ids)
            news_vec = news_vec.to(torch.device("cpu")).detach().numpy()
            news_scoring.extend(news_vec)


    news_scoring = np.array(news_scoring)

    logging.info("news scoring num: {}".format(news_scoring.shape[0]))
 

    dataloader = DataLoaderTest(
        news_index=news_index,
        news_scoring=news_scoring,
        word_dict=word_dict,
        news_bias_scoring= None,
        data_dir=os.path.join(args.root_data_dir,
                            f'{args.market}/{args.test_dir}'),
        filename_pat=args.filename_pat,
        args=args,
        world_size=hvd_size,
        worker_rank=hvd_rank,
        cuda_device_idx=hvd_local_rank,
        enable_prefetch=True,
        enable_shuffle=False,
        enable_gpu=args.enable_gpu,
    )

    from metrics import roc_auc_score, ndcg_score, mrr_score, ctr_score

    AUC = []
    MRR = []
    nDCG5 = []
    nDCG10 = []

    def print_metrics(hvd_local_rank, cnt, x):
        logging.info("[{}] Ed: {}: {}".format(hvd_local_rank, cnt, \
            '\t'.join(["{:0.2f}".format(i * 100) for i in x])))

    def get_mean(arr):
        return [np.array(i).mean() for i in arr]

    #for cnt, (log_vecs, log_mask, news_vecs, news_bias, labels) in enumerate(dataloader):
    
    for cnt, (log_vecs, log_mask, news_vecs, news_bias, labels) in enumerate(dataloader):
        his_lens = torch.sum(log_mask, dim=-1).to(torch.device("cpu")).detach().numpy()

        if args.enable_gpu:
            log_vecs = log_vecs.cuda(non_blocking=True)
            log_mask = log_mask.cuda(non_blocking=True)

        user_vecs = model.user_encoder(log_vecs, log_mask).to(torch.device("cpu")).detach().numpy()

        for index, user_vec, news_vec, bias, label, his_len in zip(
                range(len(labels)), user_vecs, news_vecs, news_bias, labels, his_lens):
                
            if label.mean() == 0 or label.mean() == 1:
                continue

            score = np.dot(
                news_vec, user_vec
            )


            
            auc = roc_auc_score(label, score)
            mrr = mrr_score(label, score)
            ndcg5 = ndcg_score(label, score, k=5)
            ndcg10 = ndcg_score(label, score, k=10)


            AUC.append(auc)
            MRR.append(mrr)
            nDCG5.append(ndcg5)
            nDCG10.append(ndcg10)

        if cnt % args.log_steps == 0:

            print_metrics(hvd_rank, cnt * args.batch_size, get_mean([AUC, MRR, nDCG5,  nDCG10]))

    # stop scoring
    dataloader.join()

    for i in range(2):
        print_metrics(hvd_rank, cnt * args.batch_size, get_mean([AUC, MRR, nDCG5,  nDCG10]))
if __name__ == "__main__":
    utils.setuplogger()
    args = parse_args()
    Path(args.model_dir).mkdir(parents=True, exist_ok=True)
    if 'train' in args.mode:
        train(args)
    if 'test' in args.mode:
        test(args)
