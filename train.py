import yaml
import argparse
from trainer import Trainer
from gpt import GPT
from tokenizer import tokenizer
from optimizers import Adam, SGD
from data import Dataset

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('config_file')

    args = parser.parse_args()
    with open(args.config_file, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    
    tok = tokenizer(config['vocab_path'], config['merges_path'])
    optimizer_config = config['optimizer']
    if optimizer_config.get('type') == 'adam':
        optimizer = Adam(lr=config.get('learning_rate', 1e-5), 
                         beta1=optimizer_config.get('beta1', 0.9),
                            beta2=optimizer_config.get('beta2', 0.999),
                            eps=optimizer_config.get('epsilon', 1e-4))
    elif optimizer_config.get('optimzer') == 'sgd':
        optimizer = SGD(lr=config.get('learning_rate'))
    else:
        raise ValueError('Unknown optimizer')
    
    model = GPT(tok.vocab.size, config['num_layers'], 
    config.get('attention_heads', 8), 
    config.get('embedding_dim', 64), 
    config.get('hidden_dropout', 1), 
    config.get('attention_dropout', 1), config['max_seq_len'], optimizer)

    if not config.get('positional_learning', True):
        model.disable_positional_learning()

    dataset = Dataset.create_dataset_from_file(config['dataset_path'], tok, config.get('max_seq_len', 64), config.get('batch_size', config.get('max_seq_len', 64)))

    trainer = Trainer(model, dataset, config.get('batch_size', config.get('max_seq_len', 64)), config.get('num_iterations', 1000), optimizer, config.get('model_save_path', './checkpoint'))
    
    trainer.train()


if __name__ == '__main__':
    main()