from model import *
import torch

if __name__ == '__main__':
    mode = 'test'

    gpu = False
    tokenizer = get_kobart_tokenizer()
    train_data_path = './open/train_data.csv'
    test_data_path = './open/sample_test_data.csv'
    save_model_path = './models/model_state_dict'
    model = KoBartModel()
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-6)

    if gpu : device = torch.device('cuda')
    else : device = torch.device('cpu')
    if mode == 'test':
        test(model, tokenizer, save_model_path, test_data_path)
    elif mode == 'train':
        train(model, tokenizer, optimizer, device, save_model_path, train_data_path)


