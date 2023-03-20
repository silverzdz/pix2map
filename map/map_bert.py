import torch
from transformers import BertTokenizer, BertModel, BertForMaskedLM

if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else 'cpu')
    bert_path = 'bert-base-cased'
    tokenizer = BertTokenizer.from_pretrained(bert_path)
    model = BertModel.from_pretrained(bert_path).to(device)
    masked_lm_model = BertForMaskedLM.from_pretrained(bert_path).to(device)

    text_1 = "Who was Jim Henson ?"
    text_2 = "Jim Henson was a puppeteer"

    # Tokenized input with special tokens around it (for BERT: [CLS] at the beginning and [SEP] at the end)
    indexed_tokens = tokenizer.encode(text_1, text_2, add_special_tokens=True)

    # Define sentence A and B indices associated to 1st and 2nd sentences (see paper)
    segments_ids = [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1]

    # Convert inputs to PyTorch tensors
    segments_tensors = torch.tensor([segments_ids]).to(device)
    tokens_tensor = torch.tensor([indexed_tokens]).to(device)

    with torch.no_grad():
        encoded_layers, _ = model(tokens_tensor, token_type_ids=segments_tensors)
        
    # Mask a token that we will try to predict back with `BertForMaskedLM`
    masked_index = 8
    indexed_tokens[masked_index] = tokenizer.mask_token_id
    tokens_tensor = torch.tensor([indexed_tokens]).to(device)

    with torch.no_grad():
        predictions = masked_lm_model(tokens_tensor, token_type_ids=segments_tensors)

    # Get the predicted token
    predicted_index = torch.argmax(predictions[0][0], dim=1)[masked_index].item()
    predicted_token = tokenizer.convert_ids_to_tokens([predicted_index])[0]
    assert predicted_token == 'Jim'
    
    print(predicted_token)