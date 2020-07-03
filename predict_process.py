import numpy as np
import pandas as pd
import time
import datetime
import torch

from torch.utils.data import TensorDataset, DataLoader, SequentialSampler


def predict_data_preparation(df, tokenizer, batch_size = 32):
    # Report the number of sentences.
    print('Number of test sentences: {:,}\n'.format(df.shape[0]))

    # Create sentence and label lists
    contents = df['contents'].values
    labels = df['label'].values

    # Tokenize all of the sentences and map the tokens to thier word IDs.
    input_ids = []
    attention_masks = []

    # For every sentence...
    for sent in contents:
    # `encode_plus` will:
    #   (1) Tokenize the sentence.
    #   (2) Prepend the `[CLS]` token to the start.
    #   (3) Append the `[SEP]` token to the end.
    #   (4) Map tokens to their IDs.
    #   (5) Pad or truncate the sentence to `max_length`
    #   (6) Create attention masks for [PAD] tokens.
        encoded_dict = tokenizer.encode_plus(
                        sent,                      # Sentence to encode.
                        add_special_tokens = True, # Add '[CLS]' and '[SEP]'
                        max_length = 512,        # Pad & truncate all sentences.
                        pad_to_max_length = True,
                        return_attention_mask = True,   # Construct attn. masks.
                        return_tensors = 'pt',     # Return pytorch tensors.
                   )
    
        # Add the encoded sentence to the list.    
        input_ids.append(encoded_dict['input_ids'])
    
        # And its attention mask (simply differentiates padding from non-padding).
        attention_masks.append(encoded_dict['attention_mask'])

    # Convert the lists into tensors.
    input_ids = torch.cat(input_ids, dim=0)
    attention_masks = torch.cat(attention_masks, dim=0)
    labels = torch.tensor(labels)

    # Set the batch size.  
      

    # Create the DataLoader.
    prediction_data = TensorDataset(input_ids, attention_masks, labels)
    prediction_sampler = SequentialSampler(prediction_data)
    prediction_dataloader = DataLoader(prediction_data, sampler=prediction_sampler, batch_size=batch_size)

    return prediction_dataloader


# Function to calculate the accuracy of our predictions vs labels
def num_correct_predict(preds, labels):
    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return np.sum(pred_flat == labels_flat)


def format_time(elapsed):
    '''
    Takes a time in seconds and returns a string hh:mm:ss
    '''
    # Round to the nearest second.
    elapsed_rounded = int(round((elapsed)))
    
    # Format as hh:mm:ss
    return str(datetime.timedelta(seconds=elapsed_rounded))



def predict_process(model, predict_dataloader, df):
    t0 = time.time()
    print("Running Prediction...")
    
    #---------------GPU or CPU mode--------------------
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print('There are %d GPU(s) available.' % torch.cuda.device_count())
        print('We will use the GPU:', torch.cuda.get_device_name(0))
    else:
        print('No GPU available, using the CPU instead.')
        device = torch.device("cpu")
    
    model.to(device)
    
    
    # Put the model in evaluation mode--the dropout layers behave differently during evaluation.
    model.eval()

    # Tracking variables 
    num_label, tmp_correct_predict =0,0
    total_predict_label = np.array([], dtype='int8')
    
    for batch in predict_dataloader:
        
        # Add batch to GPU
        # batch = tuple(t.to(device) for t in batch)
        # Unpack the inputs from our dataloader
        # b_input_ids, b_input_mask, b_labels = batch
        b_input_ids = batch[0].type(torch.LongTensor)
        b_input_mask = batch[1].type(torch.LongTensor)
        b_labels = batch[2].type(torch.LongTensor)
        
        
        b_input_ids =  b_input_ids.to(device)
        b_input_mask = b_input_mask.to(device)
        b_labels = b_labels.to(device)

        # Telling the model not to compute or store gradients, saving memory and speeding up validation
        with torch.no_grad():        

            # Forward pass, calculate logit predictions.
            # This will return the logits rather than the loss because we have not provided labels.
            # token_type_ids is the same as the "segment ids", which differentiates sentence 1 and 2 in 2-sentence tasks.
            # The documentation for this "model" function is here: 
            # https://huggingface.co/transformers/v2.2.0/model_doc/bert.html#transformers.BertForSequenceClassification
            outputs = model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask)
        
            # Get the "logits" output by the model. The "logits" are the outputvalues prior to applying an activation function like the softmax.
            logits = outputs[0]
            
            # Move logits and labels to CPU
            logits = logits.detach().cpu().numpy()
            label_ids = b_labels.to('cpu').numpy()
            
            # collect all the predicted labels with numpy structure
            batch_predict_label = np.argmax(logits, axis=1).flatten()
            total_predict_label = np.concatenate((total_predict_label,batch_predict_label),axis = None)
            
            
            # Calculate the accuracy for this batch of test sentences.
            tmp_correct_predict += num_correct_predict(logits, label_ids)
        
            # Accumulate the total accuracy.
            num_label += len(label_ids)

            
            
        # Report the final accuracy for this validation run.
        print("  num of correct:{:}".format(tmp_correct_predict))
        print("  Accuracy: {0:.2f}".format(tmp_correct_predict/num_label))
        print("  num of data: {:}".format(num_label))
        print("  Validation took: {:}".format(format_time(time.time() - t0)))
    df['predict_label']=0
    df['predict_label']=total_predict_label
    print(" Prediction complete!!")
    return df
    