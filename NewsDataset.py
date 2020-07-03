import re
import torch
from torch.utils.data import Dataset
from keras.preprocessing.sequence import pad_sequences


class NewsDataset(Dataset):
    def __init__(self, mode, df, tokenizer,max_len=512):
        assert mode in ["train", "predict"]  # 一般訓練你會需要 dev set
        self.mode = mode
        self.len = len(df)
        self.tokenizer = tokenizer
        self.text =df['contents'].values
        self.labels = df['label'].values
        
        #-------------convert to Bert-pretrained tokens-----------------
        self.tokens = []
        for sent in self.text:
        # `encode` will:(1) Tokenize the sentence.(2) Prepend the `[CLS]` token to the start. (3) Append the `[SEP]` token to the end.(4) Map tokens to their IDs.
           encoded_sent = tokenizer.encode(
                        sent,                      # Sentence to encode.
                        add_special_tokens = True, # Add '[CLS]' and '[SEP]'
                        max_length=max_len
                   )
           self.tokens.append(encoded_sent)
        
        #-------------padding-------------------------------------------
        MAX_LEN = max(len(self.tokens[i]) for i in range(len(self.tokens)))
        self.tokens = pad_sequences(self.tokens, maxlen=MAX_LEN, dtype="long", truncating="post", padding="post")

        #-------------Create attention masks----------------------------
        attention_masks = []
        # Create a mask of 1s for each token followed by 0s for padding
        for seq in self.tokens:
           mask = [int(i>0) for i in seq]
           attention_masks.append(mask) 

        #-------------Convert to tensors--------------------------------
        self.tokens = torch.tensor(self.tokens,dtype=torch.long)
        self.masks = torch.tensor(attention_masks,dtype=torch.long)
        self.labels = torch.tensor(self.labels,dtype=torch.long)       

        

    #---------inherit from Dataset needing define __len__ and __getitem__ methods--------------
    def __len__(self):
        return self.len
  
    def __getitem__(self, idx):
       if self.mode == "predict":
            token = self.tokens[idx]
            mask =self.masks[idx]
            label_tensor = None
       else:
            token = self.tokens[idx]
            mask =self.masks[idx]
            label = self.labels[idx]
       return (token, mask, label)



def contents_trimmer(text, max_len=510):
       
    if(max_len > 510):
        print('error message: max_len must <= 510')
        exit()
    
    new_content=''
    
    if len(text) >= max_len:
        sent = re.split('，|\r\n\r\n\r\n\r\n|r\n\r\n\r\n|\r\n\r\n|\r\n',text)
        sent = [x for x in sent if x !='']
        new_content=sent[0]
        for i in range(1,len(sent)):
            if(len(new_content)+len(sent[i]) < int(max_len)*2/3):
                if(new_content[-1]!='。'):
                    new_content = new_content + '，' + sent[i]
                else:
                    new_content = new_content + sent[i]
            else:
                break
        
        tail_content=sent[-1]
        for i in range(2,len(sent)):
            if(len(tail_content)+len(sent[-i]) < int(max_len)*1/3):
                if(sent[-i]!='。'):
                    tail_content = sent[-i] + '，' + tail_content 
                else:
                    tail_content = sent[-i] + tail_content
            else:
                break
        return new_content +'。'+ tail_content
    
    else:
        sent = re.split('\r\n\r\n\r\n\r\n|r\n\r\n\r\n|\r\n\r\n|\r\n',text)
        new_content = ''.join(sent)
        return new_content

    
def pick_correlative_paragraph(text,labels):
    paragraph=re.split('\r\n\r\n',text)
    picked_paragraph=[]
    for x in paragraph:    
        for label in labels:
            if(label in x):
                picked_paragraph.append(x)
                break
    return ''.join(picked_paragraph)