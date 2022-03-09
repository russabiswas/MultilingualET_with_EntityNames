import torch
from transformers import BertTokenizer, BertModel
import pandas as pd

def readfiles(filename, delimeter):
	with open(filename,'r') as f:
		f_data = f.readlines()
	if delimeter != 'NONE':
		data = [x.strip().split(delimeter) for x in f_data] 
	else:
		data = [x.strip() for x in f_data]
	return data


tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')

text = readfiles('entities', 'NONE')


lst_marked_text = []
lst_tensor_token = []
lst_segments_tensors = []
for idx, i in enumerate(text):
  #print(i)
  marked_text = "[CLS] " + i + " [SEP]"
  lst_marked_text.append(marked_text)
  tokenized_text = tokenizer.tokenize(marked_text)
  indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)
  #for tup in zip(tokenized_text, indexed_tokens):
  #print('{:<12} {:>6,}'.format(tup[0], tup[1]))
  segments_ids = [idx+1] * len(tokenized_text)
  #print(segments_ids)
  tokens_tensor = torch.tensor([indexed_tokens])
  lst_tensor_token.append(tokens_tensor)
  segments_tensors = torch.tensor([segments_ids])
  lst_segments_tensors.append(segments_tensors)

#print(lst_tensor_token,lst_segments_tensors)

res = zip(lst_tensor_token, lst_segments_tensors)

#for i in zip(lst_tensor_token, lst_segments_tensors):
#  print(i[0])

# Load pre-trained model (weights)
model = BertModel.from_pretrained('bert-base-multilingual-cased',
                                  output_hidden_states = True, # Whether the model returns all hidden-states.
                                  )

# Put the model in "evaluation" mode, meaning feed-forward operation.
model.eval()

lst_hidden_states =[]

with torch.no_grad():
    for i in res:
      outputs = model(i[0], i[1])
      #print(i)
      #print(outputs)
    # Evaluating the model will return a different number of objects based on 
    # how it's  configured in the `from_pretrained` call earlier. In this case, 
    # becase we set `output_hidden_states = True`, the third item will be the 
    # hidden states from all layers. See the documentation for more details:
    # https://huggingface.co/transformers/model_doc/bert.html#bertmodel
      hidden_states = outputs[2]
      lst_hidden_states.append(hidden_states)

# `hidden_states` is a Python list.
print(len(lst_hidden_states))
print('Type of hidden_states: ', type(hidden_states))

# Each layer in the list is a torch tensor.
print('Tensor shape for each layer: ', hidden_states[0].size())

# Concatenate the tensors for all layers. We use `stack` here to
# create a new dimension in the tensor.
lst_token_embeddings = []

for i in lst_hidden_states:
  token_embeddings = torch.stack(i, dim=0)
  token_embeddings.size()
  lst_token_embeddings.append(token_embeddings)

print('Number of token embeddings: ', len(lst_token_embeddings))

# Remove dimension 1, the "batches".
token_embed_lst =[]
for i in lst_token_embeddings:
  token_embeddings = torch.squeeze(i, dim=1)
  print(token_embeddings.size())
  token_embed_lst.append(token_embeddings)

# Swap dimensions 0 and 1.
lst_token_embeddings = []
for i in token_embed_lst:
  token_embeddings = i.permute(1,0,2)
  print(token_embeddings.size())
  lst_token_embeddings.append(token_embeddings)


for i in lst_token_embeddings:
  print(i.size())

# Stores the token vectors, with shape [22 x 3,072]
lst_token_vecs_cat =[]
for i in lst_token_embeddings:
  token_vecs_cat = []

# `token_embeddings` is a [22 x 12 x 768] tensor.

# For each token in the sentence...
  for token in i:
    
    # `token` is a [12 x 768] tensor

    # Concatenate the vectors (that is, append them together) from the last 
    # four layers.
    # Each layer vector is 768 values, so `cat_vec` is length 3,072.
      cat_vec = torch.cat((token[-1], token[-2], token[-3], token[-4]), dim=0)
    
    # Use `cat_vec` to represent `token`.
      token_vecs_cat.append(cat_vec)
  lst_token_vecs_cat.append(token_vecs_cat)

print('total number of words/phrases: ', len(lst_token_vecs_cat))

print ('Shape is: %d x %d' % (len(lst_token_vecs_cat[0]), len(lst_token_vecs_cat[0][0])))

# Stores the token vectors, with shape [22 x 768]
lst_token_vecs_sum = []
for i in lst_token_embeddings:
  token_vecs_sum = []
  for token in i:
    sum_vec = torch.sum(token[-4:], dim=0)
    token_vecs_sum.append(sum_vec)
  lst_token_vecs_sum.append(token_vecs_sum)
# `token_embeddings` is a [22 x 12 x 768] tensor.

# For each token in the sentence...
  

    # `token` is a [12 x 768] tensor

    # Sum the vectors from the last four layers.
      
    
    # Use `sum_vec` to represent `token`.
      
    

print('Number of vectors: ', len(lst_token_vecs_sum))
print ('Shape is: %d x %d' % (len(lst_token_vecs_sum[0]), len(lst_token_vecs_sum[0][0])))

# `hidden_states` has shape [13 x 1 x 22 x 768]

# `token_vecs` is a tensor with shape [22 x 768]
lst_sentence_embedding = []
for i in lst_hidden_states:
  token_vecs = i[-2][0]

# Calculate the average of all 22 token vectors.
  sentence_embedding = torch.mean(token_vecs, dim=0)
  lst_sentence_embedding.append(sentence_embedding)

print(len(lst_sentence_embedding))
print(len(lst_sentence_embedding[0]))

import io
import numpy

vec_file = open('BERT','w')

for idx, i in enumerate(lst_sentence_embedding):
	j = i.numpy()
	entityname = text[idx].replace(" ", "_")
	vec_file.write(entityname+' '+' '.join([str(elem) for elem in j]))
	vec_file.write('\n')


print ("Our final sentence embedding vector of shape:", sentence_embedding.size())



