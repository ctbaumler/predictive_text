import torch
import os, random, re
from transformers import AutoTokenizer, LlamaForCausalLM
import pandas as pd
import numpy as np

random.seed(1234)
np.random.seed(1234)


class AutoComplete():
    def __init__(self, model_name="meta-llama/Llama-2-7b-chat-hf"):
        self.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        if self.device != "cpu":
            torch.cuda.empty_cache()
        self.num_suggestions = 3
        self.model_name = model_name
        self.min_logit = 13 # Currently set high enough that it does not have an effect, but turn down to be more conservative about how many options to show the user.
        self.num_extra=2 # In case there are dupliacate or banned suggestions in the top n, look at more options to try to get n valid suggestions
        self.load()
        self.ban_list = set([""," ","\n","\n\n", '...']) # Do not include "blank" suggestions

    def load(self):
        self.model = LlamaForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=torch.float16,
            device_map='auto',
            use_auth_token=os.environ["HUGGINGFACE_TOKEN"]
        )
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, use_auth_token=os.environ["HUGGINGFACE_TOKEN"])
        self.tokenizer.use_default_system_prompt = False

        #setting the unused tokens to supress warnings
        self.tokenizer.sep_token = "[SEP]"
        self.tokenizer.pad_token = "[PAD]"
        self.tokenizer.cls_token = "[CLS]"
        self.tokenizer.mask_token = "[MASK]"

        #first gen is slow, so do it on load
        self.setup([], 'test')
        _ = self.model(self.input('test'))


    def setup(self, chat_history, system_prompt):
        self.conversation = []
        self.conversation.append({"role": "system", "content" : system_prompt})

        for h in chat_history:
            self.conversation.extend([{"role": 'user', "content": h[0]}, {'role':'assistant', 'content':h[1]}])

    
    def input(self, usr):
        return self.tokenizer.apply_chat_template(self.conversation + [{"role":"user", "content": usr}], return_tensors = 'pt').to(self.device)
    

    def update(self, seq, update):
        if type(update) == str:
            update = self.tokenizer.encode(update)

        seq = self.tokenizer.decode(self.tokenizer.encode(seq)+update, skip_special_tokens=True)

        return seq


        
    def pred_word(self, seq="", append_start=True):

        complete_word = re.compile(r"<\/s>|[^\w\d\s'-](\w+|\d+)|((\w+-\w+|\w+'\w+|\w+|\w+|\d+|[^\w\d\s]+)([^\w\d]+?\s|[^\w\d'-]+))")

        if append_start:
            # Set up as though the model has already generated up to the current point and will now generate the next word (and any spacing or puncuation that needed to come before it)
            input_ids = self.input(seq+ "[/INST] " + seq)[:,:-4]
        else:
            input_ids = self.input(seq)

        output_st = input_ids.shape[1]
      
        model_inputs = self.model.prepare_inputs_for_generation(input_ids)

        with torch.no_grad():
            logits = self.model(**model_inputs).logits[:, -1, :]

        l_values, idxes = torch.topk(logits,self.num_suggestions+self.num_extra)
        l_values = l_values[0]
        idxes = idxes[0]

        completed = [False for _ in range(len(idxes))]
        top_input_id = [torch.cat([input_ids, idx.unsqueeze(0)[:, None]], dim=-1) for idx in idxes]


        while False in completed:
            for i in range(len(top_input_id)):
                # Generate another token for every incomplete option. Stop when we decode with a word boundary to make sure we're not showing a subword to the user
                if not completed[i]:

                    model_inputs = self.model.prepare_inputs_for_generation(top_input_id[i])

                    with torch.no_grad():
                        logits = self.model(**model_inputs).logits[:, -1, :]

                    next_tokens = torch.argmax(logits, dim=-1)

                    updated = torch.cat([top_input_id[i], next_tokens[:, None]], dim=-1)


                    if next_tokens[-1] == 29889:
                        updated = torch.cat([updated,  torch.tensor([[2,1]]).to(self.device)], dim=-1)

                    dec = self.tokenizer.decode(updated[0][output_st:])
                    re_out = complete_word.findall(dec)

                    if len(re_out) == 0:
                        top_input_id[i] = updated

                    else:
                        completed[i] = True


        options = []

        for i,gen_ids in enumerate(top_input_id):
            # From the generated options, choose the top n unbaned options. 
            # If there are any above the minimum logit value, pick from those above the minimum. Else, pick the top choice only.
            if len(options) >= self.num_suggestions:
                break

            gen_ids = gen_ids[0][output_st:].tolist()
            value = self.tokenizer.decode(gen_ids, skip_special_tokens=True)
            
            logit = l_values[i].item()

            if logit < self.min_logit and len(options)>0:
                break

            if value not in self.ban_list and "\n" not in value:
                options.append([gen_ids, value, logit ])

        return options

        
def example(n=5):
    model_name = "meta-llama/Llama-2-7b-chat-hf"
    

    ac = AutoComplete(model_name=model_name)

    prompts = pd.read_csv("prompts.tsv", sep="\t")
    prompts.chat_history = prompts.apply(lambda x: eval(x.chat_history), axis=1)

    idx = 6
    row = prompts.loc[idx]


    ac.setup(row.chat_history, row.system_prompt)

    seq  = row.prefix


    for _ in range(n):
        # Simulate the user accepting the first suggestion for n iterations

        suggestions = ac.pred_word(seq, append_start = True)
        print("Partial story: " + seq + "\nSuggestions: " + str([x[1] for x in suggestions]) + "\nSelcted: " + suggestions[0][1] + "\n\n")

        seq = ac.update(seq, suggestions[0][0])

    print("Final story after " + str(n) + " iterations: " + seq)

if __name__ == '__main__':
    example()
