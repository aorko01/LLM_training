import pickle

#iterates the list and returns a dictionary of pairs and their occurrence
def get_pairs(lists:list,counts:dict=None)->dict:
    counts = counts if counts is not None else {}
    
    for pair in zip(lists[:],lists[1:]):
        counts[pair]=counts.get(pair,0)+1
    return counts

def merge(lists:list,pair:list,new_id:int)->list:
    #creating a new list because in place deletion from list is expensive O(N) 
    #so sacrificing memory for time 
    new_list=[]
    i=0
    while i<(len(lists)-1):
        if lists[i]==pair[0] and lists[i+1]==pair[1]:
            new_list.append(new_id)
            i+=2
        else:
            new_list.append(lists[i])
            i+=1
    
    if i < len(lists):
        new_list.append(lists[i])
    
    return new_list

class Tokenizer:
    
    def __init__(self):
        self.merged={}
        self.vocab=[]
        
    def train(self,text:str,vocab_size:int):
        if vocab_size<256:
            raise ValueError("vocab size must be bigger than 256")
        
        num_it=vocab_size-256
        
        bytes_data=text.encode("utf-8")
        lists=list(bytes_data)
        merges={}
        vocab={idx:bytes([idx]) for idx in range(256)}
        for i in range(num_it):
            #get the dictionary with the frequency 
            counts=get_pairs(lists)
            if not counts:
                break
            #pair with most repetition 
            best_pair = max(counts, key=counts.get)
            #merge the most occurring pairs
            new_id=256+i
            lists=merge(lists,best_pair,new_id)
            #this is needed for encoding 
            merges[best_pair]=256+i
            vocab[new_id]=vocab[best_pair[0]]+vocab[best_pair[1]]
        
        self.merged=merges
        self.vocab=vocab
        
        
    def encode(self,text:str):
        text_bytes=text.encode("utf-8")
        lists=list(text_bytes)
        while len(lists)>=2:
            counts=get_pairs(lists)
            
            if not counts:
                break
            mini=float("inf")
            lowest_pair=[]
            for pair in counts:
                curr_idx=self.merged.get(pair,float("inf"))
                if curr_idx<mini:
                    mini=curr_idx
                    lowest_pair=pair
            if len(lowest_pair)<2:
                break
            merge_idx=self.merged[lowest_pair]
            lists=merge(lists,lowest_pair,merge_idx)
            
        return lists
    
    
    def decode(self,lists:list)->str:
        text_bytes=b""
        for idx in lists:
            text_bytes+=self.vocab[idx]
        text=text_bytes.decode("utf-8")
        return text
        
        
    # def save_to_disk(self, filename):
    #     with open(filename, "w") as f:
    #         for word in self.vocab.items():
    #             f.write(f"{word}\n")
    #         word="\n"*10
    #         f.write(word)
    #         for pair, token_id in self.merged.items():
    #             f.write(f"{pair} {token_id}\n")
    def save_to_disk(self, filename: str):
        data = {
            "vocab": self.vocab,
            "merged": self.merged
        }

        with open(filename, "wb") as f:
            pickle.dump(data, f)
            
    def load_from_disk(self, filename: str):
        with open(filename, "rb") as f:
            data = pickle.load(f)

        self.vocab = data["vocab"]
        self.merged = data["merged"]
        
            
            
