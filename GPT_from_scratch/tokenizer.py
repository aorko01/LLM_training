

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
        
    def train(self,text,vocab_size):
        if vocab_size<256:
            raise ValueError("vocab size must be bigger than 256")
        
        num_it=vocab_size-256
        
        bytes_data=text.encode("utf-8")
        lists=list(bytes_data)
        merges={}
        for i in range(num_it):
            #get the dictionary 
            counts=get_pairs(lists)
            if not counts:
                break
            #pair with most repetition 
            best_pair = max(counts, key=counts.get)
            #merge the most occurring pairs
            lists=merge(lists,best_pair,256+i)
            #this is needed for encoding 
            merges[best_pair]=256+i
        
        self.merged=merges
        
    def save_to_disk(self, filename):
        with open(filename, "w") as f:
            for pair, token_id in self.merged.items():
                f.write(f"{pair} {token_id}\n")
        
            
            
def main():
    text = """
    low lower lowest
    low lower lowest
    low lower
    """

    tokenizer = Tokenizer()

    # Learn 20 new tokens (256 -> 276)
    tokenizer.train(text, vocab_size=276)

    tokenizer.save_to_disk("merges.txt")

    print("Training completed!")
    print(f"Learned {len(tokenizer.merged)} merges.")
    print("Saved merges to merges.txt")


if __name__ == "__main__":
    main()