

#iterates the list and returns a dictionary of pairs and their occurrence
def get_pairs(lists:list,counts:dict)->dict:
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


