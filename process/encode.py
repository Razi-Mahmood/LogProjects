import os
import numpy as np
import time
import transformers as ppb # pytorch transformers
from transformers import AutoTokenizer, AutoModel
from sentence_transformers import SentenceTransformer

class BERT_Encode():
    
    def __init__(self):
        model_class, tokenizer_class, pretrained_weights = (ppb.BertModel, ppb.BertTokenizer, 'bert-base-uncased')
        tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        self.bertmodel = SentenceTransformer('paraphrase-distilroberta-base-v1')
        
    #creating BERT embeddings for these blocks is done in Pytorch
    #####Encoding generation
    def get_concat_embedding(self,sentences,bertmodel):
        concat=self.concatenate_sentences(sentences)
        #print(concat)
        sentence_embeddings=bertmodel.encode(concat)
        return sentence_embeddings
    
    def concatenate_sentences(self,sentencesPerBlock):
        cumstring=None
        for sentence in sentencesPerBlock:
            if cumstring is None:
                cumstring=sentence
            else:
                cumstring+=" "+sentence
        return cumstring
    def write_blockids(self,block_map,outfile):
        fout=open(outfile,"w")
        fout.write("BlockID\n")
        for blk in block_map:
            fout.write(blk+"\n")
        fout.close()
    #Save the embedding as a pickle file maintaining the correspondence between selected block and its corresponding vector
    def save_selected_log_embedding(self,selected_blockMap,blocktextMap,bertmodel,batch_size,vector_len,outpath,block_evtMap,outpath_evt):
        #blockidfile=outpath+"_id.txt"
        
        os.makedirs(outpath, exist_ok=True)
        #save the order of the block ids for the corresponding numpy array. Later on make a single pickle file
        outfile=outpath+"_"+"blkids.txt"
        self.write_blockids(selected_blockMap,outfile)
       # fout=open(outfile,"w")
        #for blk in selected_blockMap:
        #    fout.write(blk+"\n")
        #fout.close()
            
        x_array= np.zeros((batch_size, vector_len))
        x_array_evt= np.zeros((batch_size, vector_len))
        blockIDMap={}
        start = time.time()

        #create an ordering of the blockid and store the embedding as a 2d array 
        #this is still large array of 500,000 x 786 length vectors for the training data
        i=0
        count=0
        for blockid in selected_blockMap:
            if (count%100==0):
                print(count)
            average_vec=self.get_concat_embedding(blocktextMap[blockid],bertmodel)
            average_evtvec=self.get_concat_embedding(block_evtMap[blockid],bertmodel)
            #time to save the batch
            if ((count%batch_size==0) and (count>0)):
                #print(count)
                #time to start a new batch to write out
                outfile=outpath+"_"+str(count)+".txt"
                np.save(outfile, x_array)
                outfile_evt=outpath_evt+"_"+str(count)+".txt"
                np.save(outfile_evt, x_array_evt)
                x_array= np.zeros((batch_size, vector_len))
                x_array_evt= np.zeros((batch_size, vector_len))
                i=0
            x_array[i]=average_vec
            x_array_evt[i]=average_evtvec
            i+=1
            count+=1
        end = time.time()
        print((end - start)," seconds",i)
        if (i!=0): #write out the left over batch
            outfile=outpath+"_"+str(count)+".txt"
            outfile_evt=outpath_evt+"_"+str(count)+".txt"
            np.save(outfile, x_array)
            np.save(outfile_evt, x_array_evt)