import re
import random
import pandas as pd
class LogProcess():
     
   

    ### Process 100K log structured HDFS dataset (attached in data)
    #Processing of HDFS_100k.log_structured.csv
    #1. Extracts blocks ids from the log messages
    #2. groups message text by block ids
    #3. Attaches normal or abnormal label with block ids. Normal is 0 and anomaly =1
    def read_anomaly_blockids(self,anomaly_blockfile):
        df=pd.read_csv(anomaly_blockfile)
        blockcol=df["BlockId"]
        labelcol=df["Label"]
        indexarray=df.index
        block_map={}
        for i in range(len(indexarray)):
            blockid=str(blockcol[i])
            label=str(labelcol[i])
            block_map[blockid]=label
        print("Total blocks = ",len(block_map))
        return block_map

    def read_full_dataset(self,dataset_file):
        df=pd.read_csv(dataset_file)
       # contentcol=df["Content"]
        #indexarray=df.index
        content_map={}
        event_map={}
        
        for idx, row in df.iterrows():
            blkId_list = re.findall(r'(blk_-?\d+)', row['Content'])
            blkid=blkId_list[0]
            if blkid not in content_map:
                inner=[]
                eventinner=[]
            else:
                inner=content_map[blkid]
                eventinner=event_map[blkid]
            inner.append(row['Content'])   
            eventinner.append(row['EventId'])
            content_map[blkid]=inner
            event_map[blkid]=eventinner
           # if (len(blkId_list)>1) and (blkId_list[0]!=blkId_list[1]):
             #   print(blkId_list,"->",row["LineId"])
        print("Total blocks = ",len(content_map),len(event_map))
        return content_map,event_map

    #separate the normal and abnormal for both content and event maps to try BERT encodings for raw text versus event sequence
    def separate_normal_abnormal_blocks(self,content_map,event_map,block_label_map):
        normal_map={}
        abnormal_map={}
        normal_evtmap={}
        abnormal_evtmap={}
        normal_blkids=[]
        abnormal_blkids=[]
        for blkid in content_map:
            label=block_label_map[blkid]
            if (label=='Anomaly'):
                abnormal_map[blkid]=content_map[blkid]
                abnormal_evtmap[blkid]=event_map[blkid]
                abnormal_blkids.append(blkid)
            else:
                normal_map[blkid]=content_map[blkid]
                normal_evtmap[blkid]=event_map[blkid]
                normal_blkids.append(blkid)
        print("Normal blocks = ",len(normal_map),len(normal_blkids),len(normal_evtmap),
              "abnormal blocks =",len(abnormal_map),len(abnormal_blkids),len(abnormal_evtmap),"total = ",len(content_map),len(event_map))
        return normal_map,normal_blkids,abnormal_map,abnormal_blkids,normal_evtmap, abnormal_evtmap
    def write_blockids(self,block_map,outfile):
        fout=open(outfile,"w")
        fout.write("BlockID\n")
        for blk in block_map:
            fout.write(blk+"\n")
        fout.close()
    def add_to_randomset(self,num_required,total,item_index,item_map,item_evtmap):
        xitems={}
        xitems_evt={}
        taken=[]
        count=0
        print("current items = ",len(item_map),len(item_index),len(item_evtmap))
        while (count<num_required):
            r=random.randint(0,total)
            if r<len(item_index):
               # print(r,len(item_index))
                blkid=item_index[r]
                if blkid not in taken:
                    taken.append(blkid)
                    count+=1
                    xitems[blkid]=item_map[blkid]
                    xitems_evt[blkid]=item_evtmap[blkid]
        print("retained items = ",len(xitems),len(xitems_evt))
        return taken,xitems,xitems_evt
    def subtract_items(self,item_map,taken_items,item_evtmap):
        remaining_map={}
        remaining_evtmap={}
        retainedids=[]
        for blkid in item_map:
            if (blkid not in taken_items):
                remaining_map[blkid]=item_map[blkid]
                remaining_evtmap[blkid]=item_evtmap[blkid]
                retainedids.append(blkid)
        return remaining_map,retainedids,remaining_evtmap
    #randomly divide the normal and abnormal sets into train-test with 80-20 ratio
    #each run generates one partition
    def random_partition(self,normal_map,normal_blkids,normal_evtmap,abnormal_map,abnormal_blkids,abnormal_evtmap,testratio,outpath):
        normals=len(normal_map)
        abnormals=len(abnormal_map)
        num_test_normal=testratio*normals
        num_test_abnormal=testratio*abnormals
        #go over the normal_map to pick the test samples
        normal_taken,xtest_normal,xtest_normal_evt=self.add_to_randomset(num_test_normal,normals,normal_blkids,normal_map,normal_evtmap)
        abnormal_taken,xtest_abnormal,xtest_abnormal_evt=self.add_to_randomset(num_test_abnormal,abnormals,abnormal_blkids,abnormal_map,abnormal_evtmap)
        xtrain_normal,train_normal_ids,xtrain_normal_evt=self.subtract_items(normal_map,normal_taken,normal_evtmap)
        xtrain_abnormal,train_abnormal_ids,xtrain_abnormal_evt=self.subtract_items(abnormal_map,abnormal_taken,abnormal_evtmap)
        print("Normal train test ",len(xtrain_normal),len(xtest_normal),len(xtrain_normal_evt),len(xtest_normal_evt))
        print("Abnormal train test ",len(xtrain_abnormal),len(xtest_abnormal),len(xtrain_abnormal_evt),len(xtest_abnormal_evt))
        self.write_blockids(normal_taken,outpath+"xtest_normal_ids.txt")
        self.write_blockids(abnormal_taken,outpath+"xtest_abnormal_ids.txt")
        self.write_blockids(train_normal_ids,outpath+"xtrain_normal_ids.txt")
        self.write_blockids(train_abnormal_ids,outpath+"xtrain_abnormal_ids.txt")
        
        return xtrain_normal,xtest_normal,xtrain_abnormal,xtest_abnormal,xtrain_normal_evt,xtest_normal_evt,xtrain_abnormal_evt,xtest_abnormal_evt