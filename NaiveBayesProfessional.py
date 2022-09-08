import numpy as np
import pandas as pd

class NaiveBayesClassifier:
    
         
    def learn(self,data,response):   
        # meta information of data
        self.num_samples = data.shape[0]
        self.num_features = data.shape[1]
        self.num_classes = list(np.unique(response))
            
        # Probability of each class
        self.cls_dict = {}
        for cls in self.num_classes:
            self.cls_dict.update({cls: np.count_nonzero(response == cls) / self.num_samples})
            
        self.feature_likelyhood = self.calc_feature_likelyhood(data,response)
        
    def predict(self,data):
        
        predictions = []
         
        for i in data.iterrows():    # Iterate through rows
            tmp = []
            for cls in range(len(self.num_classes)): # Iterate through number of classes in dataset
                
                feature_index = 0
                probability = 1
                               
                for feature in i[1]:        # check if feature is in likelyhood dictionary
                    if self.feature_likelyhood[cls][feature_index].get(feature) is None:
                        probability *= 0
                    else:
                        probability *= self.feature_likelyhood[cls][feature_index].get(feature)
                    
                    feature_index +=1
                    
                #print(self.cls_dict.get(self.num_classes[cls]))
                tmp.append(self.cls_dict.get(self.num_classes[cls]) * probability)
            
            for x in range(len(tmp)):
                print(tmp[x])
            #print("******")
            #print(self.num_classes[tmp.index(max(tmp))])
            #print("&&&&&&&")
            predictions.append(self.num_classes[tmp.index(max(tmp))]) 
        
        return predictions
        
        
    def calc_feature_likelyhood(self,data,response):     
    
        split_data = self.calc_lk(data,response)
        likelyHood_array = []
        
        for s in split_data:
            tmp = []
            for i in s:    # columns in s 
                
                likelyhood = {}
                for j,value in s[i].items():   # Iterates through col and counts number of unique values
                    
                    if value in likelyhood.keys():                      
                        likelyhood[value] += 1
                    else:
                        likelyhood[value] = 1 
                
                for count in likelyhood.keys(): # Iterates through dictionary keys and divides by num_observations
                    likelyhood[count] = likelyhood.get(count) / s.shape[0]
                
                tmp.append(likelyhood)
            likelyHood_array.append(tmp)
        
        return likelyHood_array

    def calc_lk(self,data,response):
        
        split_data = []
        
        for x in range(len(self.num_classes)):
            
            split_data.append(pd.DataFrame())
        
        ind_count = 0
        for cls in self.num_classes:
            tmp = []
            for i,y in response.items():
                if y == cls: 
                    tmp.append(data.iloc[i])
                    
            split_data[ind_count] = pd.DataFrame(tmp)
            ind_count +=1
        return(split_data) 