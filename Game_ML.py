#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import random
import math
import pickle 
import pandas as pd
import matplotlib.pyplot as plt
import time




sigmoid_lambda=0.8
momentum=0.5
learning_rate=0.1
no_hidden_layer_nuerons=3

max_min=[]

def sigmoid(x):
    
    #print(-1*x*sigmoid_lambda)
    sig=1/(1+math.exp(-1*x*sigmoid_lambda))
    return sig

def normalize_data(data):
    global max_min
    for i in range(len(data.iloc[0])):
        max_min.append([min(data[i]),max(data[i])])
        data[i]=(data[i]-min(data[i]))/(max(data[i])-min(data[i]))
    return data

def devide_train_test(data):
    tlen=int(len(data)*.70)
    train_data=data[:tlen]
    val_data=data[tlen:]
    val_data=val_data.reset_index(drop=True)
    return train_data,val_data

def shuffle_data(data):
    x=list(range(len(data)))
    random.shuffle(x)
    return data.iloc[x]

def save_model(model,file):
    file_name = open(file, 'wb')
    pickle.dump(model, file_name, pickle.HIGHEST_PROTOCOL)

#parameter={'sigmoid_lambda':sigmoid_lambda,'momentum':momentum,'learning_rate':learning_rate,'no_hidden_layer_nuerons':no_hidden_layer_nuerons}
def save_model_parameters(model_parameter):
    file='model_parameter.pkl'
    file_name = open(file, 'wb')
    pickle.dump(model_parameter, file_name, pickle.HIGHEST_PROTOCOL)
    
def restore_model_parameters():
    
    global sigmoid_lambda
    global momentum
    global learning_rate
    global no_hidden_layer_nuerons
    
    file_name = open('model_parameter.pkl', 'rb')
    model_parameter=pickle.load(file_name)
    
    sigmoid_lambda=model_parameter['sigmoid_lambda']
    momentum=model_parameter['momentum']
    learning_rate=model_parameter['learning_rate']
    no_hidden_layer_nuerons=model_parameter['no_hidden_layer_nuerons']
    
    
class MLP:
    
    def __init__(self,input_size):
        self.input_size=input_size
        self.n_weight=[]    # list contaon no of input weights to each nueron
        self.n_weight.append(input_size)
        self.train_error=[]
        self.total_train_errors=[]
        self.train_MSE=[]
        self.total_val_errors=[]
        self.val_MSE=[]
        self.layers=[]
        self.output=[]
        
        
    def load(self):
        self.addLayer(no_hidden_layer_nuerons,'sigmoid')
        self.addLayer(2,'linear')
        restore_model_parameters()
        self.load_weights()
        
#     def save_weights_and_params(self,params):
#         model=self
#         file_prefix_layer1="MLP_"
#         for i in range(len(model.layers)):
#             file_prefix_layer2=file_prefix_layer1+"Layer_"+str(i)+"_"
#             for j in range(len(model.layers[i].nuerons)):
#                 file=file_prefix_layer2+"Nueron"+str(j)+"_Weights"
#                 #print(model.layers[i].nuerons[j].weights)
#                 save_model(model.layers[i].nuerons[j].weights,file)
                
#         #save_model_parameters(params)
    
#     def load_weights(self):
#         model=self
#         file_prefix_layer1="MLP_"
#         for i in range(len(model.layers)):
#             file_prefix_layer2=file_prefix_layer1+"Layer_"+str(i)+"_"
#             for j in range(len(model.layers[i].nuerons)):
#                 file=file_prefix_layer2+"Nueron"+str(j)+"_Weights"
#                 with open(file, 'rb') as filename:
#                     model.layers[i].nuerons[j].weights = pickle.load(filename)


    def save_weights_and_params(self):
        model=self
        mlp=[]
        for i in range(len(model.layers)):
            layers=[]
            for j in range(len(model.layers[i].nuerons)):
                layers.append(model.layers[i].nuerons[j].weights)
            mlp.append(layers)
        time_stamp=int(time.time())
        save_model(mlp,'model_weights_'+str(time_stamp))
        
        
    def load_weights(self,file):
         with open(file, 'rb') as filename:
            mlp=pickle.load(filename)
            model=self
            for i in range(len(mlp)):
                 for j in range(len(mlp[i])):
                    model.layers[i].nuerons[j].weights = mlp[i][j]
#             return mlp


    

        
        
#     def load_weights(self):
#         with open('model_weights_', 'rb') as filename:
#             mlp=pickle.load(filename)
#             model=self
#             for i in range(len(mlp)):
#                 for j in range(len(mlp[i])):
#                     model.layers[i].nuerons[j].weights = mlp[i][j]
#             return mlp
        
    def addLayer(self,total_nuerons,activation):
        layers_index=len(self.layers)
        L=Layer(self.n_weight[layers_index],total_nuerons,activation)
        self.layers.append(L)
        self.n_weight.append(total_nuerons)   # no of weights in next layer is going to equal to total_nuerons of current layer
    
    def calculate_MSE(self,error_list):
        sum_row=0
        for row in range(len(error_list)):
            sum_col=0
            for col in range(len(error_list[row])):
                sum_col = sum_col + error_list[row][col]*error_list[row][col]
            sum_row = sum_row+ sum_col/len(error_list[0])
        return sum_row/len(error_list)
    
    def train(self,input_x,input_y):
        self.predict(input_x)
        self.train_error=self.calculate_error(self.output,input_y)
        self.calculate_gradient_and_delta_w(input_x)
        self.update_weights()
    
    def predict(self,input_x):
        inp=input_x[:]
        for i in range(len(self.layers)):
            self.layers[i].calculate_output(inp)
            inp=self.layers[i].output
        self.output=self.layers[len(self.layers)-1].output 
        
    def calculate_error(self,modal_result,ground_truth):    
        return [ground_truth[i]-modal_result[i] for i in range(len(modal_result))]
    
    #being called by fit to show error
    def plot_error(self,epoch,train_error, val_error):
        
        plt.plot(epoch,train_error, 'b',label="Train Loss")
        
        if(len(val_error)!=0):
            plt.plot(epoch,val_error, 'r',label="Val Loss")
        plt.legend()
    
    
    #being called inside fit to show error
    def show_mse_error(self,train_MSE,epoch_no,val_MSE=[]):
        print("="*110)
        print("Epoch",(epoch_no+1),": ",end='')
        print("Train Error=",train_MSE[len(train_MSE)-1],end='')
        
        print(" "*30,end='')
        
        if(val_MSE != [] and len(val_MSE)!=0):
            print("Validation Error=",val_MSE[len(val_MSE)-1])
            
    
    #being called by user
    def fit(self,train_data,train_target,epoch=0,val_data=[],val_target=[]):
        
         
        if(not(isinstance(epoch,int)) or epoch==0):
            return "Please specify the no of epochs as your 3rd parameter inside fit mathod"
        
        self.train_MSE=[]
        self.val_MSE=[]
        self.total_train_errors=[]
        for epoch_no in range(epoch):
        
            for i in range(len(train_data)):
                x=list(train_data.iloc[i])
                y=list(train_target.iloc[i])            
                self.train(x,y)
                self.total_train_errors.append(self.train_error)

            temp_mse=self.calculate_MSE(self.total_train_errors)
            self.train_MSE.append(temp_mse)

            if(len(val_data) != 0 and len(val_target) != 0): 
                self.total_val_errors=[]
                for i in range(len(val_data)):
                    x=list(val_data.iloc[i])
                    y=list(val_target.iloc[i])
                    self.predict(x)
                    self.total_val_errors.append(self.calculate_error(self.output,y))

                temp_mse=self.calculate_MSE(self.total_val_errors)
                self.val_MSE.append(temp_mse)
                
                if(len(self.val_MSE)!=0 and temp_mse<min(self.val_MSE)):
                    self.save_weights_and_params()
                    

            self.show_mse_error(self.train_MSE,epoch_no,self.val_MSE)
            
            #shuffle train_data and target indexes after each epoch to make it more robust       
#             x=list(range(len(train_data)))
#             random.shuffle(x)
#             train_data=train_data.iloc[x]
#             train_target=train_target.iloc[x]
            ######################################
            
        self.plot_error(range(len(self.train_MSE)),self.train_MSE,self.val_MSE)
               
    
    def update_weights(self):
        
        for layer in self.layers:
            for nueron in layer.nuerons:
                for k in range(len(nueron.weights)):
                    nueron.weights[k]=nueron.weights[k]+nueron.delta_weights[k]
    
    def calculate_gradient_and_delta_w(self,input_x):
    
        Layer2_delta_weights=[]
        grade_mul_weights=[]


        no_of_layers=len(self.layers)
        i=1
        while(no_of_layers-i>=0):

            if(i==1):    # handle last layer differently

                last = no_of_layers-i
                nuerons=self.layers[last].nuerons
                for j in range(len(nuerons)):
                    nuerons[j].grad=1*self.train_error[j]

                    #print("Grad=",len(nuerons[j].weights)
                    
                    #calculating deta w for last layer  Hk*Grad*Learning_Rate
                    delta=[]
                    for k in range(len(nuerons[j].weights)-1):                 #run for number of nuerons= len(weights)-1 bcs last weight belong to bias
                        delta.append(self.layers[last-1].output[k] * nuerons[j].grad * learning_rate + momentum*nuerons[j].delta_weights[k])  #k is representing kth nueron of last. weights[k] is weight coming from kth nueron of last layer
                    
                    last_weight_index=len(nuerons[j].weights)-1
                    delta.append(nuerons[j].grad*learning_rate + momentum*nuerons[j].delta_weights[last_weight_index])
                    nuerons[j].delta_weights=delta
                    
            else:
                
                
                #print("Hindden Layer")
                
                output=[]
                if(no_of_layers-i==0):
                    output=input_x[:]
                else:
                    output=self.layers[last-1].output[k]
                
                current = no_of_layers-i
                nuerons_h=self.layers[current].nuerons            # nuerons_h is current layer
                for j in range(len(nuerons_h)):
                    nuerons_hx1 =self.layers[current+1].nuerons   #nuerons_hx1 is next layer
                    summ=0
#                    str1=""
                    for k in range(len(nuerons_hx1)):
                        summ = summ+(nuerons_hx1[k].weights[j]*nuerons_hx1[k].grad)   #j is representing current hidden layer nuron index. weights[j] means weight coming jth nueron of current layer
                        
                        
#                         str1=str1+str(nuerons_hx1[k].weights[j])+"*"+str(nuerons_hx1[k].grad)+"+"
#                     str1=str1+str(nuerons_h[j].Hk)+"*"+str(1-nuerons_h[j].Hk)
#                     print(str1)
                    
                    nuerons_h[j].grad=summ*nuerons_h[j].Hk*(1-nuerons_h[j].Hk)*learning_rate
                    
                    delta=[]
                    
                    for k in range(len(nuerons_h[j].weights)-1):                 #run for number of nuerons= len(weights)-1 bcs last weight belong to bias
                        delta.append(output[k]*nuerons_h[j].grad*learning_rate + momentum*nuerons_h[j].delta_weights[k])
                    
                    last_weight_index=len(nuerons_h[j].weights)-1      # this index corresponds to bias
    
                    delta.append(nuerons_h[j].grad*learning_rate + momentum*nuerons_h[j].delta_weights[last_weight_index])
                    nuerons_h[j].delta_weights=delta
                            
                    
            i=i+1
            
            
        


            
class Layer:    
    def __init__(self,input_size,total_nuerons,activation):
        self.output=[]
        self.activation=activation
        self.nuerons=[Nueron(input_size) for i in range(total_nuerons)]        
    def calculate_output(self,input_x):
        self.output=[]
        for i in range(len(self.nuerons)):
            self.nuerons[i].calculate_output(input_x,self.activation)
            self.output.append(self.nuerons[i].Hk)  
            
class Nueron:
    def __init__(self,input_size):
        self.weights=[random.random() for i in range(input_size+1)]  #input_size  + 1(weight for bias)
        self.delta_weights=[0]*(input_size+1)
        self.Vk=None
        self.Hk=None
        self.grad=None
        
    def calculate_output(self,input_x,activation):          
        self.Vk=sum([self.weights[i]*input_x[i] for i in range(len(input_x))]) + self.weights[len(input_x)]
        if(activation=='linear'):
            self.Hk=self.Vk
        if(activation=='sigmoid'): 
            self.Hk=sigmoid(self.Vk)        

