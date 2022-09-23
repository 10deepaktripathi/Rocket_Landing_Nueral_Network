
#class NeuralNetHolder:

#    def __init__(self):
#        super().__init__()
		# load anything you need, initialize anything you need.
    
#    def predict(self, input_row):
        # WRITE CODE TO PROCESS INPUT ROW AND PREDICT X_Velocity and Y_Velocity
        
		#Input_row is string like: x_distance,y_distance
		#	Split string, normalize values, pass them through the feedforward process and denormalize the results


import Game_ML as mlp
class NeuralNetHolder:

    def __init__(self):
        super().__init__()

    
    def predict(self, input_row):
        # WRITE CODE TO PROCESS INPUT ROW AND PREDICT X_Velocity and Y_Velocity
        
        
#        l1=[[5.8319, 3.8612], [-1.2609, 8.1426], [-7.7558, 0.73511], [6.378, 3.5166], [4.1042, -6.2673] ,[2.5343, 6.6773]]
#        b1=[-6.6959, -1.6714, -0.71723, -1.8366, 0.99876, 5.8877]

#        l2=[[0.33107, -0.052528, -0.24549, -0.21267, -0.090469, -0.71713], [-0.62526 ,-0.67848, -0.74549, 0.16754, -0.72446, 0.06547]]
#        b2=[0.95474, 0.97383]

  
        
        
        
        
        weight_file="model_weights_1607277647"
        
        #weight_file="model_weights_1607025925"
        
        data=[float(n) for n in input_row.split(',')]        
        data[0]=(data[0]-(-788.6065318))/(760.0994972-(-788.6065318))
        data[1]=(data[1]-65.02018976)/(861.3191939999999-65.02018976)
        
        no_hidden_layer_nuerons=6
        model=mlp.MLP(2)
        model.addLayer(no_hidden_layer_nuerons,'sigmoid')
        model.addLayer(2,'linear')
        model.load_weights(weight_file)
        
        
#        for i in range(len(l1)):
#            l1[i].extend([b1[i]])
#            model.layers[0].nuerons[i].weights=l1[i]
        
#        for i in range(len(l2)):
#            l2[i].extend([b2[i]])
#           model.layers[1].nuerons[i].weights= l2[i]
        
        
        
        
        model.predict(list(data))
        return (model.output[1],model.output[0])



		
		#return a list like --> [X_Velocity, Y_Velocity]
