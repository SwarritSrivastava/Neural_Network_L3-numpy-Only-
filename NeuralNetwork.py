import numpy as np

class Layer:
    def __init__(self, n_inputs, n_neurons,l1_weightregularizer = 0,l1_biasregularizer = 0,l2_weightregularizer = 0 ,l2_biasregularizer = 0):
        self.weights = 0.01 * np.random.randn(n_inputs, n_neurons)               #these are weights of connections going 'to the neuron'
        self.biases = np.zeros((1,n_neurons))                                    #these are biases of neurons in the layer.
        self.l1_weights_regularizer = l1_weightregularizer
        self.l1_biases_regularizer = l1_biasregularizer
        self.l2_weights_regularizer = l2_weightregularizer
        self.l2_biases_regularizer = l2_biasregularizer
    def forward(self,inputs):
        self.inputs = inputs
        self.output = np.dot(inputs, self.weights) + self.biases
    def backward(self,dvalues):
        self.dweights = np.dot(self.inputs.T, dvalues)
        self.dbiases = np.sum(dvalues, axis=0, keepdims=True)

        if(self.l1_weights_regularizer > 0):
            dL1 = np.ones_like(self.weights)
            dL1[self.weights < 0] = -1
            self.dweights += self.l1_weights_regularizer * dL1
        if(self.l2_weights_regularizer > 0):
            self.dweights += 2 * self.l2_weights_regularizer * self.weights
        self.dinputs = np.dot(dvalues,self.weights.T)
        if(self.l1_biases_regularizer > 0):
            dL1 = np.ones_like(self.biases)
            dL1[self.biases < 0] = -1
            self.dbiases += self.l1_biases_regularizer * dL1
        if(self.l2_biases_regularizer > 0):
            self.dbiases += 2 * self.l2_biases_regularizer * self.biases

        self.dinputs = np.dot(dvalues,self.weights.T)

class Activation_ReLU:
    def forward(self,inputs):
        self.inputs = inputs
        self.output = np.maximum(0,inputs)
    def backward(self,dvalues):
        self.dinputs = dvalues.copy()
        self.dinputs[self.inputs <= 0] = 0         #relu's derivative   

class Activation_Softmax:
    def forward(self,inputs):
        exp_values = np.exp(inputs - np.max(inputs , axis = 1, keepdims=True))
        probabilities = exp_values/np.sum(exp_values,axis=1,keepdims=True)
        self.output = probabilities
    def backward(self,dvalues):
        self.dinputs = np.empty_like(dvalues)
        for index, (single_output, single_dvalues) in enumerate(zip(self.output,dvalues)):
            single_output = single_output.reshape(-1,1)
            jacobian_matrix = np.diagflat(single_output) - np.dot(single_output,single_output.T)                              
            self.dinputs = np.dot(jacobian_matrix,single_dvalues)
class Loss:
    def regularization_loss(self,layer):
        
        regularization_loss = 0

        if layer.l1_weights_regularizer > 0:
            regularization_loss += layer.l1_weights_regularizer * np.sum(np.abs(layer.weights))
        if layer.l2_weights_regularizer > 0:
            regularization_loss += layer.l2_weights_regularizer * np.sum(layer.weights ** 2)
        if layer.l1_biases_regularizer > 0:
            regularization_loss += layer.l1_biases_regularizer * np.sum(np.abs(layer.biases))
        if layer.l2_biases_regularizer > 0:
            regularization_loss += layer.l2_biases_regularizer * np.sum(layer.biases ** 2)
        return regularization_loss
    
    def calculate(self,output,y):
        sample_losses = self.forward(output, y)
        data_loss = np.mean(sample_losses)  
        return data_loss
    
class Loss_CCE(Loss):
    def forward(self,y_pred,y_true):
        samples = len(y_pred)
        y_pred_clipped = np.clip(y_pred, 1e-7, 1 - 1e-7)
        if(len(y_true.shape) == 1):
            correct_confidences  =  y_pred_clipped[range(samples), y_true]
        elif(len(y_true.shape) == 2):
            correct_confidences = np.sum(y_pred_clipped*y_true,axis=1)
        
        negative_log_likelihoods = -np.log(correct_confidences)
        return negative_log_likelihoods
    def backward(self,dvalues,y_true):
        samples = len(dvalues)
        labels = len(dvalues[0])
        if(len(y_true.shape) == 1):
            y_true = np.eye(labels)[y_true]
        self.dinputs = -y_true/dvalues
        self.dinputs = self.dinputs/samples


#to make the code faster we combine loss and softmax since they are quite literally the end step

class Softmax_and_Cross_Categroy_Entropy_Loss():
    def __init__(self):
        self.activation = Activation_Softmax()
        self.loss = Loss_CCE()
    def forward(self,inputs,y_true):
        self.activation.forward(inputs)
        self.output = self.activation.output
        return self.loss.calculate(self.output,y_true)
    def backward(self,dvalues,y_true):
        samples = len(dvalues)
        if len(y_true.shape) == 2:
            y_true = np.argmax(y_true, axis = 1)
        self.dinputs = dvalues.copy()
        self.dinputs[range(samples),y_true] -= 1
        self.dinputs = self.dinputs/samples

#---------------------------Optimizer---------------------#

class Optimizer_SGD:
    def __init__(self,learning_rate = 0.5,decay = 0.001, momentum = 0.9):
        self.learning_rate = learning_rate
        self.current_learning_rate = learning_rate
        self.decay = decay
        self.momentum = momentum
        self.iterations = 0
    def preupdate(self):
        if self.decay:
            self.current_learning_rate = self.learning_rate * (1/(1+self.decay * self.iterations))
    def update_network(self,layer):
        layer.weights += -self.learning_rate * layer.dweights
        layer.biases += -self.learning_rate * layer.dbiases
    def postupdate(self):
        self.iterations += 1

