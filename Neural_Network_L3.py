import Ready_Data as rd
import NeuralNetwork as nn
import numpy as np



passes = range(0, 20000, 100)
print(passes)
train_accuracies = []  # Store accuracy values here
train_losses = []  # Store loss values here
validation_accuracies = []
validation_losses = []
#-----------------------------DATA INITIALIZER------------------------#
X,y =rd.initialize(r"data.json")            
print(X.shape)
y = np.array(y)

X_test, y_test = rd.initialize(r"data.json")
y_test = np.array(y_test)
#--------------------------------Declarations--------------------------#
Layer1 = nn.Layer(63,16)
ReLu_activation = nn.Activation_ReLU()
ReLu_activation2 = nn.Activation_ReLU()
Layer2 = nn.Layer(16,8)
Layer3 = nn.Layer(8,4)                     
Softmax_activation = nn.Activation_Softmax()
loss_activation = nn.Softmax_and_Cross_Categroy_Entropy_Loss()
optimizer = nn.Optimizer_SGD()

Layer1.forward(X_test)
ReLu_activation.forward(Layer1.output)
Layer2.forward(ReLu_activation.output)
ReLu_activation2.forward(Layer2.output)
Layer3.forward(ReLu_activation2.output)
loss_test = loss_activation.forward(Layer2.output, y_test)
predictions_test = np.argmax(loss_activation.output, axis=1)
if len(y_test.shape) == 2:
    y_test = np.argmax(y_test, axis=1)
accuracy_test = np.mean(predictions_test == y_test)


for each_pass in range(900):
    #----------------forward pass----------------#
    Layer1.forward(X)
    ReLu_activation.forward(Layer1.output)
    Layer2.forward(ReLu_activation.output) 
    ReLu_activation2.forward(Layer2.output)
    Layer3.forward(ReLu_activation2.output)
    data_loss = loss_activation.forward(Layer3.output,y)
    regularization_loss = loss_activation.loss.regularization_loss(Layer1) + loss_activation.loss.regularization_loss(Layer2) + loss_activation.loss.regularization_loss(Layer3)      
    loss = data_loss + regularization_loss
    #--------------------Loss---------------# 
    prediction = np.argmax(loss_activation.output,axis=1)
    print(prediction)
    if(len(y.shape) == 2):
       y = np.argmax(y,axis=1)
    accuracy = np.mean(prediction == y)
    y_true = np.argmax(y, axis=1) if len(y.shape) == 2 else y
    accuracy = np.mean(prediction == y_true)

       
    #---------------checking whats hapenning ----------- #
    if (each_pass%100 == 0):
        print(f'Pass ID : {each_pass}' + f'acc :  {accuracy}' + f'loss : {loss}')
    if (each_pass % 100 == 0):
        train_accuracies.append(accuracy)
        train_losses.append(loss)
        validation_accuracies.append(accuracy_test)
        validation_losses.append(loss_test)

    #---------------------backward pass starts here---------------------#
    loss_activation.backward(loss_activation.output,y)
    Layer3.backward(loss_activation.dinputs)
    ReLu_activation2.backward(Layer3.dinputs)
    Layer2.backward(ReLu_activation2.dinputs)
    ReLu_activation.backward(Layer2.dinputs)
    Layer1.backward(ReLu_activation.dinputs)

    #------------ calling optimizer ------ #
    optimizer.preupdate()
    optimizer.update_network(Layer1)
    optimizer.update_network(Layer2)
    optimizer.update_network(Layer3)
    optimizer.postupdate()

#-------------------Testing----------------------------#
    Layer1.forward(X_test)
    ReLu_activation.forward(Layer1.output)
    Layer2.forward(ReLu_activation.output)
    ReLu_activation2.forward(Layer2.output)
    Layer3.forward(ReLu_activation2.output)
    loss_test = loss_activation.forward(Layer3.output, y_test)
    predictions_test = np.argmax(loss_activation.output, axis=1)
    if len(y_test.shape) == 2:
        y_test = np.argmax(y_test, axis=1)
    accuracy_test = np.mean(predictions_test == y_test)



np.savez("openclose_model_weights.npz", 
         Layer1_weights=Layer1.weights, Layer1_biases=Layer1.biases,
         Layer2_weights=Layer2.weights, Layer2_biases=Layer2.biases,
         Layer3_weights=Layer3.weights, Layer3_biases=Layer3.biases
)
print("Model weights and biases saved successfully!")