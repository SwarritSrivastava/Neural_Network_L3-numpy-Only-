import numpy as np
import NeuralNetwork as nn
import cv2 
import mediapipe as mp



Layer1 = nn.Layer(63,16,l2_weightregularizer = 5e-4,l2_biasregularizer = 5e-4)
ReLu_activation = nn.Activation_ReLU()
ReLu_activation2 = nn.Activation_ReLU()
Layer2 = nn.Layer(16,8)
Layer3 = nn.Layer(8,4)                     
Softmax_activation = nn.Activation_Softmax()



with np.load("openclose_model_weights.npz") as data:
    Layer1.weights, Layer1.biases = data["Layer1_weights"], data["Layer1_biases"]
    Layer2.weights, Layer2.biases = data["Layer2_weights"], data["Layer2_biases"]
    Layer3.weights, Layer3_biases = data["Layer3_weights"], data["Layer3_biases"]
print("Model weights and biases loaded successfully!")

mphandmodul = mp.solutions.hands
hands = mphandmodul.Hands(max_num_hands = 1)
mpdraw = mp.solutions.drawing_utils

vid = cv2.VideoCapture(0)
while(1):
    success, frame = vid.read()

    if not success:
        break
    frame = cv2.flip(frame , 1)
    RGBframe = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(RGBframe)
    if result.multi_hand_landmarks :
        for handLm in result.multi_hand_landmarks:
            landmarks = []
            for lm in handLm.landmark:
                landmarks.append([lm.x, lm.y, lm.z])
            landmarks = np.array(landmarks)
            landmarks -= landmarks[0]
            landmarks = landmarks.flatten()
            # print(landmarks.shape)
            Layer1.forward(landmarks)
            ReLu_activation.forward(Layer1.output)
            Layer2.forward(ReLu_activation.output)
            ReLu_activation2.forward(Layer2.output)
            Layer3.forward(ReLu_activation2.output)
            Softmax_activation.forward(Layer3.output)
            predictions = np.argmax(Softmax_activation.output)
            if(predictions == 1):
                prediction_label = "close"
            elif(predictions == 2):
                prediction_label = "thumbs_up"
            elif(predictions == 3):
                prediction_label = "fcuk_off"
            elif(predictions == 0):
                prediction_label = "open"   
            cv2.putText(frame, f"Prediction: {prediction_label}", (50, 50), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            mpdraw.draw_landmarks(frame, handLm, mphandmodul.HAND_CONNECTIONS)

    cv2.imshow("video",frame)
    if cv2.waitKey(10) & 0xFF == 27:  # Press 'Esc' to exit
        break

vid.release()
cv2.destroyAllWindows()


