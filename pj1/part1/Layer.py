import numpy as np
from Utils import sigmoid, sigmoid_derivation, softmax


class Layer(object):
    def __init__(self, input_size, output_size, random_range=0.15, 
                 last_flag=False, task_kind="Classifier"):
        self.input_data = np.zeros((input_size, 1))
        self.bias = np.random.normal(loc=0.0, scale=random_range, size=(output_size, 1))
        # print(np.shape(self.bias))
        self.weight = np.random.normal(loc=0.0, scale=random_range, size=(input_size, output_size))
        self.sum_data = np.zeros((output_size, 1)) # before activation
        self.output_data = np.zeros_like(self.sum_data)
        
        self.delta_batch_weight = np.zeros_like(self.weight)
        self.delta_batch_bias = np.zeros_like(self.bias)
        self.batch_num = 0 # an iter in one batch
        self.last_flag =last_flag
        assert task_kind=="Classifier" or task_kind=="Regression", \
            " ** Task Kind Error! You could only choose one from Classifier and Regression"
        self.tast_kind = task_kind

    def forward(self, raw_input):
        assert raw_input.shape == self.input_data.shape, " ** BPLayer input size ERROR! \n"
        
        self.input_data = raw_input
        self.sum_data = self.weight.T.dot(self.input_data) + self.bias # Wx+b
        
        if self.last_flag:
            if self.tast_kind == "Classifier":
                self.output_data = softmax(self.sum_data) # logits
            else : # Regression
                self.output_data = self.sum_data
        else:
            self.output_data = sigmoid(self.sum_data)
        return self.output_data

    def backward(self, loss):
        if self.last_flag and self.tast_kind == "Classifier":
            public_delta = loss # softmax
        else:
            public_delta = loss * sigmoid_derivation(self.sum_data) 
            
        weight_gradient = self.input_data.dot(public_delta.T)
        bias_gradient = public_delta
        
        self.delta_batch_weight -= weight_gradient
        self.delta_batch_bias -= bias_gradient
        self.batch_num += 1
        backward_loss = self.weight.dot(public_delta)

        return backward_loss
    
    def update_weight(self, lr):
        delta_weight = lr * self.delta_batch_weight / self.batch_num
        delta_bias = lr * self.delta_batch_bias / self.batch_num
        self.weight += delta_weight
        self.bias += delta_bias

        self.batch_num = 0
        self.delta_batch_weight = np.zeros_like(self.weight)
        self.delta_batch_bias = np.zeros_like(self.bias)
    
    def get_weight(self):
        return self.weight
    
    def set_weight(self, w):
        self.weight = w
    
    def get_bias(self):
        return self.bias
    
    def set_bias(self, b):
        self.bias = b
    
    def init_weight(self, w):
        self.weight = w

    def init_bias(self, b):
        self.bias = b
        