# python nnlayers.py > /dev/null
# is a good command. it stops print().
# so it runs too much faster than that with print().
# python nnlayers.py > ./a.txt
# this also works similarly,
# but saves output of all print() in file for inspection.


#this neural network has the capacity of
#several layers
#input of m features
#output of n features
#work in stochastic mode
#i.e. we back propagate error after each example input

import random
import matplotlib.pyplot as plt
import math
import pdb

#target variable should be a list

learning_rate = 0.01

#to predict sin(), we can't use relu
#in output layer neuron, because
#it always suppress negative values.
#moreover, since its output is always between 0 and 1
#the network can never learn negative sine values.

#ReLU is typically chosen
#only for hidden layer neurons,
#never for outer neurons.

#sop = sum of products = weighted_sum
def relu_func(sop):
    if sop > 0:
        return sop
    else:
        return 0

def relu_deriv(sop):
    if sop > 0:
        return 1
    else:
        return 0

#for sin() curve prediction,
#sigmoid is not so good, it gives values in [0, 1]
#since its output is always between 0 and 1
#the network can never learn negative sine values.
#and so can not be used in output neuron also.
#if sop < -709 or > 709, sigmoid() value becomes
#very high and float overflow happens, python stops!
def sigmoid_func(sop):
    return 1.0 / (1.0+math.exp(-1.0*sop))

def sigmoid_deriv(sop):
    #return sigmoid_func(sop)*(1.0-sigmoid_func(sop))
    s = sigmoid_func(sop)
    return s * (1.0 - s)

#for sin() curve prediction,
#tanh is better, it gives values in [-1, 1]
#and so can be used in output neuron
#as well as, network can learn negative values of sin()
def tanh_func(sop):
    return math.tanh(sop)

def tanh_deriv(sop):
    return 1 - math.tanh(sop)**2

#z and sop are same thing
def softmax_func(sop_list):
    #softmax= e^z_i / Sum all e^z_j

    e_power_z_list = []
    for sop in sop_list:
        e_power_z = math.exp(1.0*sop)
        e_power_z_list.append(e_power_z)

    sum_of_all_e_power_z = 0
    for e_power_z in e_power_z_list:
        sum_of_all_e_power_z = sum_of_all_e_power_z + e_power_z

    probability_output_list = []
    for e_power_z in e_power_z_list:
        probability_output = e_power_z / sum_of_all_e_power_z
        probability_output_list.append(probability_output)

    return probability_output_list


#always normalize input to neurons
'''
feature scaling is of two types:
normalization and standardization.

normalization:
-uses min and max values.
-new values are within [0,1] or [-1, +1]
-affected by outliers, so remove outliers first.
-Xnew=(X-Xmin)/(Xmax-Xmin)

standardization:
-uses mean and stddev values.
-new values are NOT within [0,1] or [-1, +1]
-less affected by outliers
-Xnew=(X-mean)/stddev
'''
#convert 0 to 6.28 to 0 to 1
def normalize_x_of_sin(x):
    normalized_x = (x-0)/(6.8-0) # (x-min)/(max-min)
    return normalized_x

def denormalize_x_of_sin(normalized_x):
    x = normalized_x*(6.8-0) + 0
    return x


'''
how error gets back propagated in neural network:

suppose we have
input->[hidden layer]->[output layer]->output

if we zoom it,
input->[(mul&sum)->(activation)]->[(mul&sum)->(activation)]->output

note that each () represents a mathematical function.

now let's think about error back propagation.
1. first we need to calculate error to be given to [output layer].
2. then [output layer] calculates error to be given to [hidden layer].
   and updates its weights.
3. then [hidden layer] calculates error to be given to actual inputs(this is absurd, so these errors are not used and ignored).
   and updates its weights.

to calculate error to be given to [output layer],
we add (loss function) and calculate loss
input->[(mul&sum)->(activation)]->[(mul&sum)->(activation)]->output->(loss function)->loss

1. now we calculate error for [output layer] as:
error for output layer = d(loss)/d(activation output)

2. now [output layer] calculates error to be given to [hidden layer]:
error for hidden layer = error for output layer * d(activation output)/d(sum of products) * d(sum of products)/d(input to output layer)
also [output layer] calculates weights to be adjusted:
weight adjustment for output layer = error for output layer * d(activation output)/d(sum of products) * d(sum of products)/d(weight of output layer)

3. now [hidden layer] calculates error to be given to actual inputs:
error for actual inputs = error for hidden layer * d(activation output)/d(sum of products) * d(sum of products)/d(input to hidden layer)
also [hidden layer] calculates weights to be adjusted:
weight adjustment for hidden layer = error for hidden layer * d(activation output)/d(sum of products) * d(sum of products)/d(weight of hidden layer)

this way error gets back propagated.

--------------------------

formulas of activation functions,
formulas of loss functions,
formulas of their derivatives for back propagation:

activation functions:
[z=sop]
1. sigmoid= 1 / (1+e^-z)
2. softmax= e^z_i / Sum all e^z_j
3. tanh= (e^z - e^-z) / (e^z + e^-z)
4. relu= z if z>0; 0 otherwise

d(activation output)/d(sum of products):
[z=sop]
1. d(sigmoid)/d(z)= sigmoid(z)*(1-sigmoid(z))
2. d(softmax)/d(z)= it is complicated
3. d(tanh)/d(z)= 1 - tanh^2(z)
4. d(relu)/d(z)= 1 if z>0; 0 otherwise

loss functions:
[y=actual/target]
[y'=predicted/activation output]
1. bce= -[y.log(y') + (1-y).log(1-y')]
2. cce= - Sum all y.log(y')
3. mse= (1/2)(y-y')^2

d(loss)/d(activation output):
[y=actual/target]
[y'=predicted/activation output]
1. d(bce)/d(y')= it is complicated
2. d(cce)/d(y')= it is complicated
3. d(mse)/d(y')= y'-y

now some interesting things:
[z=sop]
[y=actual/target]
[y'=predicted/activation output]
1. d(bce)/d(y') * d(sigmoid)/d(z) = y'-y
2. d(cce)/d(y') * d(softmax)/d(z) = y'_i - y_i
this is why we always use sigmoid-bce and softmax-cce combinations.

'''

#various valid configurations of output layer:
#------------------------

#1. binary classification,
#output neurons=1,
#activation=sigmoid,
#loss=bce(binary cross entropy)

#2. multiclass classification(only one class=1, rest all classes=0),
#output neurons=n,
#activation=softmax,
#loss=cce(categorical cross entropy)

#3. single regression,
#output neurons=1,
#activation=tanh
#loss=mse

#4. multiple regression,
#output neurons=n,
#activation=tanh
#loss=mse

#5. multilabel classification(multiple classes=1, other classes=0),
#output neurons=n,
#activation=sigmoid,
#loss=bce

#[relu is good for hidden layers only, not good for output layer]
#--------------------------

class neuron:
    def __init__(self, activation, kind):
        self.weight_list = []
        self.input_list = []
        self.output = 0
        self.sop = 0
        #we have to use sop in derivative of
        #activation function, not the output
        #which we get after applying activation.
        self.activation = activation
        self.kind = kind #'hidden' or 'outer'

    def __call__(self, input_list):
        #prepare input_list
        self.input_list = []
        self.input_list.append(1) #bias = 1*some_weight
        self.input_list.extend(input_list)

        #Activation Function    Recommended Initial Weight Range                                Why?
        #Sigmoid                Small, around [-1, 1] or smaller                                Prevents saturation (where gradients vanish)
        #Tanh                   Small, centered around 0, e.g. [-1, 1]                          Zero-centered, so works better with weights in [-1, 1]
        #ReLU                   [0, positive] or slightly negative allowed, but avoid zero      Too many zeros → dead neurons (especially if all weights are ≤ 0)

        #Activation     Init Range                                  Notes
        #Sigmoid        [-1, 1] or [-0.5, 0.5]                      Prevent saturation, If weights are too big: sigmoid(z) ≈ 1 or 0 (saturated), Derivative ≈ 0, Gradient vanishes → no learning
        #Tanh           [-1, 1]                                     Zero-centered helps
        #ReLU           positive, avoid all-zero or all-negative    Helps neurons activate, If weights are ≤ 0: Output is always zero → no gradient → dead neuron

        if len(self.weight_list) == 0:
            for i in range(len(self.input_list)):
                random_value = -0.5 + random.random() #random() gives all +ve, 0 to 1
                if random_value == 0:
                    random_value = 0.0001
                self.weight_list.append(random_value) #random_value in [-0.5, 0.5]

        print("   neuron ", id(self), " input_list ", [round(num, 4) for num in self.input_list], " weight_list ", [round(num, 4) for num in self.weight_list])

        #weighted sum
        weighted_sum = 0
        for input_i, weight_i in zip(self.input_list, self.weight_list):
            weighted_input = input_i*weight_i
            weighted_sum += weighted_input

        #save sop
        self.sop = weighted_sum

        #non linear
        if self.activation == 'softmax':
            #for softmax, layer computes softmax
            #so neuron shouldn't do anything
            output = weighted_sum
        #for tanh, relu and sigmoid, neuron computes output here
        elif self.activation == 'tanh':
            output = tanh_func(weighted_sum)
        elif self.activation == 'relu':
            output = relu_func(weighted_sum)
        elif self.activation == 'sigmoid':
            output = sigmoid_func(weighted_sum)

        self.output = output #save output
        return self.output

    def update_self(self, error_proportion):
        #first back prop error
        output_err_prop_list = [0]*(len(self.input_list)-1) #first one bias
        for i in range(1, len(self.input_list)): #first one bias
            if self.activation == 'softmax':
                #for self.activation == softmax,
                #always self.kind == 'outer'
                output_err_prop_list[i-1] = error_proportion*self.weight_list[i] #i is 1 here, 0 is for bias
                #we don't multiply softmax_deriv(self.sop) because
                #for outer layer neuron with softmax, error_proportion = d(cce)/d(y') * d(softmax)/d(z)
            elif self.activation == 'tanh':
                output_err_prop_list[i-1] = error_proportion*tanh_deriv(self.sop)*self.weight_list[i] #i is 1 here, 0 is for bias
            elif self.activation == 'relu':
                output_err_prop_list[i-1] = error_proportion*relu_deriv(self.sop)*self.weight_list[i] #i is 1 here, 0 is for bias
            elif self.activation == 'sigmoid':
                if self.kind == 'hidden':
                    output_err_prop_list[i-1] = error_proportion*sigmoid_deriv(self.sop)*self.weight_list[i] #i is 1 here, 0 is for bias
                elif self.kind == 'outer':
                    output_err_prop_list[i-1] = error_proportion*self.weight_list[i] #i is 1 here, 0 is for bias
                    #we don't multiply sigmoid_deriv(self.sop) because
                    #for outer layer neuron with sigmoid, error_proportion = d(bce)/d(y') * d(sigmoid)/d(z)

                #we put rhs value in i-1 of error propagation list
                #because error propagation list has one less number of values than weights
                #recall, first one is bias in weight list

        #then adjust own weights
        weight_adj_list = [0]*len(self.input_list)
        for i in range(len(self.input_list)):
            if self.activation == 'softmax':
                weight_adj_list[i] = error_proportion*self.input_list[i]
                #we don't multiply softmax_deriv(self.sop) because
                #for outer layer neuron with softmax, error_proportion = d(cce)/d(y') * d(softmax)/d(z)
            elif self.activation == 'tanh':
                weight_adj_list[i] = error_proportion*tanh_deriv(self.sop)*self.input_list[i]
            elif self.activation == 'relu':
                weight_adj_list[i] = error_proportion*relu_deriv(self.sop)*self.input_list[i]
            elif self.activation == 'sigmoid':
                if self.kind == 'hidden':
                    weight_adj_list[i] = error_proportion*sigmoid_deriv(self.sop)*self.input_list[i]
                elif self.kind == 'outer':
                    weight_adj_list[i] = error_proportion*self.input_list[i]
                    #we don't multiply sigmoid_deriv(self.sop) because
                    #for outer layer neuron with sigmoid, error_proportion = d(bce)/d(y') * d(sigmoid)/d(z)

            self.weight_list[i] = self.weight_list[i] - learning_rate*weight_adj_list[i] #update weight

        print("   neuron ", id(self), " output gradient ", [round(num, 4) for num in output_err_prop_list], " updated weight_list ", [round(num, 4) for num in self.weight_list])
        return output_err_prop_list

class layer:
    def __init__(self, num_neurons, activation, kind):
        self.neuron_list=[]
        for i in range(num_neurons):
            neuron_i = neuron(activation, kind) #create neuron
            self.neuron_list.append(neuron_i) #store neuron
        self.input_list = []
        self.output_list = []
        self.activation = activation
        self.kind = kind #'hidden' or 'outer'

    def __call__(self, input_list):
        self.input_list = input_list #save input
        print("  layer ", id(self), " input_list ", [round(num, 4) for num in self.input_list])
        #collect output from neurons
        output_list = []
        for i in range(len(self.neuron_list)):
            neuron_i = self.neuron_list[i] #find neuron
            output = neuron_i(input_list) #call neuron
            output_list.append(output) #append output
        #if activation is softmax
        #do softmax calculation
        #and update outputs
        if self.activation == 'softmax':
            #for softmax, layer computes here
            output_list = softmax_func(output_list)
        else:
            #for tanh, relu and sigmoid, neuron computes output
            #so layer shouldn't do anything
            pass

        self.output_list = output_list #save output
        return output_list

    def update_self(self, error_proportion_list):
        output_err_prop_list = [0]*len(self.input_list)
        print("  layer ", id(self), " input gradient ", [round(num, 4) for num in error_proportion_list])
        for i in range(len(self.neuron_list)):
            neuron_i = self.neuron_list[i] #find neuron
            output = neuron_i.update_self(error_proportion_list[i]) #call neuron
            for i in range(len(output)):
                output_err_prop_list[i] = output_err_prop_list[i] + output[i] #acumulate output
        return output_err_prop_list

class network:
    def __init__(self, num_neurons_in_each_layer_list):
        num_layers = len(num_neurons_in_each_layer_list) #find num layers
        self.layer_list=[]

        #check network validity
        #check hidden layers
        for i in range(num_layers-1):
            num_neurons_in_layer_i = num_neurons_in_each_layer_list[i][0]
            activation_in_layer_i = num_neurons_in_each_layer_list[i][1]
            if num_neurons_in_layer_i < 1:
                print("invalid number of neurons in hidden layer.")
            if ( (activation_in_layer_i == 'tanh')
                or (activation_in_layer_i == 'sigmoid')
                or (activation_in_layer_i == 'relu') ):
                pass
            else:
                print("invalid hidden layer activation function.")
        #check outer layer
        num_neurons_in_layer_i = num_neurons_in_each_layer_list[num_layers-1][0]
        activation_in_layer_i = num_neurons_in_each_layer_list[num_layers-1][1]
        if num_neurons_in_layer_i < 1:
            print("invalid number of neurons in outer layer.")
        if ( (activation_in_layer_i == 'tanh') #1 neuron case=single regression, N neuron case=multiple regression
            or (activation_in_layer_i == 'sigmoid') #1 neuron case=binary classification, N neuron case=multilabel classification
            or (activation_in_layer_i == 'softmax' #1 neuron case=invalid, N neuron case=multiclass classification
                and num_neurons_in_layer_i != 1) ):
            pass
        else:
            print("invalid outer layer activation function.")

        #init
        for i in range(num_layers):
            num_neurons_in_layer_i = num_neurons_in_each_layer_list[i][0]
            activation_in_layer_i = num_neurons_in_each_layer_list[i][1]
            if i != num_layers-1:
                layer_i = layer(num_neurons_in_layer_i, activation_in_layer_i, 'hidden') #create hidden layer
            else:
                layer_i = layer(num_neurons_in_layer_i, activation_in_layer_i, 'outer') #create outer layer
            self.layer_list.append(layer_i) #save it
        self.input_list = []
        self.output_list = []
        self.loss_func = None
        #decide loss function
        if num_neurons_in_each_layer_list[num_layers-1][1] == 'tanh':
            self.loss_func = 'mse'
        elif num_neurons_in_each_layer_list[num_layers-1][1] == 'sigmoid':
            self.loss_func = 'bce'
        elif num_neurons_in_each_layer_list[num_layers-1][1] == 'softmax':
            self.loss_func = 'cce'


    def __call__(self, input_list):
        self.input_list = input_list #save input
        #reuse output list as input list to next layer
        output_list = input_list
        print("")
        print(" network ", id(self), " input ", [round(num, 4) for num in input_list])
        for layer in self.layer_list:
            output_list = layer(output_list)
            print("  layer ", id(layer), " output ", [round(num, 4) for num in output_list])
        self.output_list = output_list #save output
        return output_list

    def update_self(self, target_list):
        print("")
        error_proportion_list = []

        for target, output in zip(target_list, self.output_list):
            if self.loss_func == 'mse':
                error_proportion = output - target
                #this is d(loss)/d(activation output)

                #error_proportion is nothing but error gradient

                #actually mean square error = (1/N)*(target - output)^2
                #but to make calculation simple, we replaced N with 2
                #anyway N is a constant and altering it won't make any harm.
                #further, mse can also be thought of as (1/2)*(output - target)^2.
                #this makes finding derivative easy while not changing value of mse.
                #in the derivative wrt output, we will not get the factor of -1.
                #also derivative will be (1/2)*2*(output - target),
                #which is simply output - target

                #our interest is to bring output equal to target step after step
                #and finally making error = 0

                #error function can't be simply output-target,
                #because it is linear and has no minima,
                #so error has to be a sqare function.

            elif self.loss_func == 'bce':
                #loss function is binary cross-entropy loss.
                #L= −[ y⋅log(y^​) + (1−y)⋅log(1−y^​) ]
                #note: when y=1, L= -y⋅log(y^​);
                #and when y=0, L= −(1−y)⋅log(1−y^​).

                #note that this is NOT d(loss)/d(activation output)
                #d(loss)/d(activation output) of bce is difficult one
                #but d(bce)/d(y') * d(sigmoid)/d(z) is easier, which is y'-y
                #so we set y'-y as error proportion and outer layer/neuron
                #takes care of not multiplying d(sigmoid)/d(z)
                #to this error proportion again inside their logic
                error_proportion = output - target

            elif self.loss_func == 'cce':
                #loss function is categorical cross-entropy loss.
                #loss= - Summation (y_i * log(y'_i))
                #where y_i is true probability or target.
                #and y'_i is the predicted probability or output.
                #note: only one term in Summation remains valid
                #where y_i = 1, rest all terms vanishes becoming 0.

                #note that this is NOT d(loss)/d(activation output)
                #d(loss)/d(activation output) of cce is difficult one
                #but d(cce)/d(y') * d(softmax)/d(z) is easier, which is y'-y
                #so we set y'-y as error proportion and outer layer/neuron
                #takes care of not multiplying d(softmax)/d(z)
                #to this error proportion again inside their logic

                #again note that d(cce)/d(y') * d(softmax)/d(z) = y'_i - y_i
                #which is always a list, because cce/softmax works on multiple outputs
                error_proportion = output - target

            error_proportion_list.append(error_proportion)

        print(" network ", id(self), " input gradient ", [round(num, 4) for num in error_proportion_list])

        for layer in reversed(self.layer_list):
            error_proportion_list = layer.update_self(error_proportion_list)
            print("  layer ", id(layer), " output gradient ", [round(num, 4) for num in error_proportion_list])



#create network
num_neurons_in_each_layer_list = [[8,'tanh'], [4,'tanh'], [1,'tanh']] #nn arch
nn = network(num_neurons_in_each_layer_list)



#training
data_range = 100

#create samples, data points, examples
random_list = [0]*data_range
for i in range(data_range):
    random_list[i] = random.random()*6.28 #input to sin() is 0 to 2*pi=6.28

sine_list = [0]*data_range
for i in range(data_range):
    sine_list[i] = math.sin(random_list[i])



#training
epochs = 1
#precision = 0.0001 #need 99% correct value, 0.01*0.01=0.0001
#precision = 0.001 #need 97% correct value, 0.03*0.03=0.001
#precision = 0.01 #need 90% correct value, 0.1*0.1=0.01

while epochs > 0:

    for i in range(100*1000):
        x = random.random()*100
        x = int(x)

        feature_list = [normalize_x_of_sin(random_list[x])]
        #input to nn is feature list of single sample
        #here feature list has only one feature.
        #we normalized the data before sending it to NN,
        #otherwise weights will be very very big.
        #here not required
        print("")
        print("i = ", i)
        print(" input feature_list ", [round(num, 4) for num in feature_list])

        #back propagation
        target_list = [sine_list[x]]
        #num elements in target list must be equal to
        #num neurons in last layer
        #here we are trying to train sine function
        print(" input target_list ", [round(num, 4) for num in target_list])
        nn_output_list = nn(feature_list)
        nn.update_self(target_list)
        #input("Press Enter to continue...")

    epochs -= 1





#draw predicting curve
x_list = [0]*data_range
y_list = [0]*data_range
for k in range(data_range):
    if k == 0:
        x_list[k] = 0
    else:
        x_list[k] = x_list[k-1]+6.28/data_range #from 0 to pi*2=6.28

    feature_list = [normalize_x_of_sin(x_list[k])]
    print("")
    print(" predict")
    #input to nn is feature list of single example
    #here feature list has only one feature
    nn_output_list = nn(feature_list)

    y_list[k] = nn_output_list[0] # because output is also one value, not list of Ys.
print("")
print(" x_list ", [round(num, 4) for num in x_list])
print("")
print(" y_list ", [round(num, 4) for num in y_list])
plt.plot(x_list, y_list, color='red')




#draw real sine curve
x_list = [0]*data_range
y_list = [0]*data_range
for k in range(data_range):
    if k == 0:
        x_list[k] = 0
    else:
        x_list[k] = x_list[k-1]+6.28/data_range #from 0 to pi*2=6.28
    y_list[k] = math.sin(x_list[k]) #sine
plt.plot(x_list, y_list, color='green')




#plot training data
plt.scatter(random_list, sine_list, color='blue', marker='.')




#show
plt.show()



