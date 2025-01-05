# Assignment execution

## Start simple

- First tests with simple problem with integrator dinamics and simple cost
- No conditions on control inputs
- No extra costs on control inputs magnitude

How to test to see if the control is good? 
- Plot solution, shoult be in global minimum


## Tests
- Created new dataset with N=50 dt=0.1 (dataset_v2.csv)

With learning rate 1e-3 those where the results of te loss function

Epoch 1/100, Loss: 380.5793
Epoch 2/100, Loss: 1231.2795
Epoch 3/100, Loss: 504.2175
Epoch 4/100, Loss: 725.8030
Epoch 5/100, Loss: 616.8210
Epoch 6/100, Loss: 826.5962
Epoch 7/100, Loss: 272.1107
Epoch 8/100, Loss: 709.0292
Epoch 9/100, Loss: 858.2076
Epoch 10/100, Loss: 678.6498
Epoch 11/100, Loss: 1077.2062
Epoch 12/100, Loss: 458.2193
Epoch 13/100, Loss: 676.9474
Epoch 14/100, Loss: 551.6497
Epoch 15/100, Loss: 817.5675
Epoch 16/100, Loss: 593.0043
Epoch 17/100, Loss: 684.4852

Decreasing it to 1e-2 the loss decreased in a sustantial way