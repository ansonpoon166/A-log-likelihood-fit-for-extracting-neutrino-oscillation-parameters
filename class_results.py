#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 17 12:24:25 2020

@author: ansonpoon
"""
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import time
from functions_class import NLL




#%%
# Importing data
#This imports the data
data_link = "https://www.hep.ph.ic.ac.uk/~ms2609/CompPhys/neutrino_data/awp18.txt"
data_to_fit = np.loadtxt(data_link, skiprows=2, max_rows=201)
unoscillated_flux = np.loadtxt(data_link, skiprows=205)

#This assigns the data to two variables
data_to_fit = np.array(data_to_fit)
unoscillated_flux = np.array(unoscillated_flux)

#This initialises the condition of the experiment
E_list = np.linspace(0.025,9.975, num = 200)
L = 295
theta23 = np.pi/4
delta_m_squared= 0.0024

#This creates an object that contains the data and the experimental condition
nll_obj = NLL(unoscillated_flux, data_to_fit, L, E_list)

#%%

plt.figure(1)
plt.title('Experimental data')
plt.xlabel('Energy')
plt.ylabel('Number of events')
plt.bar(E_list, data_to_fit)
plt.show()
#%%
plt.figure(2)
plt.title('Simulated data')
plt.xlabel('Energy')
plt.ylabel('Number of events')
plt.bar(E_list,unoscillated_flux)
plt.show()


#%%
# 3.2 Fit function

#This calculates the oscillation probability corresponding to each energy bin
osc_prob_lst = nll_obj.osc_prob(theta23, delta_m_squared)

plt.figure(3)
plt.plot(E_list, osc_prob_lst)
plt.xlabel('Energy (eV)')
plt.ylabel('Oscillation probability')
plt.title('Oscillation probability as a function of energy')


plt.figure(4)
osc_event_pred = nll_obj.osc_event_pred(theta23, delta_m_squared)
plt.plot(E_list, osc_event_pred)
plt.xlabel('Energy (eV)')
plt.ylabel('Oscillated event rate prediction')
plt.title('Oscillated event rate prediction as a function of energy')

plt.show()

#%%
# 3.3 Likelihood function


theta_23_lst = np.linspace(0, np.pi/2, num =500)
nll_fixed_m = nll_obj.neg_ll(theta_23_lst, delta_m_squared) #An array NLL values for a fixed squared mass difference


plt.figure(5)
plt.plot(theta_23_lst, nll_fixed_m)
plt.xlabel('Mixing angle')
plt.ylabel('Negative log likelihood')
plt.title('NLL as a function of mixing angle')
plt.show()

#%%
# 3.4 para_mini_nll
start_time = time.time()
min_theta_1d, nll_min_1d = nll_obj.para_mini_nll([0.5, 0.6, 1.1], minimise_variable= 'theta')
end_time = time.time()
# min_theta_1d is the value of the theta at the minimum
# nll_min_1d is the value of the NLL at the minimum

time_taken = end_time - start_time
plt.figure(6)
plt.plot(theta_23_lst, nll_fixed_m)
plt.xlabel('Mixing angle')
plt.ylabel('Negative log likelihood')
plt.title('NLL as a function of mixing angle')
plt.scatter(min_theta_1d, nll_min_1d)
plt.show()

print('\n')
print('Parabolic minimiser for mixing angle in one dimenison:\n')
print(f'The minimised NLL: {nll_obj.neg_ll(min_theta_1d, delta_m_squared)}')
print(f'The mixng angle: {min_theta_1d}')
print(f'Time taken: {time_taken} s')

# 3.5 Find accuracy of fit
para_error = nll_obj.error(min_theta_1d, 0.0024, num_variables = 1)
para_error_2 = nll_obj.error_2(min_theta_1d, 0.0024)
print(f'The mixing angle error from the curvature approximation: {para_error}')
print(f'The mixing angle error from the +/- 0.5 approximation: {para_error_2}')

#%%
# Contour plot

x_values = np.linspace(0, 1.5, 400) #theta
m_values = np.linspace(0.00,0.005, num = 400) #delta_m_squared
delta_m_squared = 2.4 * 10**(-3)
L = 295

#This creates matrices of theta values and squared mass difference values
X,Y = np.meshgrid(x_values, m_values)


Z = nll_obj.neg_ll(X, Y)

plt.figure(7)
plt.xlabel('mixing angles', fontsize= 30)
plt.ylabel('squared mass difference (eV)^2', fontsize= 30)
plt.title('Contour plot', fontsize= 30)


plt.contour(X, Y, Z, cmap = 'gray', alpha = 1, levels = np.linspace(Z.min(), Z.max(), 200))
plt.rc('xtick',labelsize=30)
plt.rc('ytick',labelsize=30)
plt.show()
#%%
#Validating para_mini_nll with a fourth power func

def fourth_power(x, y):
    return x**4 + y**4

valid_para_x, valid_para_value = nll_obj.para_mini_nll([-4, 1, 3], theta = np.pi/4, delta_m_squared = 0, 
                                                           minimise_variable= 'theta', func = fourth_power)
print('\n')
print('Validating the parabolic minimiser:\n')
print(f'The minimised value: {valid_para_value}')
print(f'The x value: {valid_para_x}')



#%%
#3.5
# NLL univariate minisation 
start_time = time.time()
uni_theta, uni_m, uni_num_it = nll_obj.univariate([0.5, 0.6, 1.1], [0.0019, 0.0024, 0.003])
end_time = time.time()

time_taken = end_time - start_time


uni_error = nll_obj.error(uni_theta, uni_m)
print('\n')
print('Univariate method:\n')
print(f'The number of iterations: {uni_num_it}')
print(f'The minimised NLL: {nll_obj.neg_ll(uni_theta, uni_m)}')
print(f'The minimised mixing angle: {uni_theta}')
print(f'The minimised mass: {uni_m}')
print(f'Time taken: {time_taken} s')
print(f'The mixing angle error from curvature estimation: {uni_error[0]}')
print(f'The squared mass difference error from curvature estimation: {uni_error[1]}')

#%%
#4.1 Validating the univariate method

valid_uni_x, valid_uni_y, valid_uni_num_it = nll_obj.univariate([-4, 1, 3],[-4, 2, 6], function = fourth_power)

print('\n')
print('Validating the univariate method:\n')
print(f'The minimised value: {fourth_power(valid_uni_x, valid_uni_y)}')
print(f'The x value: {valid_uni_x}')
print(f'The y value: {valid_uni_y}')


#%%
# Validating gradient method with a square fucntion

def square(x, y):
    return x**2 + y**2

valid_grad_vector, valid_grad_num_it = nll_obj.grad(1, 1, 0.001, first_hundred_iterations = False, func = square)

print('\n')
print('Validating the gradient method:\n')
print(f'The minimised value: {square(valid_grad_vector[0], valid_grad_vector[1])}')
print(f'The x value: {valid_grad_vector[0]}')
print(f'The y value: {valid_grad_vector[1]}')



#%%
#4.2
# Gradient method
# Comparing the step size
l_rate_lst = [1/(10**i) for i in range(3, 7)] # Creating step sizes of different magnitudes


#This defines the initial point
theta_0 = 0.6
delta_m_squared_0 = 0.0024


# This creates a plot how the value of NLL changes with each iterations with different stepsizes
fig, ax = plt.subplots(2,2)
for i in range(len(l_rate_lst)):
    
    vectors = nll_obj.grad(theta_0, delta_m_squared_0, l_rate_lst[i],first_hundred_iterations = True)
    nll_values = []
    for j in range(len(vectors)):
        nll_values.append(nll_obj.neg_ll(vectors[j][0],vectors[j][1]))
    
    if i <=1:
        ax[0, i].plot(range(len(vectors)), nll_values)
        ax[0, i].set_xlabel('Number of iterations', fontsize = 18)
        ax[0, i].set_ylabel('Negative log likelihood', fontsize = 18)
        ax[0, i].set_title(f'learning rate = {l_rate_lst[i]}', fontsize = 18)
    else:
        ax[1, i-2].plot(range(len(vectors)), nll_values)
        ax[1, i-2].set_xlabel('Number of iterations', fontsize = 18)
        ax[1, i-2].set_ylabel('Negative log likelihood', fontsize = 18)
        ax[1, i-2].set_title(f'learning rate = {l_rate_lst[i]}', fontsize = 18)
        
fig.suptitle('Gradient Method', fontsize = 18)
#%%
# Gradient method
# Finding the minimum of NLL
start_time = time.time()
grad_vector, grad_num_it = nll_obj.grad(0.6, delta_m_squared_0, 0.0001,first_hundred_iterations = False)
end_time = time.time()

time_taken = end_time - start_time

grad_error = nll_obj.error(grad_vector[0], grad_vector[1])

print('\n')
print('Gradient method:\n')
print(f'The number of iterations: {grad_num_it}')
print(f'The minimised NLL: {nll_obj.neg_ll(grad_vector[0], grad_vector[1])}')
print(f'The minimised mixing angle: {grad_vector[0]}')
print(f'The minimised mass: {grad_vector[1]}')
print(f'Time taken: {time_taken} s')
print(f'The mixing angle error from curvature estimation: {grad_error[0]}')
print(f'The squared mass difference error from curvature estimation: {grad_error[1]}')

#%%
# 4.2 Quasi_Newton method
# Comparing stepsizes
# Creating plots to compare how NLL changes with differnt stepsize

l_rate_lst = [1/(10**i) for i in range(3, 7)] #Creating a list of different step sizes

plt.figure(9)
fig, ax = plt.subplots(2,2)

theta_0 = 0.6 #Setting the initial points
delta_m_squared_0 = 0.0024

for i in range(len(l_rate_lst)):
    
    vectors_quasi_step = nll_obj.quasi_newton(theta_0, delta_m_squared_0, l_rate_lst[i], first_hundred_iterations = True)
    nll_values_quasi = []
    
    for j in range(len(vectors_quasi_step)):
        nll_values_quasi.append(nll_obj.neg_ll(vectors_quasi_step[j][0],vectors_quasi_step[j][1]))
    
    if i <=1:
        ax[0, i].plot(range(len(vectors_quasi_step)), nll_values_quasi)
        ax[0, i].set_xlabel('Number of iterations', fontsize=20)
        ax[0, i].set_ylabel('Negative log likelihood', fontsize=20)
        ax[0, i].set_title(f'learning rate = {l_rate_lst[i]}', fontsize=20)
        
    else:
        ax[1, i-2].plot(range(len(vectors_quasi_step)), nll_values_quasi)
        ax[1, i-2].set_xlabel('Number of iterations', fontsize=20)
        ax[1, i-2].set_ylabel('Negative log likelihood', fontsize=20)
        ax[1, i-2].set_title(f'learning rate = {l_rate_lst[i]}', fontsize=20)


fig.suptitle('Quasi-Newton Method', fontsize=20)
plt.show()
#%%
# Quasi-Newton Method
#Finding the minimum of NLL
start_time = time.time()
quasi_vector, quasi_num_it = nll_obj.quasi_newton(0.6, delta_m_squared_0, 0.0001, first_hundred_iterations = False)
end_time = time.time()

time_taken = end_time - start_time


quasi_error = nll_obj.error(quasi_vector[0], quasi_vector[1])

print('\n')
print('Quasi-Newton method:\n')
print(f'The number of iterations: {quasi_num_it}')
print(f'The minimised NLL: {nll_obj.neg_ll(quasi_vector[0], quasi_vector[1])}')
print(f'The minimised mixing angle: {quasi_vector[0]}')
print(f'The minimised mass: {quasi_vector[1]}')
print(f'Time taken: {time_taken} s')
print(f'The mixing angle error from curvature estimation: {quasi_error[0]}')
print(f'The squared mass difference error from curvature estimation: {quasi_error[1]}')

#%%
#Quasi-Newton validation

# Validating Quasi-Newton method witha square function
valid_quasi_vector, valid_quasi_num_it = nll_obj.quasi_newton(1, 2, 0.01, first_hundred_iterations = False, func = square)

print('\n')
print('Validating the Quasi-Newton method:\n')
print(f'The minimised value: {square(valid_quasi_vector[0], valid_quasi_vector[1])}')
print(f'The x value: {valid_quasi_vector[0]}')
print(f'The y value: {valid_quasi_vector[1]}')


#%%
# Validating Newton method
valid_newton_vectors, valid_newton_num_it = nll_obj.newton(3, 2, func = square)

print('\n')
print('Validating the Newton method:\n')
print(f'The minimised value: {square(valid_newton_vectors[-1][0], valid_newton_vectors[-1][1])}')
print(f'The x value: {valid_newton_vectors[-1][0]}')
print(f'The y value: {valid_newton_vectors[-1][1]}')


#%% 
# Newton method
start_time = time.time()
newton_vectors, newton_num_it = nll_obj.newton(0.6, 0.0024)
end_time = time.time()

time_taken = end_time - start_time

newton_error = nll_obj.error(newton_vectors[-1][0], newton_vectors[-1][1])

print('\n')
print('Newton method:\n')
print(f'The number of iterations: {newton_num_it}')
print(f'The minimised NLL: {nll_obj.neg_ll(newton_vectors[-1][0], newton_vectors[-1][1])}')
print(f'The minimised mixing angle: {newton_vectors[-1][0]}')
print(f'The minimised mass: {newton_vectors[-1][1]}')
print(f'Time taken: {time_taken} s')
print(f'The mixing angle error from curvature estimation: {newton_error[0]}')
print(f'The squared mass difference error from curvature estimation: {newton_error[1]}')


#%%
# gradient in 3D

alpha_0 = 1.7 # Initial guess for alpha
start_time = time.time()
grad_cross_sec_vector, grad_cross_sec_num_it = nll_obj.grad_cross_sec(theta_0, delta_m_squared_0, alpha_0, 0.0001, first_hundred_iterations = False)
end_time = time.time()

time_taken = end_time - start_time

grad_cross_sec_error = nll_obj.error(grad_cross_sec_vector[0], grad_cross_sec_vector[1], 
                                     alpha = grad_cross_sec_vector[2],
                                     num_variables = 3)

print('\n')
print('3D gradient method:\n')
print(f'The number of iterations: {grad_cross_sec_num_it}')
print(f'The minimised NLL: {nll_obj.neg_ll_cross_sec(grad_cross_sec_vector[0], grad_cross_sec_vector[1], grad_cross_sec_vector[2])}')
print(f'The minimised mixing angle: {grad_cross_sec_vector[0]}')
print(f'The minimised mass: {grad_cross_sec_vector[1]}')
print(f'The minimised cross section: {grad_cross_sec_vector[2]}')
print(f'Time taken: {time_taken} s')
print(f'The mixing angle error from curvature estimation: {grad_cross_sec_error[0]}')
print(f'The squared mass difference error from curvature estimation: {grad_cross_sec_error[1]}')
print(f'The cross-section error from curvature estimation: {grad_cross_sec_error[2]}')

#%%
# Validating gradeint method in 3D with a square function


def square_3d(x, y, z):
    return x**2 + y**2 + z**2



valid_grad_cross_sec_vector, grad_cross_sec_num_it = nll_obj.grad_cross_sec(1, 1, 1, 0.01, first_hundred_iterations = False, 
                                                                      func = square_3d)

print('\n')
print('Validating the 3D gradient method:\n')
# print(f'The number of iterations: {grad_cross_sec_num_it}')
print(f'The minimised NLL: {square_3d(valid_grad_cross_sec_vector[0], valid_grad_cross_sec_vector[1], valid_grad_cross_sec_vector[2])}')
print(f'The minimised mixing angle: {valid_grad_cross_sec_vector[0]}')
print(f'The minimised mass: {valid_grad_cross_sec_vector[1]}')
print(f'The minimised cross section: {valid_grad_cross_sec_vector[2]}')
#%%
# Validating Quasi-Newton method in 3D

valid_quasi_vector, valid_grad_quasi_num_it= nll_obj.quasi_newton_cross_sec(1, 1, 1, 0.01, first_hundred_iterations = False, func = square_3d)

print('\n')
print('Validating the 3D Quasi-Newton method:\n')
# print(f'The number of iterations: {valid_grad_quasi_num_it}')
print(f'The minimised value of the function: {square_3d(valid_quasi_vector[0], valid_quasi_vector[1], valid_quasi_vector[2])}')
print(f'The minimised x value: {valid_quasi_vector[0]}')
print(f'The minimised y value: {valid_quasi_vector[1]}')
print(f'The minimised z value: {valid_quasi_vector[2]}')

#%%
# Quasi-Newton in 3D
alpha_0 = 1.7 # Initial guess for alpha
start_time = time.time()
quasi_cross_sec_vector, quasi_cross_sec_num_it = nll_obj.quasi_newton_cross_sec(0.6, 0.0024, 1.7, 0.0001, first_hundred_iterations = False)
end_time = time.time()

time_taken = end_time - start_time

quasi_cross_sec_error = nll_obj.error(quasi_cross_sec_vector[0], quasi_cross_sec_vector[1], 
                                     alpha = quasi_cross_sec_vector[2],
                                     num_variables = 3)

print('\n')
print('3D Quasi-Newton method:\n')
print(f'The number of iterations: {quasi_cross_sec_num_it}')
print(f'The minimised NLL: {nll_obj.neg_ll_cross_sec(quasi_cross_sec_vector[0], quasi_cross_sec_vector[1], quasi_cross_sec_vector[2])}')
print(f'The minimised mixing angle: {quasi_cross_sec_vector[0]}')
print(f'The minimised mass: {quasi_cross_sec_vector[1]}')
print(f'The minimised cross section: {quasi_cross_sec_vector[2]}')
print(f'Time taken: {time_taken} s')
print(f'The mixing angle error from curvature estimation: {quasi_cross_sec_error[0]}')
print(f'The squared mass difference error from curvature estimation: {quasi_cross_sec_error[1]}')
print(f'The cross-section error from curvature estimation: {quasi_cross_sec_error[2]}')



#%%
# nll univariate in 3D
start_time = time.time()
uni_theta_min_3d, uni_m_min_3d, uni_alpha_min_3d, uni_num_it_3d = nll_obj.univariate_cross_sec([0.5, 0.6, 1.1], [0.0019, 0.0024, 0.003], [1.0, 1.7, 2.0])
end_time = time.time()

time_taken = end_time - start_time

uni_error_3d = nll_obj.error(uni_theta_min_3d, uni_m_min_3d, uni_alpha_min_3d, num_variables = 3)
print('\n')
print('3D univariate method:\n')
print(f'The number of iterations: {uni_num_it_3d}')
print(f'The minimised NLL: {nll_obj.neg_ll_cross_sec(uni_theta_min_3d, uni_m_min_3d, uni_alpha_min_3d)}')
print(f'The minimised mixing angle: {uni_theta_min_3d}')
print(f'The minimised mass: {uni_m_min_3d}')
print(f'The minimised cross section: {uni_alpha_min_3d}')
print(f'Time taken: {time_taken} s')
print(f'The mixing angle error from curvature estimation: {uni_error_3d[0]}')
print(f'The squared mass difference error from curvature estimation: {uni_error_3d[1]}')
print(f'The cross-section error from curvature estimation: {uni_error_3d[2]}')

#%%
# Validating univariate in 3D

def fourth_power_3d(x, y, z):
    return x**4 + y**4 + z**4

uni_valid_x_3d, uni_valid_y_3d, uni_valid_z_3d, uni_valid_num_it_3d = nll_obj.univariate_cross_sec([-3, -2, 3], 
                                                                                               [-1, 0.6, 3], 
                                                                                               [-1, 1, 3], 
                                                                                               function = fourth_power_3d)


print('\n')
print('Validating the 3D univariate method:\n')

print(f'The minimised value of the function: {fourth_power_3d(uni_valid_x_3d, uni_valid_y_3d, uni_valid_z_3d)}')
print(f'The minimised mixing angle: {uni_valid_x_3d}')
print(f'The minimised mass: {uni_valid_y_3d}')
print(f'The minimised cross section: {uni_valid_z_3d}')
# print(uni_theta_min_3d, uni_m_min_3d, uni_alpha_min_3d, uni_num_it_3d)

#%%
# Newton method in 3D

start_time = time.time()
newton_cross_sec_vectors, newton_cross_sec_num_it = nll_obj.newton_cross_sec(0.6, 0.0024, 1.7)
end_time = time.time()

time_taken = end_time - start_time

newton_cross_sec_error = nll_obj.error(newton_cross_sec_vectors[-1][0], newton_cross_sec_vectors[-1][1], 
                                     alpha = newton_cross_sec_vectors[-1][2],
                                     num_variables = 3)

print('\n')
print('3D Newton method:\n')
print(f'The number of iterations: {newton_cross_sec_num_it}')
print(f'The minimised NLL: {nll_obj.neg_ll_cross_sec(newton_cross_sec_vectors[-1][0], newton_cross_sec_vectors[-1][1], newton_cross_sec_vectors[-1][2])}')
print(f'The minimised mixing angle: {newton_cross_sec_vectors[-1][0]}')
print(f'The minimised mass: {newton_cross_sec_vectors[-1][1]}')
print(f'The minimised cross section: {newton_cross_sec_vectors[-1][2]}')
print(f'Time taken: {time_taken} s')
print(f'The mixing angle error from curvature estimation: {newton_cross_sec_error[0]}')
print(f'The squared mass difference error from curvature estimation: {newton_cross_sec_error[1]}')
print(f'The cross-section error from curvature estimation: {newton_cross_sec_error[2]}')

#%%
# Validating Newton in 3D
valid_newton_vectors_3d, valid_newton_num_it_3d= nll_obj.newton_cross_sec(1, 1, 1,func = square_3d)

print('\n')
print('Validating the 3D Newton method:\n')
print(f'The minimised value of the function: {square_3d(valid_newton_vectors_3d[-1][0], valid_newton_vectors_3d[-1][1], valid_newton_vectors_3d[-1][2])}')
print(f'The minimised x value: {valid_newton_vectors_3d[-1][0]}')
print(f'The minimised y value: {valid_newton_vectors_3d[-1][1]}')
print(f'The minimised z value: {valid_newton_vectors_3d[-1][2]}')





















