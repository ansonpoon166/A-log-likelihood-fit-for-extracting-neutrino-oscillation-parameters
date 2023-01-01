#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 17 12:06:15 2020

@author: ansonpoon
"""
import numpy as np


class NLL:
    
    def __init__(self, unosc_flux, observed_num, L, E_list):
        '''
        This initialises the object with the experimental data and condition.

        Parameters
        ----------
        unosc_flux : list
            The expected unoscillated flux of neutrinos of different energies.
        observed_num : list
            The observed number of neutrinos of different energies.
        L : float or int
            The distance the neutrinos travel.
        E_list : list
            The list of energies of the neutrinos.

        Returns
        -------
        None.

        '''
        self.unosc_flux = unosc_flux
        self.observed_num = observed_num
        self.L = L
        self.E_list=  E_list
        
        
    def osc_prob(self, theta23, delta_m_squared):
        '''
        This function calculates the oscillation probability of given mixing angle 
        and difference between the two neutrinos.

        Parameters
        ----------
        theta23 : int or numpy array
            The mixing angle.
        delta_m_squared : int or numpy array
            The difference between the squared masses of two neutrinos.

        Returns
        -------
        float or numpy array
            The oscillation probability.

        '''
        func_1 = (np.sin(2 * theta23)) ** 2
        func_2 = (np.sin((1.267 * delta_m_squared * self.L) / self.E_list)) ** 2
        return 1 - func_1*func_2
        
    
    
    def osc_event_pred(self, theta23, delta_m_squared):
        '''
        This function calculates the oscillated event prediction of given mixing angle 
        and difference between the two neutrinos.

        Parameters
        ----------
        theta23 : float
            The mixing angle.
        delta_m_squared : float
            The difference between the squared masses of two neutrinos.

        Returns
        -------
        float 
            The oscillated event prediction.

        '''
        return self.osc_prob(theta23, delta_m_squared) * self.unosc_flux
    
    
    
    def neg_ll(self, theta23, delta_m_squared):
        '''
        This function calculates the negative log likelihood of given mixing angle 
        and difference between the two neutrinos using Stirling's approximation'

        Parameters
        ----------
        theta23 : int or numpy array
            The mixing angle.
        delta_m_squared : int or numpy array
            The difference between the squared masses of two neutrinos.

        Returns
        -------
        nll : float
            The negative log likelihood.

        '''
        nll = 0
        for i in range(len(self.E_list)):
            
            #Calculate probability
            func_1 = (np.sin(2*theta23))**2
            func_2 = (np.sin((1.267*delta_m_squared*self.L)/self.E_list[i]))**2
            osc_prob = 1 - func_1*func_2
            
            #Calculate the rate, lambda
            lamb = osc_prob * self.unosc_flux[i]
            
            #This is to avoid the zero division error when the rate is zero.
            if self.observed_num[i] ==0:
                term = lamb
            else:
                term = lamb - self.observed_num[i] + self.observed_num[i] * np.log(self.observed_num[i]/lamb)
                
            nll += term
            
        return nll
    
    
    
    def para_mini_nll(self, initial_guesses, minimise_variable, theta = np.pi/4, delta_m_squared = 0.0024, func = None):
        '''
        This function minimises a one dimensional function using parabolic method.

        Parameters
        ----------
        initial_guesses : list, a list of three floats
            The initial guesses of the value at which the minimum exists.
            The initial guesses must satisfy the condition that the value of the function at the first and last number 
            must be greater than the value of the function at the second number.
            
        minimise_variable : string
            The variable used for minimisation.
            For the mixing angle, 'theta'
            For the squared mass difference, 'delta_m_squared'
            
        theta : float , optional
            The mixing angle. The default is np.pi/4.
            
        delta_m_squared : float, optional
            The squared mass differnce. The default is 0.0024.
            
        func : function, optional
            The function to be minimised. The default is None, which corresponds to the negative log likelihood the Poisson case.

        Returns
        -------
        x_3: float
            The value of the variable at which the minimum occurs.
        
        float
            The minimum value of the function, the minimum value of the negative log likelihood function by default.

        '''
        
        #This sets the default function as the negative log likelihood function
        if func == None:
            func = self.neg_ll
            
        x = initial_guesses
            
        diff = 0
        run = True
        x_3_prev = 0
         
        while run:
            # Calculating the value of the function at the initial guesses corresponding to the variable used for the minimisation.
            if minimise_variable == 'theta':
                y = [func(i, delta_m_squared) for i in x ]
            elif minimise_variable == 'delta_m_squared':
                y = [func(theta, i) for i in x ]
            
            #This calculates the the new guess.
            numerator = (x[2]**2-x[1]**2)*y[0] + (x[0]**2-x[2]**2)*y[1] + (x[1]**2-x[0]**2)*y[2]
            denominator =  (x[2]-x[1])*y[0] + (x[0]-x[2])*y[1] + (x[1]-x[0])*y[2]
            
            #The denominator tends to zero whent the points are close to the minimum, meaning it has converged to the minimum
            if denominator == 0: 
                break
            x_3 = numerator/(2*denominator)
            
        
            diff = x_3 - x_3_prev
            x_3_prev = x_3
            
            x.append(x_3)
            
            #This calculates the value of the function at the new point.
            if minimise_variable == 'theta':
                y.append(func(x_3, delta_m_squared))
            elif minimise_variable == 'delta_m_squared':
                y.append(func(theta, x_3))
                
            #This removes the point in the list that corresponds to the highest value of the function
            max_data = y.index(max(y))
            y.remove(y[max_data])
            x.remove(x[max_data])
            x.sort()
            
            #This sets the stopping condition for the algorithm.
            if abs(diff) < 10**(-7) and minimise_variable == 'theta':
                run = False
                
                
            elif minimise_variable == 'delta_m_squared' and abs(diff) < 10**(-5):
                run = False
                
        if minimise_variable == 'theta':
            return x_3, func(x_3, delta_m_squared)
        elif minimise_variable == 'delta_m_squared':
            return x_3, func(theta, x_3)
    
    
    def univariate(self, initial_theta_guesses, initial_delta_m_squared_guesses, function = None):
        '''
        This function minimises a two dimensional function using univariate method.

        Parameters
        ----------
        initial_theta_guesses : list, a list of three floats
            The initial guesses of the value of he mixing angle at which the minimum exists.
            The initial guesses must satisfy the condition that the value of the function at the first and last number 
            must be greater than the value of the function at the second number.
            
        initial_delta_m_squared_guesses : TYPE
            The initial guesses of the squared mass difference value at which the minimum exists.
            The initial guesses must satisfy the condition that the value of the function at the first and last number 
            must be greater than the value of the function at the second number.
        function : function, optional
            The function to be minimised. The default is None, which corresponds to the negative log likelihood the Poisson case.

        Returns
        -------
        theta_min : float
            The value of the mixing angle at which the minimum occurs.
        m_min : float
            The value of the squared mass difference at which the minimum occurs.
        num_it : int
            The number of iterations this function takes to find the minimum.

        '''
        run = True
        
        m_min = initial_delta_m_squared_guesses[1]
        theta_min = initial_theta_guesses[1]
        
        theta_prev = 0
        m_prev = 0
        num_it = 0
        
        while run:
            #This calculates the value of theta with a fixed delta m
            theta_min, y_min = self.para_mini_nll(initial_theta_guesses, minimise_variable = 'theta', theta = None, delta_m_squared = m_min, func = function)
            
            #This calculates the value of delta m with a fixed theta
            m_min, y_min = self.para_mini_nll(initial_delta_m_squared_guesses, minimise_variable = 'delta_m_squared', theta = theta_min, delta_m_squared = None, func = function)
            
            diff_theta = theta_min - theta_prev
            diff_m = m_min - m_prev
            #This sets the condition at which the loop needs to stop
            if abs(diff_m) < 10**-7 and abs(diff_theta) <10**-5:
                run = False
            
            num_it += 1
            m_prev = m_min
            theta_prev = theta_min
            
            
        return theta_min, m_min, num_it
    

    def grad(self, theta_0, delta_m_squared_0, l_rate, first_hundred_iterations = False, func = None):
        
        '''
        This function minimises a two dimensional function using gradient method.

        Parameters
        ----------
        theta_0 : float
            The initial guess of the x value or the mixing angle.
            
        delta_m_squared_0 : float
            The initial guess of y value or the squared mass difference.
            
        l_rate : float
            The stepsize.
            
        first_hundred_iterations : boolean, optional
            If True, the function will return the hundred iterations only. The default is False.
            
        func : function, optional
            The function to be minimised. The default is None.

        Returns
        -------
        v: numpy array
            Returned only if first_hundred_iterations == 'True'.
            An array of the values from the first hundred iterations, where the first column is the mixing angle
            and the second column is the squared mass difference.
            
        x: float
            Returned only if first_hundred_iterations == 'False'.
            The values of the two variables at the minimum
            
        num_it: int
            The number of iterations.

        '''
       
        scaling_ratio = theta_0/delta_m_squared_0
        #This scales the delta m squared accordingly.
        if func == None:
            func = self.neg_ll
        scaled_m_0 = delta_m_squared_0*scaling_ratio
           
        
        h = 0.0000001
        x = np.array([theta_0, scaled_m_0])
        v = []
        
        if first_hundred_iterations == True:
            for i in range(100):
                
                #This calculates the gradient vector
                theta_deriv = (func(x[0] + h, x[1]/scaling_ratio) - func(x[0], x[1]/scaling_ratio))/h
                m_deriv = (func(x[0], (x[1] + h)/scaling_ratio) - func(x[0], x[1]/scaling_ratio))/h
                
                deriv = np.array([theta_deriv, m_deriv])
                
                x = x - l_rate * deriv
                
                v.append(x)
            
            for i in range(len(v)):
                
                #This rescales the squared mass difference
                v[i][1] = v[i][1]/scaling_ratio
            
            return v
        
        else:
            run = True
          
            num_it = 0 
            x_prev =np.zeros(x.shape)
            while run:
                theta_deriv = (func(x[0] + h, x[1]/scaling_ratio) - func(x[0], x[1]/scaling_ratio))/h
                m_deriv = (func(x[0], (x[1] + h)/scaling_ratio) - func(x[0], x[1]/scaling_ratio))/h
                
                deriv = np.array([theta_deriv, m_deriv])
                
                x = x - l_rate * deriv
              
                num_it += 1
                
                #This creates the condition for the while to stop
                if abs(x[0] - x_prev[0]) < 0.0000001 and abs(x[1] - x_prev[1]) < 0.0000001:
                    run = False
                    
                #This can constantly replace the value of x with the better approximation
                x_prev = x
                
                
            x[1] = x[1]/scaling_ratio
            
            return x, num_it        



    def quasi_newton(self, theta_0, delta_m_squared_0, l_rate, first_hundred_iterations = False, func = None):
        '''
        This function minimises a two dimensional function using gradient method.
 
         Parameters
         ----------
         theta_0 : float
             The initial guess of the x value or the mixing angle.
             
         delta_m_squared_0 : float
             The initial guess of y value or the squared mass difference.
             
         l_rate : float
             The stepsize.
             
         first_hundred_iterations : boolean, optional
             If True, the function will return the hundred iterations only. The default is False.
             
         func : function, optional
             The function to be minimised. The default is None.
 
         Returns
         -------
         x_lst: list
             Returned only if first_hundred_iterations == 'True'.
             An list of the values of the variables from the first hundred iterations, 
             where the first column is the mixing angle
             and the second column is the squared mass difference.
             
         x: float
             Returned only if first_hundred_iterations == 'False'.
             The values of the two variables at the minimum
             
         num_it: int
             The number of iterations.
        '''
        
        # This sets the default function to be the NLL function
        if func == None:
            func = self.neg_ll
        scaling_ratio = theta_0/delta_m_squared_0
        
        h = 0.00001
        scaled_m_0 = delta_m_squared_0*scaling_ratio
        # run = False
        
        
        x_0 = np.array([theta_0, scaled_m_0])
        
        theta_deriv_0 = (func(x_0[0] + h, x_0[1]/scaling_ratio) - func(x_0[0], x_0[1]/scaling_ratio))/h
        m_deriv_0 = (func(x_0[0], (x_0[1] + h)/scaling_ratio) - func(x_0[0], x_0[1]/scaling_ratio))/h
        
        deriv_0 = np.array([theta_deriv_0, m_deriv_0])
        G_0 = np.identity(len(x_0))
        
        x_1 = x_0 - l_rate * np.matmul(G_0, deriv_0)
        
        theta_deriv_1 = (func(x_1[0] + h, x_1[1]/scaling_ratio) - func(x_1[0], x_1[1]/scaling_ratio))/h
        m_deriv_1 = (func(x_1[0], (x_1[1] + h)/scaling_ratio) - func(x_1[0], x_1[1]/scaling_ratio))/h
        
        deriv_1 = np.array([theta_deriv_1, m_deriv_1])
        
        delta = x_1 - x_0
        gamma = deriv_1 - deriv_0
        outer = np.outer(delta, delta)
        G_1 = G_0  + outer/(np.dot(gamma, delta)) - (np.matmul(G_0, np.matmul(outer, G_0)))/np.dot(gamma, np.matmul(G_0, gamma))
        
        x_lst = [x_0, x_1]
        deriv_lst = [deriv_0, deriv_1]
        G = [G_0, G_1]
        
        if first_hundred_iterations:
          
            for i in range(100):
                delta = x_lst[-1] - x_lst[-2]
                gamma = deriv_lst[-1] - deriv_lst[-2]
                
                outer = np.outer(delta, delta)
                
                
                
                
                x_next = x_lst[-1] - l_rate * np.matmul(G[-1], deriv_lst[-1])
                x_lst.append(x_next)
                
                G_next = G[-1]  + outer/(np.dot(gamma, delta)) - (np.matmul(G[-1], np.matmul(outer, G[-1])))/np.dot(gamma, np.matmul(G[-1], gamma))
                G.append(G_next)
                
                theta_deriv_next = (func(x_lst[-1][0] + h, x_lst[-1][1]/scaling_ratio) - func(x_lst[-1][0], x_lst[-1][1]/scaling_ratio))/h
                m_deriv_next = (func(x_lst[-1][0], (x_lst[-1][1] + h)/scaling_ratio) - func(x_lst[-1][0], x_lst[-1][1]/scaling_ratio))/h
                
                deriv_next = np.array([theta_deriv_next, m_deriv_next])
                deriv_lst.append(deriv_next)
                
            for i in range(len(x_lst)):
                
                x_lst[i][1] = x_lst[i][1]/scaling_ratio
                
            return x_lst
        else:
            
            run = True
            while run:
                delta = x_lst[-1] - x_lst[-2]
                gamma = deriv_lst[-1] - deriv_lst[-2]
                
              
                
                outer = np.outer(delta, delta)
                
           
                
                
                x_next = x_lst[-1] - l_rate * np.matmul(G[-1], deriv_lst[-1])
                x_lst.append(x_next)
                
                G_next = G[-1]  + outer/(np.dot(gamma, delta)) - (np.matmul(G[-1], np.matmul(outer, G[-1])))/np.dot(gamma, np.matmul(G[-1], gamma))
                G.append(G_next)
                
                theta_deriv_next = (func(x_lst[-1][0] + h, x_lst[-1][1]/scaling_ratio) - func(x_lst[-1][0], x_lst[-1][1]/scaling_ratio))/h
                m_deriv_next = (func(x_lst[-1][0], (x_lst[-1][1] + h)/scaling_ratio) - func(x_lst[-1][0], x_lst[-1][1]/scaling_ratio))/h
                
                deriv_next = np.array([theta_deriv_next, m_deriv_next])
                deriv_lst.append(deriv_next)
                if abs(delta[0]) <0.000001 and abs(delta[1]) <0.000001:
                    
                    run = False
                    
                if abs(gamma[0]) <0.000001 and abs(gamma[1]) <0.000001:
                    run = False
                
                
            for i in range(len(x_lst)):
                
                x_lst[i][1] = x_lst[i][1]/scaling_ratio
                
            return x_lst[-1], len(x_lst)
     
        
        
    def neg_ll_cross_sec(self, theta23, delta_m_squared, alpha):
        '''
        This function calculates the negative log likelihood of given mixing angle 
        square mass difference and the cross-section using Stirling's approximation'

        Parameters
        ----------
        theta23 : float or numpy array
            The mixing angle.
            
        delta_m_squared : float or numpy array
            The difference between the squared masses of two neutrinos.
            
        alpha: float or numpy array
            The neutrino cross-section
        
        Returns
        -------
        nll : float or numpy array
            The negative log likelihood.

        '''
        
        nll = 0
        for i in range(len(self.E_list)):
            
            #Calculate probability
            func_1 = (np.sin(2*theta23))**2
            func_2 = (np.sin((1.267*delta_m_squared*self.L)/self.E_list[i]))**2
            osc_prob = 1 - func_1*func_2
            
            #Calculate the rate with the cross section
            lamb = osc_prob * self.unosc_flux[i] * alpha * self.E_list[i]
            
            if self.observed_num[i] ==0:
                term = lamb
            else:
                term = lamb - self.observed_num[i] + self.observed_num[i] * np.log(self.observed_num[i]/lamb)
                
            nll += term
            
        return nll
    
    
    
    
    def grad_cross_sec(self, theta_0, delta_m_squared_0, alpha_0, l_rate, first_hundred_iterations = False, func = None):
        '''
        This function minimises a three dimensional function using gradient method.

        Parameters
        ----------
        theta_0 : float
            The initial guess of the x value or the mixing angle.
            
        delta_m_squared_0 : float
            The initial guess of y value or the squared mass difference.
            
        alpha_0: float
            The initial guess of z value or the cross-section.
            
        l_rate : float
            The stepsize.
            
        first_hundred_iterations : boolean, optional
            If True, the function will return the hundred iterations only. The default is False.
            
        func : function, optional
            The function to be minimised. The default is None.
    
        Returns
        -------
        x_lst: list
             Returned only if first_hundred_iterations == 'True'.
             An list of the values of the variables from the first hundred iterations, 
             where the first column is the mixing angle
             and the second column is the squared mass difference.
             
         x: float
             Returned only if first_hundred_iterations == 'False'.
             The values of the two variables at the minimum
             
         num_it: int
             The number of iterations.
        '''
        # This sets the default function to be the NLL with cross-section
        if func == None:
            func = self.neg_ll_cross_sec
        
        # This scales the squared mass difference by the its ratio to theta
        scale_m = theta_0/delta_m_squared_0
        h = 0.00001
        scaled_m_0 = delta_m_squared_0*scale_m
        
        x = np.array([theta_0, scaled_m_0, alpha_0])
        
        v = []
        
        if first_hundred_iterations:
            for i in range(1000):
                # This calculates the derivatives using the forward difference scheme
                theta_deriv = (func(x[0] + h, x[1]/scale_m, x[2]) - func(x[0], x[1]/scale_m, x[2]))/h
                m_deriv = (func(x[0], (x[1] + h)/scale_m, x[2]) - func(x[0], x[1]/scale_m, x[2]))/h
                alpha_deriv = (func(x[0], x[1]/scale_m, x[2]+h) - func(x[0], x[1]/scale_m, x[2]))/h
                
                deriv = np.array([theta_deriv, m_deriv, alpha_deriv])
                
                x = x - l_rate * deriv
                
                v.append(x)
            for i in range(len(v)):
                # This rescales the square mass difference
                v[i][1] = v[i][1]/scale_m
            
            return v
        
        else:
            
            run = True
          
            num_it = 0 
            x_prev =np.zeros(x.shape)
            while run:
                # This calculates the derivatives using the forward difference scheme
                theta_deriv = (func(x[0] + h, x[1]/scale_m, x[2]) - func(x[0], x[1]/scale_m, x[2]))/h
                m_deriv = (func(x[0], (x[1] + h)/scale_m, x[2]) - func(x[0], x[1]/scale_m, x[2]))/h
                alpha_deriv = (func(x[0], x[1]/scale_m, x[2]+h) - func(x[0], x[1]/scale_m, x[2]))/h
                
                deriv = np.array([theta_deriv, m_deriv, alpha_deriv])
                
                x = x - l_rate * deriv
              
                num_it += 1
                # This sets the conidtion to stop the iterations
                if abs(x[0] - x_prev[0]) <0.000001 and abs(x[1] - x_prev[1]) <0.000001:
                    run = False
                
                x_prev = x
                
            # This rescales the square mass difference
            x[1] = x[1]/scale_m
            return x, num_it
        
        
    def quasi_newton_cross_sec(self, theta_0, delta_m_squared_0, alpha_0, l_rate, first_hundred_iterations = False, func = None):
        '''
        This function minimises a three dimensional function using the Quasi-Newton method.

        Parameters
        ----------
        theta_0 : float
            The initial guess of the x value or the mixing angle.
            
        delta_m_squared_0 : float
            The initial guess of y value or the squared mass difference.
            
        alpha_0: float
            The initial guess of z value or the cross-section.
            
        l_rate : float
            The stepsize.
            
        first_hundred_iterations : boolean, optional
            If True, the function will return the hundred iterations only. The default is False.
            
        func : function, optional
            The function to be minimised. The default is None.
    
        Returns
        -------
        x_lst: list
             Returned only if first_hundred_iterations == 'True'.
             An list of the values of the variables from the first hundred iterations, 
             where the first column is the mixing angle
             and the second column is the squared mass difference.
             
         x: float
             Returned only if first_hundred_iterations == 'False'.
             The values of the two variables at the minimum
             
         num_it: int
             The number of iterations.
        ''' 
        # This sets the default function to be the NLL function with cross section.
        if func == None:
            func = self.neg_ll_cross_sec
            
        # This scales the squared mass difference by the its ratio to the mixing angle.
        scaling_ratio = theta_0/delta_m_squared_0
        
        h = 0.00001 # Stepsize used in the forward difference scheme to determine the derivatives.
        scaled_m_0 = delta_m_squared_0*scaling_ratio
        
        
        x_0 = np.array([theta_0, scaled_m_0, alpha_0])
        
        # This calculates the  derivates at the initial point.
        theta_deriv_0 = (func(x_0[0] + h, x_0[1]/scaling_ratio, x_0[2]) - func(x_0[0], x_0[1]/scaling_ratio, x_0[2]))/h
        m_deriv_0 = (func(x_0[0], (x_0[1] + h)/scaling_ratio, x_0[2]) - func(x_0[0], x_0[1]/scaling_ratio, x_0[2]))/h
        alpha_deriv_0 = (func(x_0[0], x_0[1]/scaling_ratio, x_0[2] + h) - func(x_0[0], x_0[1]/scaling_ratio, x_0[2]))/h
        deriv_0 = np.array([theta_deriv_0, m_deriv_0, alpha_deriv_0])
        
        # The identity matrix is used as the G matrix at the initial point.
        G_0 = np.identity(len(x_0))
        
        # This calculates the next point.
        x_1 = x_0 - l_rate * np.matmul(G_0, deriv_0)
        
        # This calculates the derivative at the first point.
        theta_deriv_1 = (func(x_1[0] + h, x_1[1]/scaling_ratio, x_1[2]) - func(x_1[0], x_1[1]/scaling_ratio, x_1[2]))/h
        m_deriv_1 = (func(x_1[0], (x_1[1] + h)/scaling_ratio,  x_1[2]) - func(x_1[0], x_1[1]/scaling_ratio, x_1[2]))/h
        alpha_deriv_1 = (func(x_1[0], (x_1[1] + h)/scaling_ratio, x_1[2]+h) - func(x_1[0], x_1[1]/scaling_ratio, x_1[2]))/h
        deriv_1 = np.array([theta_deriv_1, m_deriv_1, alpha_deriv_1])
        
        # This calculates the G matrix at the first point
        delta = x_1 - x_0
        gamma = deriv_1 - deriv_0
        outer = np.outer(delta, delta)
        G_1 = G_0  + outer/(np.dot(gamma, delta)) - (np.matmul(G_0, np.matmul(outer, G_0)))/np.dot(gamma, np.matmul(G_0, gamma))
        
        # This creates lists for the point, derivative and the G matrix at each iteration
        x_lst = [x_0, x_1]
        deriv_lst = [deriv_0, deriv_1]
        G = [G_0, G_1]
        
        
        if first_hundred_iterations:
          
            for i in range(100):
                # These variables are later used in the calcultion of the G matrix.
                delta = x_lst[-1] - x_lst[-2]
                gamma = deriv_lst[-1] - deriv_lst[-2]
                outer = np.outer(delta, delta)
                
           
                
                # This calculates the next point.
                x_next = x_lst[-1] - l_rate * np.matmul(G[-1], deriv_lst[-1])
                x_lst.append(x_next)
                
                # This calculates the G matrix at the next point
                G_next = G[-1]  + outer/(np.dot(gamma, delta)) - (np.matmul(G[-1], np.matmul(outer, G[-1])))/np.dot(gamma, np.matmul(G[-1], gamma))
                G.append(G_next)
                
                # This calculates the derivatives at the next point.
                theta_deriv_next = (func(x_lst[-1][0] + h, x_lst[-1][1]/scaling_ratio, x_lst[-1][2]) - func(x_lst[-1][0], x_lst[-1][1]/scaling_ratio, x_lst[-1][2]))/h
                m_deriv_next = (func(x_lst[-1][0], (x_lst[-1][1] + h)/scaling_ratio, x_lst[-1][2]) - func(x_lst[-1][0], x_lst[-1][1]/scaling_ratio, x_lst[-1][2]))/h
                alpha_deriv_next = (func(x_lst[-1][0], (x_lst[-1][1] + h)/scaling_ratio, x_lst[-1][2] + h) - func(x_lst[-1][0], x_lst[-1][1]/scaling_ratio, x_lst[-1][2]))/h
                deriv_next = np.array([theta_deriv_next, m_deriv_next, alpha_deriv_next])
                deriv_lst.append(deriv_next)
                
            for i in range(len(x_lst)):
                # This rescales th variable.
                x_lst[i][1] = x_lst[i][1]/scaling_ratio
                
            
            return x_lst
        
        else:
            
            run = True
            while run:
                # These variables are later used in the calcultion of the G matrix.
                delta = x_lst[-1] - x_lst[-2]
                gamma = deriv_lst[-1] - deriv_lst[-2]
                outer = np.outer(delta, delta)
                
           
                
                # This calculates the next point.
                x_next = x_lst[-1] - l_rate * np.matmul(G[-1], deriv_lst[-1])
                x_lst.append(x_next)
                
                # This calculates the G matrix at the next point
                G_next = G[-1]  + outer/(np.dot(gamma, delta)) - (np.matmul(G[-1], np.matmul(outer, G[-1])))/np.dot(gamma, np.matmul(G[-1], gamma))
                G.append(G_next)
                
                # This calculates the derivatives at the next point.
                theta_deriv_next = (func(x_lst[-1][0] + h, x_lst[-1][1]/scaling_ratio, x_lst[-1][2]) - func(x_lst[-1][0], x_lst[-1][1]/scaling_ratio, x_lst[-1][2]))/h
                m_deriv_next = (func(x_lst[-1][0], (x_lst[-1][1] + h)/scaling_ratio, x_lst[-1][2]) - func(x_lst[-1][0], x_lst[-1][1]/scaling_ratio, x_lst[-1][2]))/h
                alpha_deriv_next = (func(x_lst[-1][0], (x_lst[-1][1] + h)/scaling_ratio, x_lst[-1][2] + h) - func(x_lst[-1][0], x_lst[-1][1]/scaling_ratio, x_lst[-1][2]))/h
                deriv_next = np.array([theta_deriv_next, m_deriv_next, alpha_deriv_next])
                deriv_lst.append(deriv_next)
                
                # This sets the stopping condition for the iteration.
                if abs(delta[0]) <0.000001 and abs(delta[1]) <0.000001:
                   run = False
                    
                elif abs(gamma[0]) <0.000001 and abs(gamma[1]) <0.000001:
                    run = False
                
             
            for i in range(len(x_lst)):
                # This rescales the variable.
                x_lst[i][1] = x_lst[i][1]/scaling_ratio
            
            x = x_lst[-1]
            num_it = len(x_lst)
            
            return x, num_it
        
    def para_mini_nll_cross_sec(self, initial_guesses, minimise_variable, theta = np.pi/4, delta_m_squared = 0.0024, alpha = 1.7, func = None):
        '''
        
        This function minimises a one dimensional function taking into account the cross-section using parabolic method.

        Parameters
        ----------
        initial_guesses : list, a list of three floats
            The initial guesses of the value at which the minimum exists.
            The initial guesses must satisfy the condition that the value of the function at the first and last number 
            must be greater than the value of the function at the second number.
            
        minimise_variable : string
            The variable used for minimisation.
            For the mixing angle, 'theta'
            For the squared mass difference, 'delta_m_squared'
            For the cross-section, 'alpha'
            
        theta : float , optional
            The mixing angle. The default is np.pi/4.
            
        delta_m_squared : float, optional
            The squared mass differnce. The default is 0.0024.
            
        alpha: float, optional
            The cross-section. The default is 1.7.
            
        func : function, optional
            The function to be minimised. The default is None, which corresponds to the negative log likelihood function.

        Returns
        -------
        x_3: float
            The value of the variable at which the minimum occurs.
        
        float
            The minimum value of the function, the minimum value of the negative log likelihood function by default.

        '''
        # This sets the default function.
        if func == None:
            func = self.neg_ll_cross_sec
            
        x = initial_guesses
            
        diff = 0
        run = True
        x_3_prev = 0
         
        while run:
            if minimise_variable == 'theta':
                y = [func(i, delta_m_squared, alpha) for i in x ]
            elif minimise_variable == 'delta_m_squared':
                y = [func(theta, i, alpha) for i in x ]
            elif minimise_variable == 'alpha':
                y = [func(theta, delta_m_squared, i) for i in x ]
            

           # This calculates the the new guess.
            numerator = (x[2]**2-x[1]**2)*y[0] + (x[0]**2-x[2]**2)*y[1] + (x[1]**2-x[0]**2)*y[2]
            denominator =  (x[2]-x[1])*y[0] + (x[0]-x[2])*y[1] + (x[1]-x[0])*y[2]
            if denominator == 0:
                break
        
            x_3 = numerator/(2*denominator)
            
        
            diff = x_3 - x_3_prev
            x_3_prev = x_3
            
            # This adds the new guess to the list
            x.append(x_3)
            if minimise_variable == 'theta':
                y.append(func(x_3, delta_m_squared, alpha))
            elif minimise_variable == 'delta_m_squared':
                y.append(func(theta, x_3, alpha))
            elif minimise_variable == 'alpha':
                y.append(func(theta, delta_m_squared, x_3))
            
            # This removes the data of the highest value
            max_data = y.index(max(y))
            y.remove(y[max_data])
            x.remove(x[max_data])
            x.sort()
            
            # This sets the conditions that stop the iteration.
            if abs(diff) < 10**(-5) and minimise_variable == 'theta':
                run = False
                
            elif minimise_variable == 'delta_m_squared' and abs(diff) < 10**(-7):
                run = False
                
            elif minimise_variable == 'alpha' and abs(diff) < 10**(-5):
                run = False
                
            
                
        if minimise_variable == 'theta':
            return x_3, func(x_3, delta_m_squared, alpha)
        elif minimise_variable == 'delta_m_squared':
            return x_3, func(theta, x_3, alpha)
        elif minimise_variable == 'alpha':
            return x_3, func(theta, delta_m_squared, x_3)
        
        
    def univariate_cross_sec(self, initial_theta_guesses, initial_delta_m_squared_guesses, initial_alpha_guesses, function = None):
        '''
        

        Parameters
        ----------
        This function minimises a two dimensional function using univariate method.

        Parameters
        ----------
        initial_theta_guesses : list, a list of three floats
            The initial guesses of the value of he mixing angle at which the minimum exists.
            The initial guesses must satisfy the condition that the value of the function at the first and last number 
            must be greater than the value of the function at the second number.
            
        initial_delta_m_squared_guesses : list, a list of three floats
            The initial guesses of the squared mass difference value at which the minimum exists.
            The initial guesses must satisfy the condition that the value of the function at the first and last number 
            must be greater than the value of the function at the second number.
            
        initial_alpha_guesses : list, a list of three floats
            The initial guesses of the alpha value at which the minimum exists.
            The initial guesses must satisfy the condition that the value of the function at the first and last number 
            must be greater than the value of the function at the second number.
            
        function : function, optional
            The function to be minimised. The default is None, which corresponds to the negative log likelihood the Poisson case.

        


        Returns
        -------
        theta_min : float
            The value of the mixing angle at which the minimum occurs.
            
        m_min : float
            The value of the squared mass difference at which the minimum occurs.
            
        alpha_min : float
            The value of the alpha at which the minimum occurs.
            
        num_it : int
            The number of iterations this function takes to find the minimum.
       
        '''
        run = True
        # These are the inital guesses
        m_min = initial_delta_m_squared_guesses[1]
        theta_min = initial_theta_guesses[1]
        alpha_min = initial_alpha_guesses[1]
        
        theta_prev = 0
        m_prev = 0
        alpha_prev = 0
        
        num_it = 0
        
        while run:
            #This calculates the next value of theta with a fixed delta_m_squared and alpha
            theta_min, y_min = self.para_mini_nll_cross_sec(initial_theta_guesses, minimise_variable = 'theta', theta = None, delta_m_squared = m_min, alpha = alpha_min, func = function)
      
            #This calculates the next value of delta_m_squared with a fixed theta and alpha
            m_min, y_min = self.para_mini_nll_cross_sec(initial_delta_m_squared_guesses, minimise_variable = 'delta_m_squared', theta = theta_min, delta_m_squared = None, alpha = alpha_min, func = function)
            
            #This calculates the next value of alpha with a fixed theta and delta_m_squared
            alpha_min, y_min = self.para_mini_nll_cross_sec(initial_alpha_guesses, minimise_variable = 'alpha', theta = theta_min, delta_m_squared = m_min, alpha = None, func = function)
            
            # These calculates the difference between the new point and the old point for each variable.
            diff_theta = theta_min - theta_prev
            diff_m = m_min - m_prev
            diff_alpha = alpha_min - alpha_prev
            
            # This sets the condition to stop the iteration.
            if abs(diff_m) < 10**-7 and abs(diff_theta) <10**-5 and abs(diff_alpha) <10**-5:
                run = False
    
            num_it += 1
            
            m_prev = m_min
            theta_prev = theta_min
            alpha_prev = alpha_min

        return theta_min, m_min, alpha_min, num_it
    
    
    def newton(self, theta_0, delta_m_squared_0, func = None):
        '''
        This function minimises a two dimensional function using Newton method.

        Parameters
        ----------
        theta_0 : float
            The initial guess of the x value or minxing angle.
            
        delta_m_squared_0 : float
            The initial guess of the y value or square mass difference.
            
        func : function, optional
            The function being minimised. The default is None.

        Returns
        -------
        x_lst : list
            A list of the values of variables at each iterations.
        num_it : int
            Number of iterations.

        '''
        # This sets the defaluth function to be the NLL function.
        if func == None:
            func = self.neg_ll
        
        # This initialises the vaiables later used in the function.
        x = np.array([theta_0, delta_m_squared_0])
        x_lst = [x]
        H = np.zeros((len(x), len(x)))
        h = 0.00001
        num_it = 0
        run =  True
        
        while run:
            # This calculates the Hessian matrix.
            H[0,0] = (func(x[0] + 2*h, x[1]) - 2 * func(x[0] + h, x[1]) + func(x[0], x[1]))/ (h**2)
            H[1,0] = (func(x[0] + h, x[1] + h) - func(x[0] + h, x[1]) - func(x[0], x[1] + h) + func(x[0], x[1]))/(h**2)
            H[0,1] = H[1,0]
            H[1,1] = (func(x[0], x[1] + 2*h) - 2 * func(x[0], x[1] + h) + func(x[0], x[1]))/ (h**2)
            
            inverse_H = np.linalg.inv(H)
            
            # This calculates the derivatives for each variable.
            theta_deriv = (func(x[0] + h, x[1]) - func(x[0], x[1]))/h
            m_deriv = (func(x[0], x[1] + h) - func(x[0], x[1]))/h
            deriv = np.array([theta_deriv, m_deriv])
            
            num_it+=1
            x_prev = x
        
            x = x - np.matmul(inverse_H, deriv)
            x_lst.append(x)
            
            # This sets the condition for the while loop to stop.
            if x[0]- x_prev[0] <0.000001 and x[1]- x_prev[1] <0.000001:
                run = False
            
        return x_lst, num_it
        


    def newton_cross_sec(self, theta_0, delta_m_squared_0, alpha_0, func = None):
        '''
        This function minimises a three dimensional function using Newton method.

        Parameters
        ----------
        theta_0 : float
            The initial guess of the x value or minxing angle.
            
        delta_m_squared_0 : float
            The initial guess of the y value or square mass difference.
            
        alpha_0 : float
            The initial guess of the z value or cross section.
            
        func : function, optional
            The function being minimised. The default is None.

        Returns
        -------
        x_lst : list
            A list of the values of variables at each iterations.
        num_it : int
            Number of iterations.

        '''
        # This sets the default function to be the NLL with cross section
        if func == None:
            func = self.neg_ll_cross_sec
        
        
        # This initialises the variables that will be used later
        x = np.array([theta_0, delta_m_squared_0, alpha_0])
        x_lst = [x]
        H = np.zeros((len(x), len(x)))
        h = 0.00001
        num_it = 0
        run =  True
        # for i in range(100):
        while run:
            num_it += 1
            
            # This calcualtes the Hessian matrix
            H[0,0] = (func(x[0] + 2*h, x[1], x[2]) - 2 * func(x[0] + h, x[1], x[2]) + func(x[0], x[1], x[2]))/ (h**2)
            H[1,0] = (func(x[0] + h, x[1] + h, x[2]) - func(x[0] + h, x[1], x[2]) - func(x[0], x[1] + h, x[2]) + func(x[0], x[1], x[2]))/(h**2)
            H[0,1] = H[1,0]
            H[1,1] = (func(x[0], x[1] + 2*h, x[2]) - 2 * func(x[0], x[1] + h, x[2]) + func(x[0], x[1], x[2]))/ (h**2)
            H[0,2] = (func(x[0] +h, x[1], x[2] +h) - func(x[0]+h, x[1], x[2]) - func(x[0], x[1], x[2]+h) + func(x[0], x[1], x[2]))/ (h**2)
            H[1,2] =(func(x[0], x[1] + h, x[2] + h) - func(x[0], x[1] + h, x[2]) - func(x[0], x[1], x[2] + h) + func(x[0], x[1], x[2]))/(h**2)
            H[2,0] = H[0,2]
            H[2,1] = H[1,2]
            H[2,2] = (func(x[0], x[1], x[2] + 2*h) - 2 * func(x[0], x[1], x[2] + h) + func(x[0], x[1], x[2]))/ (h**2)
            
            inverse_H = np.linalg.inv(H)
            
            # This calculates the derivatives of each variable
            theta_deriv = (func(x[0] + h, x[1], x[2]) - func(x[0], x[1], x[2]))/h
            m_deriv = (func(x[0], x[1] + h, x[2]) - func(x[0], x[1], x[2]))/h
            alpha_deriv = (func(x[0], x[1], x[2]+ h) - func(x[0], x[1], x[2]))/h
            deriv = np.array([theta_deriv, m_deriv, alpha_deriv])
            
            
            x_prev = x
        
            x = x - np.matmul(inverse_H, deriv)
            x_lst.append(x)
            
            if x[0]- x_prev[0] <0.00000001 and x[1]- x_prev[1] <0.00000001 and x[2]- x_prev[2] <0.00000001:
                run = False
                
        return x_lst, num_it
        
    
    
    def error(self, theta23, delta_m_squared, alpha = None, num_variables = 2):
        '''
        This fucntion calculates the error associated to the variable by finding the curvature of 
        the negative log likelihood function at the min using finite difference approximation.

        Parameters
        ----------
        theta23 : float
            The mixing angle at the minimum.
            
        delta_m_squared : float
            The squared mass differene at the minimum.
            
        alpha : float, optional
            The cross section at the minimum for the three dimensional minimisation. The default is None.
            
        num_variables : int, optional
            The number of variables of which the errors are being calculated. This argument can only accept 1, 2 or 3. The default is 2.

        Returns
        -------
        Float or tuple
        
            This function returns the error aprroximation for each variable. Depending on the number of variables, 
             this can be returned as float for one variable and as a tuple of length three for two or three variables.

        '''
        h = 0.000001
        #This sets the conditions when to invovle cross section in the NLL calculation.
        #The following fuctions calculate the the second partial derivatives of each variable
        if num_variables == 2 or num_variables == 1:
            sec_order_theta = (self.neg_ll(theta23 + h, delta_m_squared) - 2*self.neg_ll(theta23, delta_m_squared) + self.neg_ll(theta23 - h, delta_m_squared))/ (h**2)
            sec_order_delta_m_squared = (self.neg_ll(theta23, delta_m_squared + h) - 2*self.neg_ll(theta23, delta_m_squared) + self.neg_ll(theta23, delta_m_squared - h))/ (h**2)
        
        elif num_variables == 3:
            sec_order_theta = (self.neg_ll_cross_sec(theta23 + h, delta_m_squared, alpha) 
                               - 2*self.neg_ll_cross_sec(theta23, delta_m_squared, alpha) 
                               + self.neg_ll_cross_sec(theta23 - h, delta_m_squared, alpha))/ (h**2)
            
            sec_order_delta_m_squared = (self.neg_ll_cross_sec(theta23, delta_m_squared + h, alpha) 
                                         - 2*self.neg_ll_cross_sec(theta23, delta_m_squared, alpha)
                                         + self.neg_ll_cross_sec(theta23, delta_m_squared - h, alpha))/ (h**2)
            
            sec_order_alpha = (self.neg_ll_cross_sec(theta23, delta_m_squared, alpha + h) 
                               - 2*self.neg_ll_cross_sec(theta23, delta_m_squared, alpha) 
                               + self.neg_ll_cross_sec(theta23, delta_m_squared, alpha - h))/ (h**2)
            
            alpha_error = 1/np.sqrt(sec_order_alpha)
        # This calculates the error from the second derivatives.
        theta_error = 1/np.sqrt(sec_order_theta)
        delta_m_squared_error = 1/np.sqrt(sec_order_delta_m_squared)
        
        if num_variables == 2:
            return theta_error, delta_m_squared_error
        elif num_variables == 1:
            return theta_error
        elif num_variables == 3:
            return theta_error, delta_m_squared_error, alpha_error
            
        
    def error_2(self, theta23, delta_m_squared):
        '''
        This function calculates the error in one dimension by finding the values that correspond to NLL +/- 0.5

        Parameters
        ----------
        theta23 : float
            The mixing angle.
        delta_m_squared : float
            The squared mass differnce.

        Returns
        -------
        pos_error : float
            the positive error.
        neg_error : float
            the negative error.

        '''
        positive = self.neg_ll(theta23, delta_m_squared) + 0.5
      
        
        run = True
        theta_min_pos = theta23
        theta_min_neg = theta23
        
  
        while run:
            theta_min_pos +=0.00001
            
            a=self.neg_ll(theta_min_pos, delta_m_squared)
            if a > positive:
                 pos_error = theta_min_pos - theta23
                 run = False
                 
        run_neg = True
        while run_neg:
            theta_min_neg -= 0.0001
            if self.neg_ll(theta_min_neg, delta_m_squared) > positive:
                  neg_error = theta_min_neg - theta23
                  run_neg = False
        
        
        return pos_error, neg_error
            
            
            
            
        
        
        
        
        