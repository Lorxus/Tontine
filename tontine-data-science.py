# https://tontine.cash/ is a MSCHF game. 
# On 2021-12-27, 7141 players each paid $10 into a central pot and have been required to check in at least once a day by midnight EST/EDT or else be eliminated. 
# The last remaining player will win the pot.
# 
# One can model player (game) death rates as a Markov process, with the distribution of deaths/day given by a vector P of the (truncated) form:
# list[p_0 p_1 p_2 p_3(...)]
# (although later on I might revise this to check that the data reflect a truly memoryless process - two days with deaths in succession are rarely lately than you would expect.)
# 
# Average daily death rates (as of 2024-12-2) have been steadily on the decline from 0.5/d earlier this year to 0.25/d now, such that
# P ~ [0.72 0.23 0.041 0]
# (although later on I might revise this to account for the linear-ish decline and have not yet looked for patterns in changes in P.)
# 
# This program is a quick proof of concept to use numpy and matplotlib to load in the entire population count data since the beginning of the game
# and then process it to calculate parametrized guesses at P, and also generate Markov chain-based predictions.
# 
# It uses numpy for the analysis, and matplotlib for graphs.
# Much of this analysis is duplicated from an existing google sheet you can probably find somewhere at https://github.com/Lorxus/Portfolio/.

import math
import matplotlib.pyplot as mplplt
import numpy as np
import random as r

FILENAME = "tontinepop.txt" # contains a list of all population counts in Tontine since 2021-12-27.
rawfile = open(FILENAME, 'r')  # I might eventually make this be able to autoupdate, given an internet connection.
print('Reading file...')

file = [int(line) for line in rawfile]


# basic data digestion and manipulation

def deathcounts() -> list[int]:
    deathcounts = []
    
    for i in range(len(file)-1):
        deathcounts.append(file[i] - file[i+1])
    
    return deathcounts

def P_calc_recent(window: int) -> list[float]:  # given a time window to sample past data from, outputs the relative frequency of death counts on a given day.
    d = deathcounts()
    
    if window > len(d):
        window = len(d)
        print('Overflow - setting window to max length.')
    elif window <= 0:
        window = 1
        print('Underflow - setting window to min length.')
    
    d = d[-window:]  # take only the last time window 
    countfreq = [0] * (max(d) + 1)

    for count in d:
        countfreq[count] += 1  # increment the entry corresponding to the death count 
    
    for i in range (len(countfreq)):
        countfreq[i] /= window

    return countfreq

def markov_run(days: int, window: int) -> list[int]:  # given the number of days to run over and the time window to sample past data, outputs a possible future trajectory of death totals by day.
    markov_vector = P_calc_recent(window)
    max_d = len(markov_vector)

    deaths = []
    for i in range(days):
        diceroll = r.random()  # it is cast

        for i in range(max_d):
            if sum(markov_vector[:i]) <= diceroll:
                outcome = i
        
        deaths.append(outcome)
    
    return deaths

def run_run_markov(days: int, window: int, numruns: int) -> list[list[int]]:  # as markov_run, but outputs an entire numruns-length list of possible future trajectories.
    runs = []
    for i in range(numruns):
        runs.append(markov_run(days, window))

    return runs

def chart_matrix(maxdeaths: int, maxdays: int, window: int) -> list[list[float]]:  # actually, np.zeros(shape=(maxdays, maxdeaths), dtype=np.float32)
    markov_vector = P_calc_recent(window)                                          # outputs a matrix whose entries [i, j] are the probability of exactly j deaths on day i.
    max_single_d = len(markov_vector)
    prob_matrix = np.zeros(shape=(maxdays, maxdeaths), dtype=np.float32)

    for i in range(max_single_d):
        prob_matrix[0, i] = markov_vector[i]

    for i in range(1, maxdays):
        for j in range(maxdeaths):
            # print(i, j)
            # print(prob_matrix[i-1, max(0, j-max_single_d+1):min(j, maxdeaths)+1])
            # print(np.shape(prob_matrix[i-1, max(0, j-max_single_d+1):min(j, maxdeaths)+1]), np.shape(np.flip(markov_vector[0:min(j, max_single_d)+1])))
            prob_matrix[i, j] = np.dot(prob_matrix[i-1, max(0, j-max_single_d+1):min(j, maxdeaths)+1], np.flip(markov_vector[0:min(j, max_single_d)+1]))
    
    return prob_matrix


# plots and pictures

# line graph showing different runs over the days
def ensemble_plot(runs: int, window: int, maxday: int):  # -> Picture
    runs = run_run_markov(maxday, window, runs)
    cumruns = []
    ends = []

    for run in runs:
        this_cumrun = []
        for i in range(len(run)):
            this_cumrun.append(sum(run[:i+1]))
        cumruns.append(this_cumrun)

    for run in cumruns:
        ends.append(run[-1])
        x = np.array(range(maxday))
        temp_y = run

        mplplt.plot(x, temp_y)
    
    mplplt.xlabel('Day')
    mplplt.ylabel('Game Deaths')
    biggest = math.ceil(max(ends) * 1.25)
    mplplt.yticks(range(0, biggest, biggest//10))
    mplplt.show()
    
    mplplt.hist(ends)
    mplplt.show()
    return

def the_chart(window: int, maxdeaths: int, maxdays: int): # -> Picture
    # heatmap-style chart showing probabilities of maxdeaths deaths on day maxdays 
    return



# quick little user interface 

def looping_prompt():
    while True:
            try:
                print('Enter i for images, m for the Markov model menu, p for the P calc menu, or x to exit.')
                command = input() 
                if command not in ['i', 'm', 'p', 'x']:
                    raise ValueError("That's not one of the options.")
                
                if command == 'x':
                    print('Quitting!')
                    return
                
                if command == 'p':
                     return looping_p()
                
                if command == 'm':
                    print('TO DO')
                    # return looping_m()
                    return
                
                if command == 'i':
                    print('TO DO')
                    # return looping_i()
                    return
            
            except:
                return looping_prompt()
    
    return

def looping_m():  # returns a single MC run, an ensemble run, or the whole probability chart, depending on commands.
    while True:
        try:
            print('Enter s for a single Markov chain run, e for an ensemble of runs, or c for a chart of probabilities by day and total death count.')
            command = input() 
            if command not in ['c', 'e', 's', 'x']:
                raise ValueError("That's not one of the options.")
            
            if command == 'x':  # deliberately not presented as an obvious option 
                    print('Quitting!')
                    return
            
            else:  # no need to do this three times
                window = None
                while window is None:
                    try:
                        print('How long should I sample back to, to estimate P? Enter an int.')
                        window = int(input())
                        if type(window) is not int or float:
                            raise TypeError("C'mon, enter an int already.")
                        elif type(window) is float:
                            raise TypeError("No, a float's not good enough. An int. INT.")
                        else:
                            est_p = P_calc_recent(window)
                            print('Estimate of P from the last', window, 'days:')
                            print(est_p)
                    except:
                        return True  # TO DO

                days = None
                while days is None:
                    try:
                        print('How many days in the future should I make predictions for? Again, enter an int.')
                        days = input()
                        if type(days) is not int or float:
                            raise TypeError("C'mon, enter an int already.")
                        elif type(days) is float:
                            raise TypeError("No, a float's not good enough. An int. INT.")
                        elif days < 1:
                            raise ValueError("The present is already known.")
                    except TypeError or ValueError:
                        return looping_m()
                
            if command == s:
                print('A single run!')
        except TypeError or ValueError:
            return looping_m()
        
    print('How many runs')


def looping_p():  # returns P eventually.
    print('What time window should I calculate the estimate of P for? Enter an int.')
    # to add: submenu for the linear-decreasing rather than flat estimate of P
    while True:
        try:
            window = input()
            if type(window) is not int or float:
                raise TypeError("C'mon, enter an int already.")
            elif type(window) is float:
                raise TypeError("No, a float's not good enough. An int. INT.")
            else:
                est_p = P_calc_recent(window)
                print('Estimate of P from the last', window, 'days:')
                print(est_p)
                return est_p

        except TypeError:
            return looping_p()

    return looping_prompt()
    
    
# main

if __name__ == "__main__":
    print('Lorxus\'s Tontine Analysis Program')
    print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
    print('https://tontine.cash/')
    print('https://github.com/Lorxus/Portfolio/')
    print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
    



    # print(P_calc_recent(180))
    # print(markov_run(30, 180))
    # print(run_run_markov(50, 200, 8))
    for entry in chart_matrix(10, 10, 500):
        np.set_printoptions(precision=3, suppress=True)
        print(entry)
    
    ensemble_plot(32, 200, 50)