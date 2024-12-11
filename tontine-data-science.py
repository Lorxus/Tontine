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

import collections as c
import math
import matplotlib.colors as mplcol
import matplotlib.patches as mplpat
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
# histogram of ending outcomes for the runs
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
    
    outcomebars = c.Counter(ends)

    mplplt.xlabel('Day')
    mplplt.ylabel('Game Deaths')
    biggest = math.ceil(max(ends) * 1.25)
    mplplt.yticks(range(0, biggest, biggest//10))
    mplplt.title('Ensemble Forecast Runs')
    mplplt.show()
    
    mplplt.bar(outcomebars.keys(), outcomebars.values(), align='center')
    mplplt.xticks(range(int(min(ends)*0.9), int(max(ends)*1.1)))
    mplplt.title('Histogram of Ensemble Forecast Outcomes')
    mplplt.show()
    return

# heatmap-style chart showing log-probabilities of maxdeaths deaths on day maxdays
# replicates THE CHART from https://docs.google.com/spreadsheets/d/1scomCAeojAeMXYI4x7dtftkxXGdvG7g8Kj7MWulscU8, except better
def the_chart(window: int, maxdeaths: int, maxdays: int): # -> Picture
    bits_matrix = np.log2(chart_matrix(maxdeaths, maxdays, window))
    bits_matrix[np.isinf(bits_matrix)] = -101
    bits_matrix = np.ceil(bits_matrix)

    for i in range(maxdays):
        for j in range(maxdeaths):
            if (bits_matrix[i, j] < -6 and bits_matrix[i, j] != -101):
                bits_matrix[i, j] = -9 
                
    # print(bits_matrix)

    colordict = {0 : 'magenta',
            -1 : 'darkred',
            -1.5 : 'red',
            -2 : 'coral',
            -2.5 : 'orange',
            -3: 'yellow',
            -4: 'limegreen',
            -5: 'teal',
            -6: 'mediumblue',
            -9: 'darkviolet',
            -101: 'black'}
    #labels = np.array('0.5', '0.25', '0.176', '0.125', '0.088', '0.063', '0.044', '0.031', '0.016', '0.0078', '0.0001', '0')
    keys = list(colordict.keys())
    colors = [colordict[key] for key in keys]

    scalefactor = 0.75 * (20/max(maxdeaths, maxdays))
    fig, ax = mplplt.subplots(figsize=(scalefactor * 1.2 * max(5, bits_matrix.shape[1]), scalefactor * max(5, bits_matrix.shape[0])))
    
    mplplt.xlabel('Day')
    mplplt.ylabel('Game Deaths')
    mplplt.title('THE CHART: log_2(p(Total Game Deaths By d Days))')

    mplplt.xticks(range(0, maxdeaths, maxdeaths//10))
    mplplt.yticks(range(0, maxdays, maxdays//10))

    # Remove axis ticks for a cleaner visualization
    # ax.set_xticks([])
    # ax.set_yticks([])
    
    # Create a grid of colored squares
    # Adding things to the axes, not the figure, is how to do it
    # The axes define a canvas - the figure is just "all the stuff to see" 
    for i in range(bits_matrix.shape[0]):
        for j in range(bits_matrix.shape[1]):
            # Get the color for the current matrix value
            # Default to a neutral color if not found in dict
            color = colordict.get(bits_matrix[i, j], 'gray')
            
            # Draw a rectangle representing the matrix entry
            rect = mplplt.Rectangle((j, i), 
                                 1, 1, 
                                 facecolor=color, 
                                 edgecolor='white', 
                                 linewidth=1)
            ax.add_patch(rect)
    
    # Set the plot limits to match matrix dimensions
    ax.set_xlim(0, bits_matrix.shape[1])
    ax.set_ylim(0, bits_matrix.shape[0])

    patchdict = {}
    for key in colordict:
        if key != -101:
            patchdict['>' + str(np.round(2**(key-1), 4))] = colordict[key]
        else:
            patchdict[0] = colordict[-101]

    legend_patches = [
            mplpat.Patch(color=color, label=str(value))
            for value, color in patchdict.items()
        ]
        
        # Add the legend to the plot and place it outside the main plot area
    mplplt.legend(handles=legend_patches, 
                title='Matrix Values', 
                loc='center left',  # Position relative to the plot
                bbox_to_anchor=(1, 0.5),  # Adjust this to fine-tune legend position
                fontsize='large',
                title_fontsize='large')
    
    # Adjust layout to prevent cutting off the legend
    mplplt.tight_layout()
    ax.set_aspect('equal')
    # ax.invert_yaxis()
    # TO DO: figure out how to get the plot to have its origin at the top left or to flip the y-axis
    mplplt.show()
    # image = [[keys.index(val) for val in row] for row in bits_matrix]

    # mplplt.imshow(image, interpolation='nearest', cmap=colormap)
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
                    return looping_m()
                    
                if command == 'i':
                    print('TO DO - you can find all the images under m or p...')
                    return looping_prompt()
            
            except ValueError as v:
                print(v)
                return looping_prompt()
    
    return

def looping_i():  # returns the specified image
    # print('Images, all in one place.')
    # print('Which image to show?')
    return True

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
                        print('What time window should I calculate the estimate of P for? Enter an int.')
                        window = int(input())
                        est_p = P_calc_recent(window)
                        print('Estimate of P from the last', window, 'days:')
                        print(est_p)

                    except ValueError:
                        print("C'mon, enter an int already.")

                days = None
                while days is None:
                    try:
                        print('How many days in the future should I make predictions for? Again, enter an int.')
                        days = int(input())
                        if days < 1:
                            print("The present is already known.")
                            return looping_m()
                    except ValueError:
                        print("C'mon, enter an int already.")
                
            if command == 's':
                print('A single run!')
                mkv = markov_run(days, window)
                print('This run of the Markov process ended with', sum(mkv), 'deaths after', days, 'days.')
                print(mkv)

                print('Enter x to quit, p to return to the main prompt, or anything else to go back to the Markov model menu.')
                command = input()
                if command == 'x':
                    print('Quitting...')
                    return
                elif command == 'm':
                    return looping_m()
                else:
                    return looping_prompt()
                
            elif command == 'e':
                print('An ensemble forecast!')
                numruns = None
                while numruns is None:
                    try:
                        print('How many runs should I forecast over? Enter an int.')
                        numruns = int(input())
                        if numruns < 2:
                            print('Run number underflow! Returning to Markov model menu...')
                            return looping_m()
                        else:
                            allruns = run_run_markov(days, window, numruns)
                            outcomes = []
                            print('Here\'s all the outcomes:')
                            for line in allruns:
                                print(line)
                                outcomes.append(sum(line))
                            ensemble_plot(numruns, window, days)
                            mean = sum(outcomes)/numruns
                            stddev = np.std(outcomes)
                            print('Over the next', days, 'days, the mean number of deaths is predicted to be', np.round(mean, 2), 'and the standard deviation is', np.round(stddev, 2), 'such that the true number of deaths is ~95'+'%'+' likely to be between', mean-2*stddev, 'and', str(mean+2*stddev)+'.')
                    except ValueError:
                        print("C'mon, enter an int already.")

            else:  # command == 'c'
                print('Prepare for... THE CHART!')
                maxdeaths = None
                while maxdeaths is None:
                    try:
                        print('How large should the max number of player deaths to show be? Enter an int.')
                        maxdeaths = int(input())
                        if maxdeaths < days/10:
                            print('Trust me, you don\'t want your max death count to be that small. Try something positive, or at least closer to the max day count.')
                            return looping_m()
                        the_chart(window, maxdeaths, days)
                        return looping_prompt()

                    except ValueError:
                        print("C'mon, enter an int already.")
                


        except ValueError as v:
            print(v)
            return looping_m()
     
def looping_p():  # returns P eventually.
    print('What time window should I calculate the estimate of P for? Enter an int.')
    # to add: submenu for the linear-decreasing rather than flat estimate of P
    while True:
        try:
            window = int(input())
            est_p = P_calc_recent(window)
            print('Estimate of P from the last', window, 'days:')
            print(est_p)
            return looping_prompt()

        except ValueError:
            print("C'mon, enter an int already.")
            return looping_p()
    
    
# main

if __name__ == "__main__":
    print('Lorxus\'s Tontine Analysis Program')
    print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
    print('https://tontine.cash/')
    print('https://github.com/Lorxus/Portfolio/')
    print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
    
    # looping_prompt()
    
    the_chart(350, 40, 40)

    # print(P_calc_recent(180))
    # print(markov_run(30, 180))
    # print(run_run_markov(50, 200, 8))
    # for entry in chart_matrix(20, 20, 100):
    #     np.set_printoptions(precision=3, suppress=True)
    #     print(entry)
    
    # ensemble_plot(32, 200, 50)
    # the_chart(340, 20, 20)