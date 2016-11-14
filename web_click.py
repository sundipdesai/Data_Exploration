import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from numpy import linalg as LA
import seaborn as sns

'''
MSNBC Web Click Data
--------------------
* Exploratory Analysis of anonymous web click data

* Build a predictor to predict
  the probability what the next web page the user will
  click given their previous web page clicks

Data set taken from:
https://archive.ics.uci.edu/ml/datasets/MSNBC.com+Anonymous+Web+Data

'''

# -- Preliminary Data Gathering -- #
# Read in data
data = pd.read_csv('msnbc990928.seq', header=None)

# Gather topics
topics = (str.split(data[0][1]))

# Dictionarize topics
web_topics = {k+1:v for k, v in enumerate(topics)}

# Find the row value that contains an integer and use that below instead of using 3

# Fetch clicks, then transform to integers
clicks = [str.split(data[0][k]) for k in np.arange(3,len(data))]
clicks = [list(map(int, row)) for row in clicks]


# --- Total Click Counts --- #
keys = list(web_topics.keys())
click_counts = {key: 0 for key in keys}

for x in clicks:
    for y in x:
        click_counts[y] += 1

pages = click_counts.keys()
total_hits_per_page = click_counts.values() # Total clicks per web page
total_hits = np.sum(total_hits_per_page) # Aggregate total clicks in data set
click_proportion = [ round((float(x)/total_hits),3) for x in total_hits_per_page] #proportion of clicks relative to total

click_proportion_ = {k:v for k, v in enumerate(click_proportion)} # dictionarize click proportion

c={}
for key in sorted(click_proportion_.keys()):
    c[key] = click_proportion_[key]

cv = c.values()
ck = c.keys()

# -- Average click stream per user -- #
total_click_stream = 0
for x in clicks:
    total_click_stream += len(x)
average_click_stream = round(total_click_stream/float(len(clicks)),2)
print "Average click stream per user: ", average_click_stream

# -- Highest Repeated Web Page click -- #
repeat_clicks = np.zeros(len(pages))

# User click repeats
# This is the aggregate count of number of repeated clicks within a single topic
for click in clicks:
    sort_click = np.sort(click)
    value = -1
    for i, y in enumerate(sort_click):
        if i < len(sort_click)-1:
            if y == sort_click[i+1] and y != value:
                repeat_clicks[y-1] += 1
                value = y
#print("Repeat Clicks: ", repeat_clicks)

# First page clicks
# Check to see which page is visited
# first --> should be front page
first_page_clicks = np.zeros(len(pages))
for click in clicks:
    first_page_clicks[click[0] - 1] += 1


# Last page clicks
# Which is the last page the user visits before
# they finish a session on MSNBC?
last_page_clicks = np.zeros(len(pages))
for click in clicks:
    last_page_clicks[click[-1] - 1] += 1

# Largest single repetition of clicks within a single topic



'''
Build a web page click predictor
'''

# -- Markov Transition Matrix -- #
'''
Count # of sequence changes from A -> B for all
transitions. There are 16 states (pages)
therefore there should be 16*16 = 256 elements in the
transition matrix

Also count the initial states for every user. This will
allow us to compute prior probabilities!

'''

state_transition_counts = np.zeros((len(pages), len(pages)))
initial_states = np.zeros(len(pages))

for click in clicks:
    if click: initial_states[click[0]-1] += 1

for counts in clicks:
    for x, y in enumerate(counts):
        if len(counts) > 1 and x >=1:
            state_transition_counts[counts[x-1]-1][y-1] += 1


state_transition_probs = np.zeros((len(pages), len(pages)))

# Transition Probabilities
for x, counts in enumerate(state_transition_counts):
    sumCounts = sum(counts)
    for y, element in enumerate(counts):
        state_transition_probs[x][y] = element/sumCounts


# Use the logic below to get the state probs after N transitions
x = np.array([1, 0 ,1])
y = np.matrix( [[1, 1, 1], [1, 1, 1], [0,0,0]])

# ------
#  Sequence predictor
# Function predictSeq
#
# Inputs:
# -------
# M - Transition Matrix
# N - Current State
# n - Number of state transitions
# Return:
# --------
# State probabilities after n transitions
# -------
def predictSeq(M, N, n):
    A = LA.matrix_power(M, n)
    return N.dot(A)

# -- Example use of sequence predictor -- #
nn = np.zeros(len(pages)) # Create vector of empty states
nn[5] = 1                 # Let's say the last state (webpage) visited in the
                          # sequence was state 1 i.e., Front Page

# Call predict sequence function to fetch probabilities
# of what the next webpage click will be after
# n transitions.
print("Probabilities of next web page: \n")

for i, topic in enumerate(topics):
    print topic, ': ', round(100*predictSeq(state_transition_probs, nn, 1)[i],2), '%'

'''
# ------------------------- #
           Plots
# --------------------------#
'''
# Set seaborn color codes
sns.set(color_codes=True)

plt.figure(figsize=(12, 9))

# Axis preferences
ax = plt.subplot(111)
ax.spines["top"].set_visible(False)
ax.spines["bottom"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.spines["left"].set_visible(False)


ylim = 1000
xlim = 0.5
plt.ylim(0, max(first_page_clicks) + ylim)
plt.xlim(-0.25, len(first_page_clicks) - xlim)

width = 0.35
height_text = 0.2

# Ensure that the axis ticks only show up on the bottom and left of the plot.
# Ticks on the right and top of the plot are generally unnecessary chartjunk.

for x, y in enumerate(first_page_clicks):
    if y != max(first_page_clicks):
        ax.bar(x, y, width, color='blue', edgecolor = "none")

    else:
        ax.bar(x, y, width, color='red', edgecolor = "none")

plt.xticks(np.arange(0,len(topics)) + width/2, topics, rotation="vertical")
yticks = np.arange(0, round(max(first_page_clicks)+20000), 20000)
plt.yticks(yticks)

plt.tick_params(
    axis='x',          # changes apply to the x-axis
    which='both',      # both major and minor ticks are affected
    bottom='off',      # ticks along the bottom edge are off
    top='off',         # ticks along the top edge are off
    )


ax.set_title('MSNBC First Page Clicks', fontsize=14)
fig = plt.gcf()
fig.subplots_adjust(bottom=0.15)
plt.text(0,-47000, "Data Set: MSNBC Anonymous Web Data (c/o David Heckerman)", fontsize=8, ha="left")


plt.show()


# Total Clicks per Topic sans Front Page
plt.figure(figsize=(13,8))
g=sns.barplot(range(len(total_hits_per_page)-1), total_hits_per_page[1:])
g.set( xticklabels=topics[1:], ylabel="Total Clicks", title="Total Clicks per Topic", xlabel="Topics")
plt.text(0,-47000, "Data Set: MSNBC Anonymous Web Data (c/o David Heckerman)", fontsize=8, ha="left")
plt.show()

# Total Last Page Clicks
plt.figure(figsize=(13,8))
g=sns.barplot(range(len(last_page_clicks)), last_page_clicks)
g.set( xticklabels=topics, ylabel="Total Clicks", title="Total Clicks of Last Page per Topic", xlabel="Topics")
plt.text(0,-47000, "Data Set: MSNBC Anonymous Web Data (c/o David Heckerman)", fontsize=8, ha="left")
plt.show()
