# %% [markdown]
# # Instructions to use

# %% [markdown]
# - Python cells,"Ctrl + Enter" to run, "Shift + Enter" to run and select next cell
# - Most common error is "variable not found". A likely cause previous cells were not run.
# - 

# %%
# Download the dataset

# %%
import pandas as pd
import matplotlib.pyplot as plt

# %%
import pandas as pd

og_df = pd.read_csv('out_LED.csv')

df = og_df[og_df['repeat_trial'].isin([0, 2]) | og_df['repeat_trial'].isna()]
session_type = 7
df = df[df['session_type'].isin([session_type])]
training_level = 16
df = df[df['training_level'].isin([training_level])]

# Filter for ABL == 40
df = df[df['ABL'] == 40]

# Keep only trials where success == 1 or -1
df = df[df['success'].isin([1, -1])]

# Calculate right_db and left_db from ILD and ABL
# ILD = right_db - left_db
# ABL = (right_db + left_db) / 2
# Therefore: right_db = (ABL + ILD) / 2, left_db = (ABL - ILD) / 2
df['right_db'] = (df['ABL'] + df['ILD']) / 2
df['left_db'] = (df['ABL'] - df['ILD']) / 2

# Transform choice: choice = 2*choice - 5
df['choice'] = 2 * df['response_poke'] - 5

# Keep only specified columns with new names
df = df[['right_db', 'left_db', 'timed_RT', 'choice']]
df = df.rename(columns={'timed_RT': 'RT'})

# Remove rows where RT > 1
df = df[df['RT'] <= 1]

# %%
# save db as csv
df.to_csv('workshop_dataset_2AFC.csv', index=False)

# %%
df = pd.read_csv('workshop_dataset_2AFC.csv')



# %%
df.head()
# %%
# visualize
# print a random row
print("Random Sample Row:")
print("-" * 50)
random_row = df.sample(1)
for col in random_row.columns:
    print(f"{col:15}: {random_row[col].iloc[0]}")
print("-" * 50)
# %%
# `right_db` is the right ear sound level
# `left_db` is the left ear sound level
# `RT` is the reaction time, How long it took to make a choice
# `choice` is 1 if it chose right poke, left if it chose left poke
# Mouse is given the task of identifying which side is larger
# So, if right_db > left_db and the choice is 1, the mouse gets water
# and similarly, if left_db > right_db and the choice is -1(mouse pokes left poke), the mouse gets in the left book
# in the other cases, mouse is not rewarded

# %%
# Plot psychometric function
# to get an idea of how well mouse performed, we plot the prob of choosing side vs difficulty
# here we characterise difficulty as the difference between right_db and left_db

difficulty = (df['right_db'] - df['left_db']).unique()
difficulty.sort()
# print difficulty in ascending order
print(difficulty)

# as u can see, the difficulty is from -16 to 16 dB. Trials where the difference is large, we expect animals to more correct 
# and trials where difference is small, we expect animals to be more uncertain
# to visualize this we plot something called a "psychometric function". 
# it is a plot where prob of choosing right poke vs difficulty
# u need to complete the below psychometric function

# %%

# %%%
def plot_psychometric_function(df):
    """
    To plot a psychometric, we need
    - difficulty: right dB - left_dB on x-axis
    - prob of choosing right poke on y-axis

    return difficulty, prob_right_vector
    """

    # TODO-1: find out the range of difficulty ( hint: unique() and sort() as above)
    
    # TODO 2: 
    # for d in difficulty:
        # filter trials based on difficulty

        # calc total num of trials

        # calc num of trials where choice = 1

        # prob of choosing right = num of trials where choice = 1 / total num of trials

        # append it to array

    # return difficulty, prob_right_vector
    pass

# %%
# solution below
def plot_psychometric_function(df):
    """
    To plot a psychometric, we need
    - difficulty: right dB - left_dB on x-axis
    - prob of choosing right poke on y-axis

    return difficulty, prob_right_vector
    """

    # TODO-1: find out the range of difficulty ( hint: unique() and sort() as above)
    difficulty = (df['right_db'] - df['left_db']).unique()
    difficulty.sort()
    # TODO 2: 
    prob_right_vector = []
    for d in difficulty:
        # filter trials based on difficulty
        trials_with_d = df[df['right_db'] - df['left_db'] == d]

        # calc total num of trials
        total_trials = len(trials_with_d)

        # calc num of trials where choice = 1
        num_trials_choice_1 = len(trials_with_d[trials_with_d['choice'] == 1])

        # prob of choosing right = num of trials where choice = 1 / total num of trials
        prob_right = num_trials_choice_1 / total_trials

        # append it to array
        prob_right_vector.append(prob_right)

    # return difficulty, prob_right_vector
    return difficulty, prob_right_vector
# %%
difficulty, prob_right_vector = plot_psychometric_function(df)

# %%
# Plotting the psychometric function
plt.plot(difficulty, prob_right_vector, '-o')
plt.xlabel('Difficulty = (Right - Left dB)')
plt.ylabel('Probability of choosing right')
plt.axvline(x=0, color='r', linestyle='--', alpha=0.3)
plt.axhline(y=0.5, color='r', linestyle='--', alpha=0.3)
plt.title('Psychometric Function')
plt.show()
# %%
# Chronometric function
# next, we also need to get sense of how long animal takes to react for different difficulty
# so we plot mean RT vs difficulty
# since we know Left and right are symmetric, we can club them in abs ILD

# %%
def plot_chronometric_function(df):
    """
    To plot a chronometric, we need
    - difficulty: right dB - left_dB on x-axis
    - mean RT on y-axis

    return difficulty, mean_rt_vector
    """
    # TODO 1: find out the range of difficulty and 
    
    
    # TODO 2: 
    # mean_RT_for_each_difficulty = []
    # for d in difficulty:
        # filter trials based on difficulty
        
        # calc mean RT

        # append it to array
        
    pass


# %%
# solution
def plot_chronometric_function(df):
    """
    To plot a chronometric, we need
    - difficulty: right dB - left_dB on x-axis
    - mean RT on y-axis

    return difficulty, mean_rt_vector
    """
    # TODO 1: find out the range of difficulty ( hint: unique() and sort() as above)
    difficulty = (df['right_db'] - df['left_db']).abs().unique()
    difficulty.sort()

    # 
    
    # TODO 2: 
    mean_RT_for_each_difficulty = []
    for d in difficulty:
        # filter trials based on difficulty
        trials_with_d = df[abs(df['right_db'] - df['left_db']) == d]
        
        # calc mean RT
        mean_RT = trials_with_d['RT'].mean()

        # append it to array
        mean_RT_for_each_difficulty.append(mean_RT)
        
    return difficulty, mean_RT_for_each_difficulty
# %%
difficulty, mean_rt_vector = plot_chronometric_function(df)
# %%
# PLotting Chronometric
plt.plot(difficulty, mean_rt_vector, '-o')
plt.xlabel('Difficulty = (Right - Left dB)')
plt.ylabel('Mean RT')
plt.title('Chronometric Function')
plt.show()
# %%
