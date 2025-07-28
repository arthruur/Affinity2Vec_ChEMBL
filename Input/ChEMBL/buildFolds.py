"""
Angelo June, 2024

Create the indexes for the cross validation sets.


"""
import random
import numpy as np


# Number of folds. 5 Initial used for train/validation and the Last fold used for test.
folds = 6
fpath= "Input/ChEMBL/folds/"

print("Reading dataset...")
dataset= open("Input/ChEMBL/affinitiesChEMBL34-filtered.tsv",'r')
content = dataset.readlines()
dataset.close()
total_samples= (len(content) - 1) # discard the header
# total_samples = 22
samples_per_fold = int(total_samples/folds)
print(total_samples,"samples -", folds, "folds with", samples_per_fold, "samples per fold")

indexes= range(total_samples)
samples= random.sample(indexes,total_samples)
# print(samples)

train_idx=np.zeros((folds-1,samples_per_fold))
# print(train_val.shape)
# Initial folds used for train/validation
for fold in range(folds-1):
    i= fold*samples_per_fold
    # print("fold",fold,'\t',samples[i:i+samples_per_fold])
    train_idx[fold]= np.asarray(samples[i:i+samples_per_fold])

# print(train_val)

print("Writing train/validation indexes...")
np.savetxt(fpath+'train_val_idx.txt',train_idx.astype(int), fmt='%i')

# last fold get the remaing samples for test
fold+= 1
test = np.asarray(samples[fold*samples_per_fold:])
# print(test)
print("Writing test indexes...")
np.savetxt(fpath+'test_idx.txt',test.astype(int), fmt='%i')

