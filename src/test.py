import matplotlib.pyplot as plt
import numpy as np

## A noisy sine wave as query
idx = np.linspace(0, 6.28, num=100)
query = np.sin(idx) + np.random.uniform(size=100) / 10.0

## A cosine is for template; sin and cos are offset by 25 samples
template = np.cos(idx)

## Find the best match with the canonical recursion formula
from dtw import *

alignment = dtw(query, template, keep_internals=True)

## Display the warping curve, i.e. the alignment curve
ax = alignment.plot(type="threeway")

plt.savefig("output.png")
