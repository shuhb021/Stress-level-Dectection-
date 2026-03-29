import pandas as pd
import numpy as np
import time

# Create a dummy data211.csv if not exists, or just test on a string
pixel_sequence = "36 139 142 143 142"
start = time.time()
face = np.array(pixel_sequence.split(' '), dtype='float32')
end = time.time()
print(f"Split took {end-start}")

start = time.time()
face2 = np.fromstring(pixel_sequence, sep=' ', dtype='float32')
end = time.time()
print(f"fromstring took {end-start}")

print(f"Equal? {np.array_equal(face, face2)}")
