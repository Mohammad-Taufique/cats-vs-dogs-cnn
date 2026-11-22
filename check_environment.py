
import sys
import tensorflow as tf
import numpy as np
import matplotlib
import sklearn
import streamlit as st
from PIL import Image

print("PYTHON:", sys.version)
print("TENSORFLOW:", tf.__version__)
print("NUMPY:", np.__version__)
print("MATPLOTLIB:", matplotlib.__version__)
print("SKLEARN:", sklearn.__version__)
print("STREAMLIT:", st.__version__)

print("\nAvailable devices:")
print(tf.config.list_physical_devices())

print("\nAll libraries imported successfully!")
