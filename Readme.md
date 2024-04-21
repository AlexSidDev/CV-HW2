# Implementation of Conv2d.
In this repository you can find naive implementation (conv2d.py file) and im2col implementation (im2col.py file) <br>
## Installation

To install all dependences please run
    
    pip install -r requirements.txt

## Testing

To run tests of conv2d please run

    python3 test.py

There are 2 variants of tests: first compares naive implementation and PyTorch version. Second compares naive and im2col implementations. All tests run with strides in range [1, 5]. First test also checks kernel with both odd and even kernel size. 