# person-detection
person detection using tensorflow (pre trained mobilenet ssd). 
works on both thermal and normal images.

## How to setup
### 1. seup vitual environment (optional but recomended)
```
python -m venv env
source env/bin/activate
```
### 2. Install dependencies
`pip install -r requirements.txt`

## How to run

usage
```
usage: detect-person.py [-h] [-s] [-m MODE] input

person detection v1.0

positional arguments:
  input                 Input image file name

optional arguments:
  -h, --help            show this help message and exit
  -s, --save            Save processed image
  -m MODE, --mode MODE  Select mode 1-image, 2-video
```
### Example
```
python detect-person.py -m 1 test/th.jpg
```
