# OCR-TextRecog
useful text recognition algorithms, CRNN and SVTR text recognition

### Paper

- [CRNN](https://arxiv.org/abs/1507.05717)
- [SVTR](https://arxiv.org/abs/2205.00159)

### Evaluation DataSet

It is a public dataset containing 509164, 63645 and 63646 training, validation, and test images.
[ Chinese Dataset Scene Images](https://github.com/fudanvi/benchmarking-chinese-text-recognition#download)

### Implementation Details

The training method of SVTR papers is used !
The difference is that the text category of this project is 5881 (the text category of Scene Images data), while the paper category is 6625。
The cosine learning rate scheduler with 5 epochs linear warm-up is used in all 100 epochs. Data augmentation was not used for training。

### Some Result


| algorithms | backbone | STN | Val ACC（scene_val） |Test ACC（scene_test）|
| ------- | --------- | ------ | ----- | ----- |
| CRNN  | vgg(paper)   | N  | - |53.4|
| CRNN  | RepVGG-A0 | N | 55.90 |55.87|
| CRNN  | RepVGG-A0   | Y   | 57.80|57.80|
| CRNN  | mobilev3    | Y   | 51.50|51.40|
| CRNN  | repmobilev3 | Y   | 0|0|
| CRNN  | lcnet     | Y   | 54.75|54.57|
| CRNN  | replcnet     | Y  | 55.21|54.83|
| SVTR  | Tiny(paper)     | Y  | -|67.90|
| SVTR  | Tiny     | N | 63.01|62.86|
| SVTR  | Tiny     | Y | 69.18|69.06|

### Reference Resources
- https://github.com/PaddlePaddle/PaddleOCR
- https://github.com/DingXiaoH/RepVGG
- https://github.com/BADBADBADBOY/pytorchOCR
