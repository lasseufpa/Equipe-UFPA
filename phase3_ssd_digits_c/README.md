# Phase 3 - SSD (36 classes)

## Dependencies
1. Python 3.6+
2. OpenCV
3. Pytorch 1.0 or Pytorch 0.4+
4. Caffe2
5. Pandas
6. Boto3 if you want to train models on the Google OpenImages Dataset.

## 3) Evaluating

To evaluate ssd model on [digitsC_test](https://nextcloud.lasseufpa.org/s/YTTWt6wPYC3XnCD) you need to download the [model](https://nextcloud.lasseufpa.org/s/7WJnNCirCWqeCyJ) and run the folling command.

```
python eval_ssd.py --net vgg16-ssd  --dataset <path-to-test-set> --trained_model <path-to-the-model-file> --label_file models./petrobras/petrobras-labels.txt 
```

## 2) Using SSD

To use ssd to detec classes from [digitsC_test](https://nextcloud.lasseufpa.org/s/YTTWt6wPYC3XnCD) you need to download the [model](https://nextcloud.lasseufpa.org/s/7WJnNCirCWqeCyJs) and run the folling command.

```bash
python run_ssd_example.py vgg16-ssd <path-to-the-model-file> ./models/petrobras-labels.txt <path-to-image_test>
```