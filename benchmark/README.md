# Object Detection Models Testing

## Installation

Run the following command to install necessary dependencies:

```
pip install -r requirements.txt
```

## Usage

### Count and move failed images when detecting

The failed images are classified as three following categories:

1. **WRONG_NUMBER_BOXES**: If the detection results have different number of objects to the groundtruth results.

2. **LOW_IOU_BOXES**: If the detection results have the same number of objects. But the boxes of detection and groundtruth have low IOU scores.

3. **WRONG_CLASSES_BOXES**: If the detection and groundtruth have the same number objects, high IOU score. But the model classifies wrong classes.

To run with your dataset, go to `eval_config.yaml` and change the path of dataset and the dataset configuration file (`data.yaml` file when generating with roboflow). And change the model based on your project.

To move the failed image from dataset to a new folder. Change the path of `relabel_folder` field. Otherwise leave it as null.

When you have done with configuration. Run:

```
python get_failed_cases.py
```

### Calculate mAP, Precision, Recall

The configuration step is same as above.

```
python metric_py_pt.py
```

