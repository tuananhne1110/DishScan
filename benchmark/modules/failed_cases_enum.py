from enum import Enum


class FailedCases(Enum):
    WRONG_NUMBER_BOXES = 0
    LOW_IOU_BOXES = 1
    WRONG_CLASSES_BOXES = 2
    NOTHING = 3


class FailedCasesFolder(Enum):
    WRONG_NUMBER_BOXES = "1_wrong_number"
    LOW_IOU_BOXES = "2_low_iou"
    WRONG_CLASSES_BOXES = "3_wrong_classes"  
