from .formAnalysis import FormAnalysis

class BasketballFormAnalysis(FormAnalysis):

# position 1 - 3 are arbitrary positions that follow chronological order of a basketball shot
# in this case it is used as the "perfect form" reference for a basketball shot
# each number is an angle

    def __init__(self):
        super().__init__()

        self.positions = {
            "position_1": {
                "left_wrist": 33, "left_elbow": 83, "left_shoulder": 50,
                "left_knee": 154, "left_ankle": 160,
                "right_wrist": 38, "right_elbow": 81, "right_shoulder": 45,
                "right_knee": 147, "right_ankle": 160
            },
            "position_2": {
                "left_wrist": 10, "left_elbow": 96, "left_shoulder": 100,
                "left_knee": 135, "left_ankle": 149,
                "right_wrist": 60, "right_elbow": 100, "right_shoulder": 104,
                "right_knee": 120, "right_ankle": 160
            },
            "position_3": {
                "left_wrist": 3, "left_elbow": 120, "left_shoulder": 115,
                "left_knee": 168, "left_ankle": 155,
                "right_wrist": 21, "right_elbow": 109, "right_shoulder": 137,
                "right_knee": 170, "right_ankle": 145
            }
        }
