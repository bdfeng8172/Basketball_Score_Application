from .formAnalysis import FormAnalysis

class BasketballFormAnalysis(FormAnalysis):

# position 1 - 3 are arbitrary positions that follow chronological order of a basketball shot
# in this case it is used as the "perfect form" reference for a basketball shot
# each number is an angle

    def __init__(self):
        super().__init__()

        self.positions = {
            "position_1": {
                "left_wrist": 71, "left_elbow": 100, "left_shoulder": 30,
                "left_knee": 165, "left_ankle": 165,
                "right_wrist": 114, "right_elbow": 130, "right_shoulder": 27,
                "right_knee": 168, "right_ankle": 161
            },
            "position_2": {
                "left_wrist": 31, "left_elbow": 86, "left_shoulder": 56,
                "left_knee": 157, "left_ankle": 163,
                "right_wrist": 53, "right_elbow": 110, "right_shoulder": 58,
                "right_knee": 153, "right_ankle": 162
            },
            "position_3": {
                "left_wrist": 19, "left_elbow": 113, "left_shoulder": 102,
                "left_knee": 161, "left_ankle": 161,
                "right_wrist": 31, "right_elbow": 129, "right_shoulder": 107,
                "right_knee": 150, "right_ankle": 160
            }
        }
