

#  this is just a copy in case something goes wrong with the original file


import numpy as np

class FormAnalysis:
    def __init__(self):
        # Reference (ideal) joint angles for each shooting position
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

        # Tolerances for grading accuracy
        self.acceptable_std = {"default": 4, "knee": 6}
        self.passable_std = {"default": 10, "knee": 12}

    def evaluate_position(self, smoothed_angles, position_name):
        """
        Evaluate how close the current smoothed angles are to a target shooting position.
        Returns 'acceptable', 'passable', or 'unacceptable' for that position.
        """
        ref_angles = self.positions[position_name]
        status_per_joint = {}
        acceptable = True
        passable = False

        for joint, ideal_angle in ref_angles.items():
            current_angle = smoothed_angles.get(joint)
            if current_angle is None:
                status_per_joint[joint] = "no data"
                continue

            diff = abs(current_angle - ideal_angle)

            # Use knee-specific tolerances if applicable
            if "knee" in joint:
                acc_tol = self.acceptable_std["knee"]
                pas_tol = self.passable_std["knee"]
            else:
                acc_tol = self.acceptable_std["default"]
                pas_tol = self.passable_std["default"]

            # Determine state
            if diff <= acc_tol:
                status = "acceptable"
            elif diff <= pas_tol:
                status = "passable"
                acceptable = False
                passable = True
            else:
                status = "unacceptable"
                acceptable = False

            status_per_joint[joint] = status

        # Determine overall position status
        if acceptable:
            overall = "acceptable"
        elif passable:
            overall = "passable"
        else:
            overall = "unacceptable"

        # Print evaluation summary
        print(f"\n{position_name.upper()} Evaluation:")
        for joint, status in status_per_joint.items():
            print(f"  {joint}: {status}")
        print(f"→ Overall: {overall.upper()}")

        return overall, status_per_joint
