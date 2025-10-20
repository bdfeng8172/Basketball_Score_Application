import time
import numpy as np

class FormAnalysis:

    def __init__(self, evaluation_interval=1.0):
        # Default tolerances shared across sports
        self.acceptable_std = {"default": 4, "knee": 6}
        self.passable_std = {"default": 10, "knee": 12}

        # Minimum time (in seconds) between evaluations
        self.evaluation_interval = evaluation_interval
        self.last_evaluation_time = 0.0

        # Placeholder for sport-specific positions
        self.positions = {}

        # Keep previous result to reuse when skipping frames
        self.last_result = None

        # Define color codes for each category (BGR for OpenCV)
        self.color_map = {
            "acceptable": (0, 255, 0),    # green
            "passable": (0, 255, 255),    # yellow
            "unacceptable": (0, 0, 255),  # red
            "no data": (128, 128, 128)    # gray
        }

    def evaluate_position(self, smoothed_angles, position_name):


        current_time = time.time()
        if current_time - self.last_evaluation_time < self.evaluation_interval:
            # Too soon — skip evaluation, reuse last result
            return self.last_result if self.last_result else (None, {})

        self.last_evaluation_time = current_time  # update timestamp

        if position_name not in self.positions:
            print(f"Unknown position: {position_name}")
            return None, {}

        ref_angles = self.positions[position_name]
        joint_feedback = {}
        acceptable = True
        passable = False

        for joint, ideal_angle in ref_angles.items():
            current_angle = smoothed_angles.get(joint)
            if current_angle is None:
                joint_feedback[joint] = {"status": "no data", "color": self.color_map["no data"]}
                continue

            diff = abs(current_angle - ideal_angle)

            # Knee-specific tolerance
            if "knee" in joint:
                acc_tol = self.acceptable_std["knee"]
                pas_tol = self.passable_std["knee"]
            else:
                acc_tol = self.acceptable_std["default"]
                pas_tol = self.passable_std["default"]

            # Determine joint classification
            if diff <= acc_tol:
                status = "acceptable"
            elif diff <= pas_tol:
                status = "passable"
                acceptable = False
                passable = True
            else:
                status = "unacceptable"
                acceptable = False

            color = self.color_map[status]
            joint_feedback[joint] = {"status": status, "color": color}

        # Determine overall rating based on key joints
        key_joints = ["left_knee", "right_knee",
                      "left_elbow", "right_elbow",
                      "left_shoulder", "right_shoulder"]

        # Extract statuses for key joints only
        key_statuses = [joint_feedback[j]["status"] for j in key_joints if j in joint_feedback]

        # If any key joint is "unacceptable" → overall unacceptable
        if any(status == "unacceptable" for status in key_statuses):
            overall = "unacceptable"
        # If all key joints are acceptable → overall acceptable
        elif all(status == "acceptable" for status in key_statuses):
            overall = "acceptable"
        # If all key joints are either acceptable or passable → overall passable
        elif all(status in ["acceptable", "passable"] for status in key_statuses):
            overall = "passable"
        else:
            overall = "unacceptable"


        # Save result for reuse
        self.last_result = (overall, joint_feedback)

        # Console feedback
        print(f"\n{position_name.upper()} Evaluation:")
        for joint, info in joint_feedback.items():
            status = info["status"]
            color_name = (
                "Green" if status == "acceptable" else
                "Yellow" if status == "passable" else
                "Red" if status == "unacceptable" else "Gray"
            )
            print(f"  {joint}: {status} ({color_name})")
        print(f"→ Overall: {overall.upper()}")

        return overall, joint_feedback



#code has to pass to the next position consecutively even if the previous position was not acceptable
# currently, the code only evaluates every second, so it may skip some positions if the user moves too quickly