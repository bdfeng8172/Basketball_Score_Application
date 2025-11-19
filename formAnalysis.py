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

        # Initialize print cooldown if not already set
        if not hasattr(self, "last_print_time"):
            self.last_print_time = 0.0
        self.print_cooldown = 2  # seconds

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

            # Joint-specific tolerances
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

        # Key joints determine overall evaluation
        key_joints = [
            "left_knee", "right_knee",
            "left_elbow", "right_elbow",
            "left_shoulder", "right_shoulder"
        ]
        key_statuses = [joint_feedback[j]["status"] for j in key_joints if j in joint_feedback]

        if any(status == "unacceptable" for status in key_statuses):
            overall = "unacceptable"
        elif all(status == "acceptable" for status in key_statuses):
            overall = "acceptable"
        elif all(status in ["acceptable", "passable"] for status in key_statuses):
            overall = "passable"
        else:
            overall = "unacceptable"

        # Save result
        self.last_result = (overall, joint_feedback)

        # rate-limit the console output
        current_time = time.time()
        if current_time - self.last_print_time >= self.print_cooldown:
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
            self.last_print_time = current_time  

        return overall, joint_feedback




#code has to pass to the next position consecutively even if the previous position was not acceptable
# currently, the code only evaluates every second, so it may skip some positions if the user moves too quickly