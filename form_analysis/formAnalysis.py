import numpy as np

class FormAnalysis:


    def __init__(self):
        # Default tolerances shared across sports
        self.acceptable_std = {"default": 4, "knee": 6}
        self.passable_std = {"default": 10, "knee": 12}

        # Placeholder for sport-specific positions
        self.positions = {}

    def evaluate_position(self, smoothed_angles, position_name):

        if position_name not in self.positions:
            print(f" Unknown position: {position_name}")
            return None, {}

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

            # Use knee-specific tolerances
            if "knee" in joint:
                acc_tol = self.acceptable_std["knee"]
                pas_tol = self.passable_std["knee"]
            else:
                acc_tol = self.acceptable_std["default"]
                pas_tol = self.passable_std["default"]

            # Determine joint status
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

        # Overall classification
        if acceptable:
            overall = "acceptable"
        elif passable:
            overall = "passable"
        else:
            overall = "unacceptable"

        # Print summary
        print(f"\n{position_name.upper()} Evaluation:")
        for joint, status in status_per_joint.items():
            print(f"  {joint}: {status}")
        print(f"→ Overall: {overall.upper()}")

        return overall, status_per_joint

# next code will make it so that the display will actually show:
# acceptable in green, passable in yellow, and unacceptable in red
#currently code runs continuously but does not have any type of buffer to slow down the output