import mediapipe as mp
import cv2
import numpy as np
from form_analysis.basketball_FA import BasketballFormAnalysis

# initialize form analyzer
form_analyzer = BasketballFormAnalysis()

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

# smoothing factor (higher = more responsive, lower = smoother)
ALPHA = 0.1

# store previous smoothed angles
smoothed_angles = {
    'left_elbow': None, 'left_wrist': None, 'left_knee': None, 'left_ankle': None,
    'right_elbow': None, 'right_wrist': None, 'right_knee': None, 'right_ankle': None,
    'left_shoulder': None, 'right_shoulder': None
}


#  specific 2D angle calculation function
def calculate_angle_2d(a, b, c):

    a, b, c = np.array(a[:2]), np.array(b[:2]), np.array(c[:2])  # Use only x and y
    ba = a - b
    bc = c - b

    # Calculate cosine using dot product formula
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))

    # Clamp values to prevent floating-point issues (e.g., >1.0 or <-1.0)
    cosine_angle = np.clip(cosine_angle, -1.0, 1.0)

    # Convert from radians to degrees
    angle = np.degrees(np.arccos(cosine_angle))
    return angle

# function that utilizes exponential moving average for smoothing
def smooth_angle(name, new_angle):
    prev = smoothed_angles[name]
    if prev is None:
        smoothed_angles[name] = new_angle
    else:
        smoothed_angles[name] = ALPHA * new_angle + (1 - ALPHA) * prev
    return smoothed_angles[name]

video = "Basketball_vids/basketball_shot.mp4" 
cap = cv2.VideoCapture(video)
fps = cap.get(cv2.CAP_PROP_FPS)
delay = int(1000 / fps)

# Track video completion
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
current_frame = 0
video_completed = False
final_results_printed = False

# Store latest results for each position
latest_position_results = {
    "position_1": None,
    "position_2": None,
    "position_3": None
}
# Store best status achieved for each joint in each position (best status across all frames)
best_position_details = {
    "position_1": {},
    "position_2": {},
    "position_3": {}
}
# Track if all key joints were ever acceptable/passable simultaneously for each position
best_simultaneous_overall = {
    "position_1": None,
    "position_2": None,
    "position_3": None
}
latest_final_status = None

# Status hierarchy: acceptable > passable > unacceptable > no data
def get_better_status(status1, status2):
    """Returns the better status between two statuses"""
    if status1 is None:
        return status2
    if status2 is None:
        return status1
    hierarchy = {"acceptable": 3, "passable": 2, "unacceptable": 1, "no data": 0}
    return status1 if hierarchy.get(status1, 0) >= hierarchy.get(status2, 0) else status2

def calculate_overall_status_from_joints(joint_feedback):
    """Calculate overall status from joint feedback using the same logic as formAnalysis"""
    # Key joints determine overall evaluation (same as formAnalysis.py)
    key_joints = [
        "left_knee", "right_knee",
        "left_elbow", "right_elbow",
        "left_shoulder", "right_shoulder"
    ]
    key_statuses = [joint_feedback[j]["status"] for j in key_joints if j in joint_feedback]
    
    if not key_statuses:
        return "unacceptable"
    
    if any(status == "unacceptable" for status in key_statuses):
        return "unacceptable"
    elif all(status == "acceptable" for status in key_statuses):
        return "acceptable"
    elif all(status in ["acceptable", "passable"] for status in key_statuses):
        return "passable"
    else:
        return "unacceptable"

def get_better_overall_status(status1, status2):
    """Returns the better overall status between two statuses"""
    if status1 is None:
        return status2
    if status2 is None:
        return status1
    hierarchy = {"acceptable": 3, "passable": 2, "unacceptable": 1}
    return status1 if hierarchy.get(status1, 0) >= hierarchy.get(status2, 0) else status2

with mp_pose.Pose(min_detection_confidence=0.5,
                  min_tracking_confidence=0.5) as pose:
    
    form_analyzer = BasketballFormAnalysis()
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            # Video finished - print final results if not already printed
            if not final_results_printed and video_completed:
                print("\n" + "="*50)
                print("VIDEO ANALYSIS COMPLETE")
                print("="*50)
                # Print detailed results for each position
                for pos in ["position_1", "position_2", "position_3"]:
                    if best_position_details[pos]:
                        print(f"\n{pos.upper()} Evaluation:")
                        # Show individual joint statuses (best ever achieved)
                        for joint, info in best_position_details[pos].items():
                            status = info["status"]
                            color_name = (
                                "Green" if status == "acceptable" else
                                "Yellow" if status == "passable" else
                                "Red" if status == "unacceptable" else "Gray"
                            )
                            print(f"  {joint}: {status} ({color_name})")
                        # Show overall status (best simultaneous status - all key joints at same time)
                        if best_simultaneous_overall[pos]:
                            print(f"→ Overall (simultaneous): {best_simultaneous_overall[pos].upper()}")
                        else:
                            print(f"→ Overall (simultaneous): UNACCEPTABLE")
                
                # Calculate final form status based on best simultaneous statuses achieved
                p1_status = best_simultaneous_overall["position_1"] or "unacceptable"
                p2_status = best_simultaneous_overall["position_2"] or "unacceptable"
                p3_status = best_simultaneous_overall["position_3"] or "unacceptable"
                
                # If position 2 is unacceptable, shot is unacceptable
                if p2_status == "unacceptable":
                    final_form_status = "unacceptable"
                # If all 3 positions are acceptable → acceptable
                elif all(x == "acceptable" for x in [p1_status, p2_status, p3_status]):
                    final_form_status = "acceptable"
                # If 2 out of 3 positions are acceptable → passable
                elif [p1_status, p2_status, p3_status].count("acceptable") == 2:
                    final_form_status = "passable"
                # If at least 2 positions are passable or better → passable
                elif [p1_status, p2_status, p3_status].count("unacceptable") <= 1:
                    final_form_status = "passable"
                # Otherwise → unacceptable
                else:
                    final_form_status = "unacceptable"
                
                latest_final_status = final_form_status
                print(f"\nFINAL FORM OUTPUT: {final_form_status.upper()}")
                print("="*50 + "\n")
                final_results_printed = True
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # Loop video
            current_frame = 0
            video_completed = False
            final_results_printed = False
            # Reset position results for next cycle
            latest_position_results = {
                "position_1": None,
                "position_2": None,
                "position_3": None
            }
            best_position_details = {
                "position_1": {},
                "position_2": {},
                "position_3": {}
            }
            best_simultaneous_overall = {
                "position_1": None,
                "position_2": None,
                "position_3": None
            }
            latest_final_status = None
            continue
        
        current_frame += 1
        # Check if we've reached the end of the video
        if current_frame >= total_frames and not video_completed:
            video_completed = True

        # converts to RGB for MediaPipe processing
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(image)

        # converts back to BGR for OpenCV display
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        if results.pose_landmarks:
            lm = results.pose_landmarks.landmark
            h, w, _ = image.shape

            # function to get normalized coordinates
            def get_coords(idx):
                return [lm[idx].x, lm[idx].y, lm[idx].z]

            # left side joints
            left_shoulder, left_elbow, left_wrist = get_coords(11), get_coords(13), get_coords(15)
            left_hip, left_knee, left_ankle = get_coords(23), get_coords(25), get_coords(27)
            left_heel, left_foot = get_coords(29), get_coords(31)

            # right side joints
            right_shoulder, right_elbow, right_wrist = get_coords(12), get_coords(14), get_coords(16)
            right_hip, right_knee, right_ankle = get_coords(24), get_coords(26), get_coords(28)
            right_heel, right_foot = get_coords(30), get_coords(32)

            # CALCULATE 2D ANGLES 
            angles = {
                'left_elbow': calculate_angle_2d(left_shoulder, left_elbow, left_wrist),
                'left_wrist': calculate_angle_2d(left_elbow, left_wrist, left_foot),
                'left_knee': calculate_angle_2d(left_hip, left_knee, left_ankle),
                'left_ankle': calculate_angle_2d(left_knee, left_ankle, left_heel),
                'left_shoulder': calculate_angle_2d(left_elbow, left_shoulder, left_hip),

                'right_elbow': calculate_angle_2d(right_shoulder, right_elbow, right_wrist),
                'right_wrist': calculate_angle_2d(right_elbow, right_wrist, right_foot),
                'right_knee': calculate_angle_2d(right_hip, right_knee, right_ankle),
                'right_ankle': calculate_angle_2d(right_knee, right_ankle, right_heel),
                'right_shoulder': calculate_angle_2d(right_elbow, right_shoulder, right_hip),
            }

            # apply smoothing
            for key in angles:
                angles[key] = smooth_angle(key, angles[key])

            # perform form analysis
            # Evaluate all positions continuously
            positions = ["position_1", "position_2", "position_3"]
            position_results = {pos: {"acceptable": False, "passable": False, "unacceptable": False} for pos in positions}
            final_form_status = None  # "acceptable" / "passable" / "unacceptable"

            # Perform sequential form analysis with custom logic
            for i, pos in enumerate(positions):
                # Disable printing during playback - we'll print everything at the end
                overall_status, joint_feedback = form_analyzer.evaluate_position(smoothed_angles, pos, print_output=False)
                position_results[pos][overall_status] = True
                
                # Store latest result for this position
                latest_position_results[pos] = overall_status
                
                # Track best status achieved for each joint (if ever acceptable, keep it as acceptable)
                for joint, info in joint_feedback.items():
                    current_status = info["status"]
                    if joint not in best_position_details[pos]:
                        # First time seeing this joint - initialize with current status
                        best_position_details[pos][joint] = {
                            "status": current_status,
                            "color": info["color"]
                        }
                    else:
                        # Update to better status if current is better
                        best_status = get_better_status(best_position_details[pos][joint]["status"], current_status)
                        if best_status != best_position_details[pos][joint]["status"]:
                            # Update status and color
                            color_map = {
                                "acceptable": (0, 255, 0),    # green
                                "passable": (0, 255, 255),     # yellow
                                "unacceptable": (0, 0, 255),   # red
                                "no data": (128, 128, 128)     # gray
                            }
                            best_position_details[pos][joint] = {
                                "status": best_status,
                                "color": color_map.get(best_status, (128, 128, 128))
                            }
                
                # Track best simultaneous overall status (all key joints at the same time)
                # This checks if all key joints are acceptable/passable at THIS specific frame
                current_simultaneous_status = calculate_overall_status_from_joints(joint_feedback)
                best_simultaneous_overall[pos] = get_better_overall_status(
                    best_simultaneous_overall[pos], 
                    current_simultaneous_status
                )
                
                # Continue evaluating all positions - final status will be calculated at end
                continue


            # function to draw text
            def draw_text(label, coords, value, color):
                pos = tuple((np.array(coords[:2]) * [w, h]).astype(int))
                cv2.putText(image, f'{label}: {int(value)}°', pos,
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2, cv2.LINE_AA)

            # Draw left-side angles
            draw_text('L-Elbow', left_elbow, angles['left_elbow'], (0, 255, 0))
            draw_text('L-Wrist', left_wrist, angles['left_wrist'], (255, 255, 0))
            draw_text('L-Knee', left_knee, angles['left_knee'], (255, 0, 0))
            draw_text('L-Ankle', left_ankle, angles['left_ankle'], (0, 0, 255))
            draw_text('L-Shoulder', left_shoulder, angles['left_shoulder'], (255, 0, 255))

            # Draw right-side angles
            draw_text('R-Elbow', right_elbow, angles['right_elbow'], (0, 255, 0))
            draw_text('R-Wrist', right_wrist, angles['right_wrist'], (255, 255, 0))
            draw_text('R-Knee', right_knee, angles['right_knee'], (255, 0, 0))
            draw_text('R-Ankle', right_ankle, angles['right_ankle'], (0, 0, 255))
            draw_text('R-Shoulder', right_shoulder, angles['right_shoulder'], (255, 0, 255))

            
            # After obtaining joint_feedback from form_analyzer (for drawing skeleton colors)
            overall_status, joint_feedback = form_analyzer.evaluate_position(smoothed_angles, "position_1", print_output=False)

        # draw the pose skeleton (including colored segments)
            if results.pose_landmarks:
                lm = results.pose_landmarks.landmark
                h, w, _ = image.shape

        # Helper function to convert Mediapipe coords to pixels
                def to_pixel_coords(landmark):
                    return int(landmark.x * w), int(landmark.y * h)

        # Define pairs of landmarks (based on Mediapipe’s POSE_CONNECTIONS)
                connections = [
                    (11, 13), (13, 15),   # Left arm
                    (12, 14), (14, 16),   # Right arm
                    (23, 25), (25, 27),   # Left leg
                    (24, 26), (26, 28),   # Right leg
                    (11, 12), (23, 24),   # Shoulders & hips
                    (11, 23), (12, 24)    # Torso
                ]

                # Mapping landmark pairs to logical joint names for color lookup
                connection_to_joint = {
                    (11, 13): "left_shoulder",
                    (13, 15): "left_elbow",
                    (12, 14): "right_shoulder",
                    (14, 16): "right_elbow",
                    (23, 25): "left_hip",
                    (25, 27): "left_knee",
                    (24, 26): "right_hip",
                    (26, 28): "right_knee"
                }

                for start_idx, end_idx in connections:
                    start_point = to_pixel_coords(lm[start_idx])
                    end_point = to_pixel_coords(lm[end_idx])
                # Default to gray if joint not in feedback
                    joint_name = connection_to_joint.get((start_idx, end_idx))
                    if joint_name and joint_name in joint_feedback:
                        color = joint_feedback[joint_name]["color"]
                    else:
                        color = (128, 128, 128)

                    # Draw colored skeleton segment
                    cv2.line(image, start_point, end_point, color, 4)

                # Optionally draw circles for each landmark
                for idx, landmark in enumerate(lm):
                    cx, cy = to_pixel_coords(landmark)
                    cv2.circle(image, (cx, cy), 4, (255, 255, 255), -1)

        

        cv2.imshow('Pose Angles (2D, Smoothed)', image)

        # Quit on 'q' key
        if cv2.waitKey(delay) & 0xFF == ord('q'):
            break


cap.release()
cv2.destroyAllWindows()

