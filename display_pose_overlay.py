import cv2
import mediapipe as mp
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import argparse
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean
from txt_parse import compile_text_file_to_array, convert_compiled_data_to_numpy

def process_frame(frame, pose):
    """Process a video frame and return bounding-box-normalized pose landmark coordinates and results."""
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(rgb_frame)
    normalized_coords = None
    if results.pose_landmarks:
        landmarks = results.pose_landmarks.landmark
        coords = np.array([[lm.x, lm.y, lm.z] for lm in landmarks])
        # Bounding box normalization
        min_vals = coords.min(axis=0)
        max_vals = coords.max(axis=0)
        center = (min_vals + max_vals) / 2
        size = (max_vals - min_vals).max()  # Largest dimension for uniform scaling
        if size != 0:
            normalized_coords = (coords - center) / size
        else:
            normalized_coords = coords - center
    return normalized_coords, results

def draw_bounding_box(ax):
    r = [-0.5, 0.5]
    edges = [
        ([r[0], r[0], r[0]], [r[1], r[0], r[0]]),
        ([r[1], r[0], r[0]], [r[1], r[1], r[0]]),
        ([r[1], r[1], r[0]], [r[0], r[1], r[0]]),
        ([r[0], r[1], r[0]], [r[0], r[0], r[0]]),
        ([r[0], r[0], r[1]], [r[1], r[0], r[1]]),
        ([r[1], r[0], r[1]], [r[1], r[1], r[1]]),
        ([r[1], r[1], r[1]], [r[0], r[1], r[1]]),
        ([r[0], r[1], r[1]], [r[0], r[0], r[1]]),
        ([r[0], r[0], r[0]], [r[0], r[0], r[1]]),
        ([r[1], r[0], r[0]], [r[1], r[0], r[1]]),
        ([r[1], r[1], r[0]], [r[1], r[1], r[1]]),
        ([r[0], r[1], r[0]], [r[0], r[1], r[1]]),
    ]
    for s, e in edges:
        ax.plot([s[0], e[0]], [s[1], e[1]], [s[2], e[2]], c='g', linewidth=1)

def plot_landmarks(ax, normalized_coords, connections, title):
    ax.clear()
    draw_bounding_box(ax)
    if normalized_coords is not None:
        xs = normalized_coords[:, 0]
        ys = -normalized_coords[:, 1]
        zs = -normalized_coords[:, 2]
        ax.scatter(xs, ys, zs, c='b')
        for connection in connections:
            start_idx, end_idx = connection
            x_vals = [normalized_coords[start_idx, 0], normalized_coords[end_idx, 0]]
            y_vals = [-normalized_coords[start_idx, 1], -normalized_coords[end_idx, 1]]
            z_vals = [-normalized_coords[start_idx, 2], -normalized_coords[end_idx, 2]]
            ax.plot(x_vals, y_vals, z_vals, c='r')
    ax.set_xlim(-0.5, 0.5)
    ax.set_ylim(-0.5, 0.5)
    ax.set_zlim(-0.5, 0.5)
    ax.set_xlabel('X (bbox normalized)')
    ax.set_ylabel('Y (bbox normalized, flipped)')
    ax.set_zlabel('Z (bbox normalized, flipped)')
    ax.set_title(title)

def cosine_similarity_pose(pose1, pose2):
    vec1 = pose1.flatten()
    vec2 = pose2.flatten()
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    if norm1 == 0 or norm2 == 0:
        return 0.0
    vec1 = vec1 / norm1
    vec2 = vec2 / norm2
    return np.dot(vec1, vec2)

def main(video1_path, video2_path, min_detection_conf=0.5, min_tracking_conf=0.5):
    mp_pose = mp.solutions.pose
    mp_drawing = mp.solutions.drawing_utils
    pose1 = mp_pose.Pose(
        static_image_mode=False,
        min_detection_confidence=min_detection_conf,
        min_tracking_confidence=min_tracking_conf
    )
    pose2 = mp_pose.Pose(
        static_image_mode=False,
        min_detection_confidence=min_detection_conf,
        min_tracking_confidence=min_tracking_conf
    )
    cap1 = cv2.VideoCapture(video1_path)
    cap2 = cv2.VideoCapture(video2_path)
    if not cap1.isOpened() or not cap2.isOpened():
        print(f"Error: Could not open video files {video1_path} or {video2_path}")
        return
    # Get FPS and calculate frame interval for 4 FPS
    fps1 = cap1.get(cv2.CAP_PROP_FPS)
    fps2 = cap2.get(cv2.CAP_PROP_FPS)
    target_fps = 4
    interval1 = int(round(fps1 / target_fps)) if fps1 > 0 else 1
    interval2 = int(round(fps2 / target_fps)) if fps2 > 0 else 1
    plt.ion()
    fig = plt.figure(figsize=(20, 5))
    ax1 = fig.add_subplot(1, 4, 1, projection='3d')
    ax2 = fig.add_subplot(1, 4, 2, projection='3d')
    ax3 = fig.add_subplot(1, 4, 3)
    ax4 = fig.add_subplot(1, 4, 4)  # New subplot for DTW path
    similarity_scores = []
    frame_indices = []
    pose_seq1 = []
    pose_seq2 = []
    frame_idx = 0
    frame_count1 = 0
    frame_count2 = 0
    while True:
        ret1, frame1 = cap1.read()
        ret2, frame2 = cap2.read()
        if not ret1 or not ret2:
            break
        # Skip frames to achieve ~4 FPS
        if (frame_count1 % interval1 != 0) or (frame_count2 % interval2 != 0):
            frame_count1 += 1
            frame_count2 += 1
            continue
        norm_coords1, results1 = process_frame(frame1, pose1)
        norm_coords2, results2 = process_frame(frame2, pose2)
        if results1.pose_landmarks:
            mp_drawing.draw_landmarks(
                frame1,
                results1.pose_landmarks,
                mp_pose.POSE_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(0,255,0), thickness=2, circle_radius=2),
                mp_drawing.DrawingSpec(color=(0,0,255), thickness=2, circle_radius=2)
            )
        if results2.pose_landmarks:
            mp_drawing.draw_landmarks(
                frame2,
                results2.pose_landmarks,
                mp_pose.POSE_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(0,255,0), thickness=2, circle_radius=2),
                mp_drawing.DrawingSpec(color=(0,0,255), thickness=2, circle_radius=2)
            )
        plot_landmarks(ax1, norm_coords1, mp_pose.POSE_CONNECTIONS, 'Pose 1 (3D)')
        plot_landmarks(ax2, norm_coords2, mp_pose.POSE_CONNECTIONS, 'Pose 2 (3D)')
        # Compute and plot similarity
        # Remove cosine similarity calculation for all frames
        # if norm_coords1 is not None and norm_coords2 is not None:
        #     sim = cosine_similarity_pose(norm_coords1, norm_coords2)
        #     similarity_scores.append(sim)
        #     frame_indices.append(frame_idx)
        if norm_coords1 is not None and norm_coords2 is not None:
            pose_seq1.append(norm_coords1.flatten())
            pose_seq2.append(norm_coords2.flatten())
        ax3.clear()
        # Remove plotting of similarity for all frames
        # ax3.plot(frame_indices, similarity_scores, c='purple')
        ax3.set_ylim(-1, 1)
        ax3.set_xlabel('Frame')
        ax3.set_ylabel('Cosine Similarity')
        ax3.set_title('Pose Similarity Over Time')
        plt.tight_layout()
        plt.draw()
        plt.pause(0.001)
        # Show both video frames
        cv2.imshow('Video 1 Pose Overlay', frame1)
        cv2.imshow('Video 2 Pose Overlay', frame2)
        frame_idx += 1
        frame_count1 += 1
        frame_count2 += 1
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap1.release()
    cap2.release()
    cv2.destroyAllWindows()
    # DTW calculation and visualization
    if len(pose_seq1) > 0 and len(pose_seq2) > 0:
        distance, path = fastdtw(pose_seq1, pose_seq2, dist=euclidean)
        print(f"DTW distance between videos: {distance}")
        # Compute cosine similarity only for DTW-selected frame pairs
        dtw_similarities = []
        dtw_indices = []
        for i, j in path:
            sim = cosine_similarity_pose(
                np.array(pose_seq1[i]).reshape(-1, 3),
                np.array(pose_seq2[j]).reshape(-1, 3)
            )
            dtw_similarities.append(sim)
            dtw_indices.append(i)
        # Plot similarity graph for DTW-selected pairs
        ax3.clear()
        ax3.plot(dtw_indices, dtw_similarities, c='purple', label='Cosine Similarity (DTW pairs)')
        ax3.set_ylim(-1, 1)
        ax3.set_xlabel('Video 1 Frame Index (DTW)')
        ax3.set_ylabel('Cosine Similarity')
        ax3.set_title('Pose Similarity (DTW-selected pairs)')
        ax3.legend()
        # Plot DTW path as connections between frame indices
        ax4.clear()
        dtw_i = [p[0] for p in path]
        dtw_j = [p[1] for p in path]
        ax4.plot(dtw_i, dtw_j, c='blue', linewidth=1)
        ax4.scatter(dtw_i, dtw_j, c='red', marker='o', label='DTW Connections')
        ax4.set_xlabel('Video 1 Frame Index')
        ax4.set_ylabel('Video 2 Frame Index')
        ax4.set_title('DTW Frame Alignment Path')
        ax4.legend()
        plt.tight_layout()
        plt.show(block=True)
    else:
        plt.ioff()
        plt.show()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Compare two videos with pose overlay and similarity graph.')
    parser.add_argument('--video1', type=str, required=True, help='Path to the first video file')
    parser.add_argument('--video2', type=str, required=True, help='Path to the second video file')
    args = parser.parse_args()
    main(args.video1, args.video2)