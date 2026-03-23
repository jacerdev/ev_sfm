import logging

import numpy as np
import cv2
import matplotlib.pyplot as plt


def draw_points(img, pts, thickness=4, color=(0, 170, 0)):
    canvas = img.copy()
    for x, y in pts:
        if x < 0 or x >= canvas.shape[1] or y < 0 or y >= canvas.shape[1]:
            continue
        cv2.circle(canvas, (int(x), int(y)), thickness, color, -1)
    return canvas

def draw_matches(img1, img2, pts1, pts2, match_stride=0, thickness=4):
    assert len(pts1) == len(pts2), f"pts1 and pts2 must have the same length, got {len(pts1)} and {len(pts2)} respectively"
    if match_stride == 0: match_stride = max(len(pts1)//15, 1) # draw at least 15 matches
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]

    canvas = np.zeros((max(h1, h2), w1 + w2, 3), dtype=np.uint8)
    canvas[:h1, :w1] = img1
    canvas[:h2, w1:w1 + w2] = img2

    for i, (x1, x2) in enumerate(zip(pts1, pts2)):
        if x1[0] < 0 or x1[1] < 0 or x2[0] < 0 or x2[1] < 0 or x1[0] >= w1 or x1[1] >= h1 or x2[0] >= w2 or x2[1] >= h2:
            continue
        u1, v1 = int(x1[0]), int(x1[1])
        u2, v2 = int(x2[0]) + w1, int(x2[1])
        cv2.circle(canvas, (u1, v1), thickness, (0, 255, 0), -1)
        cv2.circle(canvas, (u2, v2), thickness, (0, 255, 0), -1)
        if i % match_stride == 0:
            cv2.line(canvas, (u1, v1), (u2, v2), (200, 200, 0), max(thickness//2,1))

    return canvas

def make_border(img, border_size=1):
    img = img[border_size:-border_size, border_size:-border_size]
    img = cv2.copyMakeBorder(img,
                             top=border_size, bottom=border_size, left=border_size, right=border_size,
                             borderType=cv2.BORDER_CONSTANT, value=[0, 0, 0])
    return img

def plot_images(imgs, ncols=None, title=None, size=(5,5)):
    if isinstance(imgs, np.ndarray) and imgs.ndim == 3: imgs = [imgs]
    num_imgs = len(imgs)
    if ncols is None: ncols = int(np.sqrt(num_imgs) + .99)
    nrows = (num_imgs + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * size[0], nrows * size[1]), squeeze=False)
    axes = axes.flatten()

    for i in range(num_imgs): axes[i].imshow(imgs[i]); axes[i].axis('off')
    for j in range(i + 1, len(axes)): axes[j].axis('off') # hide unused subplots

    if title: plt.title(title, fontsize=fig.get_size_inches()[0] * 2)

    plt.tight_layout()
    plt.show()

def plot_sfm(frames_positions, frames_directions, points_xyz, frames_rgb=None, points_rgb=None, camera_size=1, point_size=1, color_range=(5, 95)):
    """
    Visualize 3D points and camera positions and orientations.
    """
    if frames_rgb is None: frames_rgb = np.zeros((len(frames_positions), 3))
    ax = plt.figure(figsize=(10,8)).add_subplot(111, projection='3d')

    # Plot 3D points
    if len(points_xyz) != 0:
        points_rgb = points_rgb.astype(np.float32) / 255.0 if points_rgb is not None else get_depth_colors(points_xyz, range_percentile=color_range)
        ax.scatter(points_xyz[:,0], points_xyz[:,1], points_xyz[:,2], c=points_rgb, s=point_size, alpha=0.6)

    # Plot camera positions
    #ax.scatter(camera_positions[:,0], camera_positions[:,1], camera_positions[:,2],
    #           c='r', marker='^', s=camera_size*1000)

    # Plot camera directions
    if len(frames_directions) != 0:
        frames_rgb = frames_rgb.astype(np.float32) / 255.0
        for i, (C, direction) in enumerate(zip(frames_positions, frames_directions)):
            ax.quiver(C[0], C[1], C[2],
                      direction[0], direction[1], direction[2],
                      length=camera_size, color=frames_rgb[i])
            #ax.text(C[0], C[1], C[2], str(i), color=frames_rgb[i], fontsize=camera_size*4)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    #ax.set_title('3D Points and Camera Positions')


    all_xyz = np.vstack([arr for arr in [points_xyz, frames_positions] if len(arr) > 0])
    max_range = np.ptp(all_xyz, axis=0).max() / 2
    mid = all_xyz.mean(axis=0)
    ax.set_xlim(mid[0] - max_range, mid[0] + max_range)
    ax.set_ylim(mid[1] - max_range, mid[1] + max_range)
    ax.set_zlim(mid[2] - max_range, mid[2] + max_range)

    # Force the plot to be cubic
    ax.set_box_aspect([1, 1, 1])

    """
    C = frame_positions[0]
    d = frame_directions[0]
    d = d / np.linalg.norm(d)
    azim = np.degrees(np.arctan2(d[1], d[0]))
    elev = np.degrees(np.arcsin(d[2]))    
    ax.view_init(elev=elev, azim=azim)
    """
    plt.show()


def get_depth_colors(points_3d, cmap_name='jet_r', range_percentile=(5, 95)):
    """
    Generates an array of RGB colors based on the depth (Z-coordinate) of 3D points
    Returns: (N, 3) array of RGB values in range [0, 1]
    """
    depths = points_3d[:, 2]
    min_d = np.percentile(depths, range_percentile[0])
    max_d = np.percentile(depths, range_percentile[1])

    # Normalize depths to [0, 1]
    norm_depths = np.clip(depths, min_d, max_d)
    norm_depths = (norm_depths - min_d) / (max_d - min_d)

    # apply colormap
    cmap = plt.get_cmap(cmap_name)
    colors = cmap(norm_depths)[:, :3]

    return colors
