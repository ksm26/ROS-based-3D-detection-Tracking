import numpy as np


def convert_bbs_type(boxes, input_box_type):
    boxes = np.array(boxes)

    assert input_box_type in ["Kitti", "OpenPCDet", "Waymo"], 'unsupported input box type!'

    if input_box_type in ["OpenPCDet", "Waymo"]:
        return boxes

    if input_box_type == "Kitti":  # (h,w,l,x,y,z,yaw) -> (x,y,z,l,w,h,yaw)

        t_id = boxes.shape[1] // 7
        new_boxes = np.zeros(shape=boxes.shape)
        new_boxes[:, :] = boxes[:, :]
        for i in range(t_id):
            b_id = i * 7
            new_boxes[:, b_id + 0:b_id + 3] = boxes[:, b_id + 3:b_id + 6]
            new_boxes[:, b_id + 3] = boxes[:, b_id + 2]
            new_boxes[:, b_id + 4] = boxes[:, b_id + 1]
            new_boxes[:, b_id + 5] = boxes[:, b_id + 0]
            new_boxes[:, b_id + 6] = (np.pi - boxes[:, b_id + 6]) + np.pi / 2
            new_boxes[:, b_id + 2] += boxes[:, b_id + 0] / 2
        return new_boxes

def get_registration_angle(mat):
    cos_theta = mat[0, 0]
    sin_theta = mat[1, 0]

    if cos_theta < -1:
        cos_theta = -1
    if cos_theta > 1:
        cos_theta = 1

    theta_cos = np.arccos(cos_theta)

    if sin_theta >= 0:
        return theta_cos
    else:
        return 2 * np.pi - theta_cos


def register_bbs(boxes, pose):

    if pose is None:
        return boxes

    ang = get_registration_angle(pose)

    t_id = boxes.shape[1] // 7

    ones = np.ones(shape=(boxes.shape[0], 1))
    for i in range(t_id):
        b_id = i * 7
        box_xyz = boxes[:, b_id:b_id + 3]
        box_xyz1 = np.concatenate([box_xyz, ones], -1)

        box_world = np.matmul(box_xyz1, pose.T)

        boxes[:, b_id:b_id + 3] = box_world[:, 0:3]
        boxes[:, b_id + 6] += ang
    return boxes

def corners3d_to_img_boxes(P2, corners3d):
    """
    :param corners3d: (N, 8, 3) corners in rect coordinate
    :return: boxes: (None, 4) [x1, y1, x2, y2] in rgb coordinate
    :return: boxes_corner: (None, 8) [xi, yi] in rgb coordinate
    """
    sample_num = corners3d.shape[0]
    corners3d_hom = np.concatenate((corners3d, np.ones((sample_num, 8, 1))), axis=2)  # (N, 8, 4)

    img_pts = np.matmul(corners3d_hom, P2.T)  # (N, 8, 3)

    x, y = img_pts[:, :, 0] / img_pts[:, :, 2], img_pts[:, :, 1] / img_pts[:, :, 2]
    x1, y1 = np.min(x, axis=1), np.min(y, axis=1)
    x2, y2 = np.max(x, axis=1), np.max(y, axis=1)

    img_boxes = np.concatenate((x1.reshape(-1, 1), y1.reshape(-1, 1), x2.reshape(-1, 1), y2.reshape(-1, 1)), axis=1)
    boxes_corner = np.concatenate((x.reshape(-1, 8, 1), y.reshape(-1, 8, 1)), axis=2)

    img_boxes[:, 0] = np.clip(img_boxes[:, 0], 0, 1242 - 1)
    img_boxes[:, 1] = np.clip(img_boxes[:, 1], 0, 375 - 1)
    img_boxes[:, 2] = np.clip(img_boxes[:, 2], 0, 1242 - 1)
    img_boxes[:, 3] = np.clip(img_boxes[:, 3], 0, 375 - 1)

    return img_boxes, boxes_corner

def bb3d_2_bb2d(bb3d,P2):

    x,y,z,l,w,h,yaw = bb3d[0],bb3d[1],bb3d[2],bb3d[3],bb3d[4],bb3d[5],bb3d[6]

    pt1 = [l / 2, 0, w / 2, 1]
    pt2 = [l / 2, 0, - w / 2, 1]
    pt3 = [- l / 2, 0, w / 2, 1]
    pt4 = [- l / 2, 0, - w / 2, 1]
    pt5 = [l / 2, - h, w / 2, 1]
    pt6 = [l / 2, - h, - w / 2, 1]
    pt7 = [- l / 2, - h, w / 2, 1]
    pt8 = [- l / 2, - h, - w / 2, 1]
    pts = np.array([[pt1, pt2, pt3, pt4, pt5, pt6, pt7, pt8]])
    transpose = np.array([[np.cos(np.pi - yaw), 0, -np.sin(np.pi - yaw), x],
                          [0, 1, 0, y],
                          [np.sin(np.pi - yaw), 0, np.cos(np.pi - yaw), z],
                          [0, 0, 0, 1]])
    pts = np.matmul(pts, transpose.T)
    box, _ = corners3d_to_img_boxes(P2, pts[:, :, 0:3])

    return box

def bb3d_corners_nuscenes(bb3d) -> np.ndarray:
    """
    Returns the bounding box corners.
    :param wlh_factor: Multiply w, l, h by a factor to scale the box.
    :return: <np.float: 3, 8>. First four corners are the ones facing forward.
        The last four are the ones facing backwards.
    """
    x, y, z, l, w, h, yaw = bb3d[0], bb3d[1], bb3d[2], bb3d[3], bb3d[4], bb3d[5], bb3d[6]
    # w, l, h = self.wlh * wlh_factor

    # 3D bounding box corners. (Convention: x points forward, y to the left, z up.)
    x_corners = l / 2 * np.array([1,  1,  1,  1, -1, -1, -1, -1])
    y_corners = w / 2 * np.array([1, -1, -1,  1,  1, -1, -1,  1])
    z_corners = h / 2 * np.array([1,  1, -1, -1,  1,  1, -1, -1])
    homog_coord =  np.array([1, 1, 1, 1, 1, 1, 1, 1])
    corners = np.vstack((x_corners, y_corners, z_corners,homog_coord ))

    # Rotate
    transpose = np.array([[np.cos(yaw), -np.sin(yaw), 0, x],
                          [np.sin(yaw), np.cos(yaw), 0, y],
                          [0, 0, 1, z],
                          [0, 0, 0, 1]])
    corners = np.matmul(transpose,corners)

    return corners

def corners_transform_nuscenes(corners,pose) -> np.ndarray:
    """
    Returns the bounding box corners in transformed coordinates.
    :param corners: numpy array (3x8) of the 8 coorners of 3D bbox
    :param:transl and rot angle of the new coordinate frame
    :return: <np.float: 3, 8>. First four corners are the ones facing forward.
        The last four are the ones facing backwards.
    """

    # transformed_corners = corners+pose[3,0:3]
    # transformed_corners = np.dot(pose[0:3,0:3], corners)

    transformed_corners = np.dot(pose, corners)

    return transformed_corners

def bb3d_centres_nuscenes(bb3d) -> np.ndarray:
    """
    Returns the bounding box center.
    """

    x_center = np.sum(bb3d[0,:]) / 8
    y_center = np.sum(bb3d[1,:]) / 8
    z_center = np.sum(bb3d[2,:]) / 8
    return [x_center,y_center,z_center]
