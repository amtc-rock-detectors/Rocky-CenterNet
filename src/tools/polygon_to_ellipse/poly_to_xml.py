
import numpy as np
import cv2
import xml.etree.ElementTree as ET
from xml.dom import minidom


def poly_to_ellipse(points: list, tolerance=1e-2, use_convex_hull=False):

    def ellipse_format_change(A, c):

        """
        Inputs:

        A: Ellipse Center form Matrix
        c: Ellipse Center form center coordinates

        Outputs:

        list with:

        - center x coordinate
        - center y coordinate
        - major axis (semi-major * 2)
        - minor axis (semi-minor * 2)
        - rotation angle (degrees)
        """

        U, D, V = np.linalg.svd(A)

        b = 2 / np.sqrt(D[0])
        a = 2 / np.sqrt(D[1])

        if np.allclose(V[0, 0], V[1, 1]):
            cos = V[0, 0]
            sin = -V[0, 1]
            theta = np.arccos(cos) * 180. / np.pi 
            if sin < 0:
                theta *= -1
        else:
            cos = V[1, 0]
            sin = V[1, 1]
            theta = np.arccos(cos) * 180. / np.pi 
            if sin < 0:
                theta *= -1

        return [c[0], c[1], a, b, theta]

    def reformat_ellipse(vals):

        xc, yc, a_1, a_2, angle = vals

        if a_1 >= a_2:

            return [xc, yc, a_1, a_2, (angle % 180.)]

        else:

            return [xc, yc, a_2, a_1, (angle + 90.) % 180.]

    """
    Algorithm for  Minimum Volume Enclosing Ellipsoid, or in this 2D case, 
    the minimum area. This optimization problem is convex and can be solved 
    efficiently.

    Adapted from Jacob's algorithm in Java from Stackoverflow:
    https://stackoverflow.com/questions/1768197/bounding-ellipse

    Input: A list of lists of length 2, storing 2D points
           and tolerance = tolerance for error.
    Output: The equation of the ellipse in the matrix form, 
            i.e. a 2x2 matrix A and a 2x1 vector C representing 
            the center of the ellipse.
    """

    # Dimension of the points
    d = 2 
    if use_convex_hull:
        points = cv2.convexHull(np.array(points, np.float32), False).reshape(-1, 2).tolist()

    # Points array
    P = np.array(points).reshape(-1, 2).T

    # Number of points
    N = P.shape[1]

    c_bias = P.mean(axis=1)
    P -= c_bias[:, np.newaxis]

    # Add a row of 1s to the 2xN matrix points_arr - so Q is 3xN now.
    Q = np.concatenate([P, np.ones((1,N))], axis=0)

    # Initialize 
    count = 1
    err = 1
    
    # u is an Nx1 vector where each element is 1/N
    u = np.ones(N) / N

    try:

        # Khachiyan Algorithm
        while err > tolerance:
        
            # Matrix multiplication: 
            # diag(u) : if u is a vector, places the elements of u 
            # in the diagonal of an NxN matrix of zeros
            X = Q @ np.diag(u) @ Q.T

            # inv(X) returns the matrix inverse of X
            # diag(M) when M is a matrix returns the diagonal vector of M
            M = np.diag(Q.T @ np.linalg.inv(X) @ Q)

            # Find the value and location of the maximum element in the vector M
            maximum = np.max(M)
            j = np.argmax(M)

            # Calculate the step size for the ascent
            step_size = (maximum - d - 1) / ((d + 1) * (maximum - 1))

            # Calculate the new_u:
            # Take the vector u, and multiply all the elements in it by (1-step_size)
            new_u = (1 - step_size) * u

            # Increment the jth element of new_u by step_size
            new_u[j] = new_u[j] + step_size

            # Store the error by taking finding the square root of the SSD 
            # between new_u and u
            # The SSD or sum-of-square-differences, takes two vectors 
            # of the same size, creates a new vector by finding the 
            # difference between corresponding elements, squaring 
            # each difference and adding them all together. 

            # So if the vectors were: a = [1 2 3] and b = [5 4 6], then:
            # SSD = (1-5)^2 + (2-4)^2 + (3-6)^2;
            # And the norm(a-b) = sqrt(SSD);
            err = np.linalg.norm(new_u - u)

            # Increment count and replace u
            count += 1
            u = new_u

        # Put the elements of the vector u into the diagonal of a matrix
        # U with the rest of the elements as 0
        U = np.diag(u)

        # Compute the A-matrix
        A = (1 / d) * np.linalg.inv(P @ U @ P.T - (P @ u) @ (P @ u).T )

        # And the center,
        c = P @ u + c_bias

        return reformat_ellipse(ellipse_format_change(A, c))

    except:

        return None


def polygon_to_rot_rect(points):

    def angle_to_x_axis(segment):

        # return angle
        point_a = segment[0]
        point_b = segment[1]

        vector = np.array(point_b) - np.array(point_a)

        return -np.arctan(np.divide(vector[1], vector[0]))


    def rotate_point(point, angle):

        # return points
        rot_x = point[0] * np.cos(angle) - point[1] * np.sin(angle)
        rot_y = point[0] * np.sin(angle) + point[1] * np.cos(angle)

        return [rot_x, rot_y]

    def reformat_ellipse(vals):

        xc, yc, a_1, a_2, angle = vals

        if a_1 >= a_2:

            return [xc, yc, a_1, a_2, (angle % 180.)]

        else:

            return [xc, yc, a_2, a_1, (angle + 90.) % 180.]

    # get convex hull
    points = np.array(points).reshape(-1, 2)
    center = np.array(points).mean(axis=0)
    # points = cv2.convexHull(np.array(points, np.float32), False).reshape(-1, 2)
    points = np.array(points)

    # center points
    points = (points - center).tolist()

    box = None
    box_angle = None
    min_area = np.inf

    N = len(points)

    if N >= 3:

        for i, current in enumerate(points):

            next = points[(i + 1) % N]
            segment = np.array([current, next])
            angle = angle_to_x_axis(segment)

            top = -np.inf
            left = np.inf
            bot = np.inf
            right = -np.inf

            for point in points:

                rot_point = rotate_point(point, angle)

                top = max(top, rot_point[1])
                left = min(left, rot_point[0])
                bot = min(bot, rot_point[1])
                right = max(right, rot_point[0])

            h, w = top - bot, right - left

            if h > 0 and w > 0:

                box_area = h * w

                if box is None or box_area < min_area:

                    box = [left, top, right, bot]
                    box_angle = -angle
                    min_area = box_area

        left, top, right, bot = box

        return reformat_ellipse([center[0], center[1], right - left, top - bot, box_angle * 180. / np.pi])

    else:

        return None


def json_to_xml(dict_, folder_name, valid_classes, dataset_name,
                conversion_method=poly_to_ellipse):
    
    im_name = dict_["imagePath"]
    h, w = dict_["imageHeight"], dict_["imageWidth"]

    root = ET.Element('annotation')

    folder = ET.SubElement(root, 'folder')
    folder.text = folder_name

    filename = ET.SubElement(root, "filename")
    filename.text = os.path.basename(im_name)

    source = ET.SubElement(root, "source")
    database = ET.SubElement(source, "database")
    database.text = dataset_name

    size = ET.SubElement(root, "size")
    width = ET.SubElement(size, "width")
    width.text = str(w)
    height = ET.SubElement(size, "height")
    height.text = str(h)
    depth = ET.SubElement(size, "depth")
    depth.text = str(3)

    segmented = ET.SubElement(root, "segmented")
    segmented.text = str(0)

    for k in range(len(dict_['shapes'])):

        class_name = dict_['shapes'][k]['label']

        if class_name not in valid_classes:
            continue

        vals = dict_['shapes'][k]['points']

        values = conversion_method(vals)

        if values is None:
            continue

        obj = ET.SubElement(root, "object")

        name = ET.SubElement(obj, "name")
        name.text = class_name if "rock" not in class_name else "rock"

        pose = ET.SubElement(obj, "pose")
        pose.text = "Front"

        truncated = ET.SubElement(obj, "truncated")
        truncated.text = str(0)

        difficult = ET.SubElement(obj, "difficult")
        difficult.text = str(0)

        bndbox = ET.SubElement(obj, "bndbox")
        xc = ET.SubElement(bndbox, "xc")
        xc.text = "%.3f" % values[0]
        yc = ET.SubElement(bndbox, "yc")
        yc.text = "%.3f" % values[1]
        a = ET.SubElement(bndbox, "a")
        a.text = "%.3f" % values[2]
        b = ET.SubElement(bndbox, "b")
        b.text = "%.3f" % values[3]
        angle = ET.SubElement(bndbox, "angle")
        # Considering the angle is respect to the first axis value, that is value[k, 2],
        # the angle is given a shift so that its value is that of the major axis wrt x axis.
        angle.text = "%.3f" % values[4]

    annotation = minidom.parseString(ET.tostring(root, 'utf-8')).toprettyxml(indent="   ")
    
    return annotation


if __name__ == "__main__":

    import glob
    import json
    import os

    base_path = "../../../data"
    dataset_name = "rock_hammer_dataset_v2"
    folder_name = "train"

    for file in glob.glob(os.path.join(base_path, dataset_name, folder_name, "*.json")):

        print(f"Getting ellipses from file {file}...")

        with open(file, 'r') as json_file:

            ann_json = json.load(json_file)

        xml_file = json_to_xml(ann_json, folder_name, ['rock', 'hard_rock'], dataset_name)
        xmlpath = os.path.splitext(file)[0] + ".xml"

        with open(xmlpath, 'w') as f:

            f.write(xml_file)
