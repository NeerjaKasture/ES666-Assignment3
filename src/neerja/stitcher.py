import cv2
import numpy as np
import glob
import os

class PanaromaStitcher():
    def __init__(self):
        pass

    def homography_matrix(self, source_points, destination_points):
        matrix_A = []
        for i in range(len(source_points)):
            x1, y1 = source_points[i][0], source_points[i][1]
            x2, y2 = destination_points[i][0], destination_points[i][1]
            
            entry = [[0, 0, 0, -1 * x1, -1 * y1, -1, y2 * x1, y2 * y1, y2],
                    [x1, y1, 1, 0, 0, 0, -x2 * x1, -x2 * y1, -x2]]
            
            matrix_A.extend(entry)

        matrix_A = np.array(matrix_A)
        U, D, V = np.linalg.svd(matrix_A)
        homography = V[-1].reshape((3, 3))

        return homography 


    def RANSAC(self, source_points, destination_points, iterations=1000):
        threshold = 5.0
        optimal_H, best_inliers = None, []
        max_inliers_count = 0

        for _ in range(iterations):
            random_indices = np.random.choice(source_points.shape[0], 4, replace=False)
            src_subset, dest_subset = source_points[random_indices], destination_points[random_indices]
            homography = self.homography_matrix(src_subset, dest_subset)
            source_homogeneous = np.hstack([source_points, np.ones((source_points.shape[0], 1))])
            projected_points = np.dot(homography, source_homogeneous.T).T
            projected_points = projected_points[:, :2] / projected_points[:, 2].reshape(-1, 1)
            error = np.linalg.norm(destination_points - projected_points, axis=1)
            inliers = np.where(error < threshold)[0]

            if len(inliers) > max_inliers_count:
                max_inliers_count, optimal_H, best_inliers = len(inliers), homography, inliers

        if len(best_inliers) > 4:
            optimal_H = self.homography_matrix(source_points[best_inliers], destination_points[best_inliers])

        return optimal_H, best_inliers

    def cylindrical_projection(self, img, focal_length):
        h, w = img.shape[:2]
        cylindrical_img = np.zeros_like(img)
        center_x, center_y = w // 2, h // 2

        for x in range(w):
            for y in range(h):
                theta = (x - center_x) / focal_length
                h_ = (y - center_y) / focal_length
                X, Y, Z = np.sin(theta), h_, np.cos(theta)
                new_x, new_y = int(focal_length * X / Z + center_x), int(focal_length * Y / Z + center_y)

                if 0 <= new_x < w and 0 <= new_y < h:
                    cylindrical_img[y, x] = img[new_y, new_x]

        return cylindrical_img

    def compute_homography(self, img1, img2):
        detector = cv2.SIFT_create()
        kp1, desc1 = detector.detectAndCompute(img1, None)
        kp2, desc2 = detector.detectAndCompute(img2, None)
        
        index_params = dict(algorithm=1, trees=5)
        search_params = dict(checks=50)
        matcher = cv2.FlannBasedMatcher(index_params, search_params)
        
        match_pairs = matcher.knnMatch(desc1, desc2, k=2)
        good_matches = [m for m, n in match_pairs if m.distance < 0.75 * n.distance]

        if len(good_matches) < 4:
            print("Insufficient matches.")
            return None

        src_points = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 2)
        dst_points = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 2)

        H, inliers = self.RANSAC(src_points, dst_points)
        return H

    def crop_black_borders(self, img):
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, binary_img = cv2.threshold(gray_img, 1, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(binary_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            x, y, w, h = cv2.boundingRect(contours[0])
            return img[y:y + h, x:x + w]
        return img

    def calculate_output_size(self, homographies, base_shape):
        min_x = min_y = float('inf')
        max_x = max_y = float('-inf')

        for H in homographies:
            corners = np.array([[0, 0, 1], [base_shape[1], 0, 1],
                                [base_shape[1], base_shape[0], 1], [0, base_shape[0], 1]])
            transformed = H @ corners.T
            transformed /= transformed[2, :]
            min_x, max_x = min(min_x, transformed[0, :].min()), max(max_x, transformed[0, :].max())
            min_y, max_y = min(min_y, transformed[1, :].min()), max(max_y, transformed[1, :].max())

        width, height = int(max_x - min_x), int(max_y - min_y)
        return width, height

    def make_panaroma_for_images_in(self, path):
        img_files = sorted(glob.glob(path + os.sep + '*'))
        imgs = [cv2.imread(img) for img in img_files]

        center_index = len(imgs) // 2
        focal_length = imgs[center_index].shape[1] / (2 * np.pi) * 8

        imgs = [self.cylindrical_projection(img, focal_length) for img in imgs]
        base_img = imgs[center_index]

        homography_matrix_list = []
        homography_matrix_list.append(np.eye(3)) 

        for i in range(center_index - 1, -1, -1):
            H = self.compute_homography(imgs[i], imgs[i + 1])
            if H is not None:
                homography_matrix_list.insert(0, H @ homography_matrix_list[0])

        for i in range(center_index + 1, len(imgs)):
            H = self.compute_homography(imgs[i - 1], imgs[i])
            if H is not None:
                homography_matrix_list.append(np.linalg.inv(H) @ homography_matrix_list[-1])

        corners = []
        for i, img in enumerate(imgs):
            h, w = img.shape[:2]
            img_corners = np.array([
                [0, 0, 1], [w, 0, 1], [w, h, 1], [0, h, 1]
            ])  
            transformed_corners = (homography_matrix_list[i] @ img_corners.T).T
            transformed_corners = transformed_corners[:, :2] / transformed_corners[:, 2:3]  # Normalize by last coordinate
            corners.extend(transformed_corners)

        corners = np.array(corners)
        min_x, min_y = np.min(corners, axis=0).astype(int)
        max_x, max_y = np.max(corners, axis=0).astype(int)

        pano_width = max_x - min_x
        pano_height = max_y - min_y

        translation = np.array([
            [1, 0, -min_x],
            [0, 1, -min_y],
            [0, 0, 1]
        ])
        homographies = [translation @ H for H in homography_matrix_list]

        panorama = cv2.warpPerspective(base_img, homographies[center_index], (pano_width, pano_height))
        for i in range(len(imgs)):
            if i != center_index:
                warped_img = cv2.warpPerspective(imgs[i], homographies[i], (pano_width, pano_height))
                mask = warped_img > 0
                panorama[mask] = warped_img[mask]

        panorama = self.crop_black_borders(panorama)

        return panorama, homography_matrix_list
