# import pdb
import glob
import cv2
import os
import numpy as np
import matplotlib.pyplot as plt

class PanaromaStitcher():
    def __init__(self):
        pass

    def make_panaroma_for_images_in(self,path,resize=False):
        imf = path
        all_images = sorted(glob.glob(imf+os.sep+'*'))
        print('Found {} Images for stitching'.format(len(all_images)))

        images = [cv2.imread(img) for img in all_images]
        if(resize): 
            print("Images resized")
            images = self.resize_images(images)
        images = [cv2.cvtColor(img, cv2.COLOR_BGR2RGB) for img in images]

        homography_matrix_list =[]
        stitched_image=images[0]

        for i in range(1,len(images)):
            H = self.homography_matrix(stitched_image,images[i])
            if H is None:
                continue
            homography_matrix_list.append(H)
            stitched_image=self.warp_images(stitched_image,images[i],H)

        return stitched_image, homography_matrix_list 

    def homography_matrix(self,image1,image2,threshold=0.75):

        sift = cv2.SIFT_create()
        keypoints1, descriptors1 = sift.detectAndCompute(image1, None)
        keypoints2, descriptors2 = sift.detectAndCompute(image2, None)

        bf = cv2.BFMatcher() #Brute Force matcher
        matches = bf.knnMatch(descriptors1,descriptors2,k=2)

        good_matches = []
        for m, n in matches:
            if m.distance < threshold * n.distance:
                good_matches.append(m)
        
        if(len(good_matches)>4):
            src_pts = np.float32([keypoints1[m.queryIdx].pt for m in good_matches]).reshape(-1, 2)
            dst_pts = np.float32([keypoints2[m.trainIdx].pt for m in good_matches]).reshape(-1, 2)
            H, inliers = self.RANSAC(src_pts,dst_pts)
            return H
        else:
            print("Not enough good matches found")
            return None
        

    def compute_homography(self,src_pts, dst_pts): 
        A=[]
        for i in range(len(src_pts)):
            x1, y1 = src_pts[i][0], src_pts[i][1]
            x2, y2 = dst_pts[i][0], dst_pts[i][1]
            z1=z2=1
            A_i = [  [0, 0, 0, -z2*x1, -z2*y1, -z2*z1, y2*x1, y2*y1, y2*z1],
                    [z2*x1, z2*y1, z2*z1, 0, 0, 0, -x2*x1, -x2*y1, -x2*z1]  ]
            A.extend(A_i)

        A = np.array(A)
        _, _, V = np.linalg.svd(A)
        H = V[-1].reshape((3, 3))

        return H / H[2, 2]  # Normalize H

    def RANSAC(self,src_pts,dst_pts,max_iters=1000,sigma=1):
        threshold = np.sqrt(5.99) * sigma
        best_H = None
        best_inliers = []
        max_inliers = 0

        for _ in range(max_iters):
            # Step 1: Randomly sample 4 points
            random_idxs = np.random.choice(src_pts.shape[0], 4, replace=False)
            src_sample = src_pts[random_idxs]
            dst_sample = dst_pts[random_idxs]

            # Step 2: calculate H
            H = self.compute_homography(src_sample, dst_sample)

            src_pts_homogeneous = np.hstack([src_pts, np.ones((src_pts.shape[0], 1))])  # Convert to homogeneous coordinates
            projected_pts = np.dot(H, src_pts_homogeneous.T).T  # Apply homography
            projected_pts = projected_pts[:, :2] / projected_pts[:, 2].reshape(-1, 1)
            
            # Step 4: Calculate the error (distance between projected points and destination points)
            errors = np.linalg.norm(dst_pts - projected_pts, axis=1)
            
            # Step 5: Find inliers where the error is below the threshold
            inliers = np.where(errors < threshold)[0]
            
            # Step 6: If the number of inliers is larger than the previous best, update the best model
            if len(inliers) > max_inliers:
                max_inliers = len(inliers)
                best_H = H
                best_inliers = inliers
            
        if len(best_inliers) > 4:
            best_H = self.compute_homography(src_pts[best_inliers], dst_pts[best_inliers])
        
        return best_H, best_inliers
    
    

    def warp_images(self, image1, image2, H):
        # Step 5: Warp image1 to image2's perspective using the homography matrix
        height1, width1 = image1.shape[:2]
        height2, width2 = image2.shape[:2]

        # Calculate the dimensions of the panorama (bounding box of the result)
        corners_image1 = np.float32([[0, 0], [0, height1], [width1, height1], [width1, 0]]).reshape(-1, 1, 2)
        corners_image2 = np.float32([[0, 0], [0, height2], [width2, height2], [width2, 0]]).reshape(-1, 1, 2)

        # Warp the corners of image1 to find the bounding box for the new panorama
        warped_corners_image1 = cv2.perspectiveTransform(corners_image1, H)
        all_corners = np.concatenate((warped_corners_image1, corners_image2), axis=0)

        # Find the bounding box of the stitched panorama
        [x_min, y_min] = np.int32(all_corners.min(axis=0).ravel() - 0.5)
        [x_max, y_max] = np.int32(all_corners.max(axis=0).ravel() + 0.5)

        # Set a reasonable minimum size for the bounding box to avoid excessive black space
        x_min = max(x_min, -width2)  # Prevent excessive negative space on left
        y_min = max(y_min, -height2) # Prevent excessive negative space on top

        # Calculate the translation matrix to shift the panorama into the positive region of the bounding box
        translation_dist = [-x_min, -y_min]
        H_translation = np.array([[1, 0, translation_dist[0]], [0, 1, translation_dist[1]], [0, 0, 1]])

        # Warp image1 using the homography matrix combined with the translation matrix
        panorama = cv2.warpPerspective(image1, H_translation.dot(H), (x_max - x_min, y_max - y_min))

        # Step 6: Place image2 in the panorama using translation
        panorama[translation_dist[1]:height2 + translation_dist[1], translation_dist[0]:width2 + translation_dist[0]] = image2

        return panorama


    def display(self,panorama,save_image=False,save_path='panaroma.jpg'):
        panorama_rgb = cv2.cvtColor(panorama, cv2.COLOR_BGR2RGB)
        plt.imshow(panorama_rgb)
        plt.axis('off')  # Hide axis labels
        plt.show()
        
        if save_image:
            cv2.imwrite(save_path, panorama)
    
    def resize_images(self,images, scale=0.5):
        resized_images = []
        for img in images:
            height, width = img.shape[:2]
            new_size = (int(width * scale), int(height * scale))
            resized_images.append(cv2.resize(img, new_size, interpolation=cv2.INTER_LINEAR))
        return resized_images
    
stitcher=PanaromaStitcher()
pan,homo=stitcher.make_panaroma_for_images_in('ES666-Assignment3/Images/I3/',resize=True)
stitcher.display(pan,save_image=True,save_path='panaroma_I3.jpg')
    
