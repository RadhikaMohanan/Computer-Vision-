import numpy as np
import cv2


def computeH(x1, x2):
	#Q2.2.1
	#Compute the homography between two sets of points
    # a = np.zeros((2*x1.shape[0],9))
    # for i in range(0,x1.shape[0]):
    #     a[2*i,:]=np.array([-x2[i, 0], -x2[i, 1], -1, 0, 0, 0, x1[i, 0]*x2[i, 0], x1[i, 0]*x2[i, 1], x1[i, 0]])
    #     a[(2*i)+1,:]=np.array([0, 0, 0, -x2[i, 0], -x2[i, 1], -1, x1[i, 1]*x2[i, 0], x1[i, 1]*x2[i, 1], x1[i, 1]])
    # diag_matx,eig_vecs=np.linalg.eig(a.T@a)
    # h=eig_vecs[:,np.argmin(diag_matx)]
    # H2to1=h.reshape((3,3))
    assert(x1.shape[0] == x2.shape[0])
    assert(x1.shape[1] == 2)
    a = []
    for i in range(x1.shape[0]):
        a.append([-x2[i, 0], -x2[i, 1], -1, 0, 0, 0, x2[i, 0]*x1[i, 0], x1[i, 0]*x2[i, 1], x1[i, 0]])
        a.append([0, 0, 0, -x2[i, 0], -x2[i, 1], -1, x2[i, 0]*x1[i, 1], x1[i, 1]*x2[i, 1], x1[i, 1]])
    U, S, V = np.linalg.svd(np.asarray(a))
    # H2to1 = V[-1, :].reshape(3, 3)/V[-1, -1]
    H2to1 = V[-1, :].reshape(3, 3)
    return H2to1


def computeH_norm(x1, x2):
 	# #Q2.2.2
 	# #Compute the centroid of the points

  #   x1_centroid=[np.mean(x1[:,0]),np.mean(x1[:,1])]
  #   x2_centroid=[np.mean(x2[:,0]),np.mean(x2[:,1])]
    
  #   #Shift the origin of the points to the centroid
  #   p1=np.zeros((x1.shape[0]), dtype=float)
  #   p2=np.zeros((x2.shape[0]), dtype=float)
  #   #Shift the origin of the points to the centroid
  #   for i in range(x1.shape[0]):
  #       p1[i]=np.sqrt((x1[i,0]-x1_centroid[0])**2+(x1[i,1]-x1_centroid[1])**2)
  #       p2[i]=np.sqrt((x2[i,0]-x2_centroid[0])**2+(x2[i,1]-x2_centroid[1])**2)
 
  #   #Normalize the points so that the largest distance from the origin is equal to sqrt(2)
  #   S1=np.sqrt(2)/(np.max(p1))
  #   S2=np.sqrt(2)/(np.max(p2))
  #   t1=np.eye(3)
  #   t2=np.eye(3)
  #   for i in range(0,2):
  #       t1[i,i]=S1
  #       t2[i,i]=S2
  #   t11=np.eye(3)
  #   t22=np.eye(3)
  #   for i in range(0,2):
  #       t11[i,2]=-x1_centroid[i]
  #       t22[i,2]=-x2_centroid[i]
  #   #Similarity transform 1
  #   T1=t1@t11
    
 	# #Similarity transform 2
  #   T2=t2@t22
  #   #Compute homography
  #   x1_diff=np.asarray([[(x[0]-x1_centroid[0]),(x[1]-x1_centroid[1])] for x in x1])
  #   x2_diff=np.asarray([[(x[0]-x2_centroid[0]),(x[1]-x2_centroid[1])] for x in x2])
  #   x11=x1_diff/S1
  #   x22=x2_diff/S2
  #   H=computeH(x11, x22) #should it be divided byx1
  #   #Denormalization
  #   H1=np.dot(np.linalg.inv(T1),H)
  #   H2to1=np.dot(H1,T2)
  #   return H2to1
    x1_centroid=[np.mean(x1[:,0]),np.mean(x1[:,1])]
    x2_centroid=[np.mean(x2[:,0]),np.mean(x2[:,1])]
        #Shift the origin of the points to the centroid
   
    p1=np.zeros((x1.shape[0]), dtype=float)
    p2=np.zeros((x2.shape[0]), dtype=float)
     #Shift the origin of the points to the centroid
    for i in range(x1.shape[0]):
        p1[i]=np.sqrt((x1[i,0]-x1_centroid[0])**2+(x1[i,1]-x1_centroid[1])**2)
        p2[i]=np.sqrt((x2[i,0]-x2_centroid[0])**2+(x2[i,1]-x2_centroid[1])**2)
        #Normalize the points so that the largest distance from the origin is equal to sqrt(2)
    S1=np.sqrt(2)/(np.amax(p1))
    S2=np.sqrt(2)/(np.amax(p2))
    t1=np.eye(3)
    t2=np.eye(3)
    for i in range(0,2):
        t1[i,i]=S1
        t2[i,i]=S2
    t11=np.eye(3)
    t22=np.eye(3)
    for i in range(0,2):
        t11[i,2]=-x1_centroid[i]
        t22[i,2]=-x2_centroid[i]
     #Similarity transform 1
    T1=t1@t11
    
 	 #Similarity transform 2
    T2=t2@t22
    #Compute homography
    x1_homo=np.vstack((x1.T,np.ones((x1.shape[0]))))
    x2_homo=np.vstack((x2.T,np.ones((x2.shape[0]))))
    x11=T1@x1_homo
    x22=T2@x2_homo
    x11=x11/x11[2,:]
    x22=x22/x22[2,:]
    x11_new=x11.T[:,0:2]
    x22_new=x22.T[:,0:2]
    H=computeH(x11_new, x22_new) #should it be divided byx1
    #Denormalization
    H1=np.dot(np.linalg.inv(T1),H)
    H2to1=np.dot(H1,T2)
    return H2to1

def computeH_ransac(locs1, locs2, opts):
 	#Q2.2.3
 	#Compute the best fitting homography given a list of matching points
    max_iters = opts.max_iters  # the number of iterations to run RANSAC for
    inlier_tol = opts.inlier_tol # the tolerance value for considering a point to be an inlier
    max_inliers=-np.inf
    x1_homo=np.vstack((locs1.T,np.ones((locs1.shape[0]))))
    x2_homo=np.vstack((locs2.T,np.ones((locs2.shape[0]))))
    for i in range(0,max_iters):
        idx_r=np.random.choice(locs1.shape[0],4)
        r1=locs1[idx_r,:]
        r2=locs2[idx_r,:]
        H_r=computeH(r1,r2)
        
        x1_eg=np.matmul(H_r,x2_homo)
        x1_eg=x1_eg/x1_eg[2,:]
        inliers=np.zeros(locs1.shape[0])

        for j in range(np.size(inliers)):
            error=np.square(x1_homo[0,j]-x1_eg[0,j])+np.square(x1_homo[1,j]-x1_eg[1,j])
            if np.sqrt(error)<inlier_tol:
                inliers[j]=1
        no_inliers=np.sum(inliers)
        # print(no_inliers)
        if no_inliers > max_inliers:
            max_inliers=no_inliers
            best_inliers=inliers

    idx=[i for i in range(np.size(best_inliers)) if (best_inliers[i]==1)]
    bestH2to1=computeH_norm(locs1[idx,:],locs2[idx,:])
    inliers=best_inliers
    return bestH2to1, inliers



def compositeH(H2to1, template, img):
 	
 	#Create a composite image after warping the template image on top
 	#of the image using the homography

 	#Note that the homography we compute is from the image to the template;
 	#x_template = H2to1*x_photo
 	#For warping the template to the image, we need to invert it.
    mask=np.ones(template.shape)
    template=cv2.transpose(template)
    mask=cv2.transpose(mask)
       
    # 	 #Create mask of same size as template
    h_mask=cv2.warpPerspective(mask,np.linalg.inv(H2to1),(img.shape[0],img.shape[1]))
    # 	 #Warp mask by appropriate homography
    h_mask=cv2.transpose(h_mask)
    idx=np.nonzero(h_mask)
    # 	 #Warp template by appropriate homography
    new_template=cv2.warpPerspective(template,np.linalg.inv(H2to1),(img.shape[0],img.shape[1]))
    # 	 #Use mask to combine the warped template and the image
    new_template=cv2.transpose(new_template)
    img[idx]=new_template[idx]
    composite_img=img
    # composite_img=template
    # for i in range(img.shape[0]):
    #     for j in range(img.shape[1]):
    #         # if img[i,j,:]==[255,255,255]:
    #         #     continue
    #         # elif img[i,j,:]==[0,0,0]:
    #         #     continue
    #         composite_img[i,j,:]=img[i,j,:]
    # idx=np.nonzero(template)
    # img[idx]=template[idx]
    # composite_img=img
    
    
    return composite_img


