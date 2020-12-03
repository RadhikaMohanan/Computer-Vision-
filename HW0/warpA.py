def warp(im, A, output_shape):
    """ Warps (h,w) image im using affine (3,3) matrix A
    producing (output_shape[0], output_shape[1]) output image
    with warped = A*input, where warped spans 1...output_size.
    Uses nearest neighbor interpolation."""
    import numpy as np
    xv, yv = np.meshgrid(range(output_shape[0]), range(output_shape[1]), indexing='xy')
    image_warped=np.zeros(output_shape)
    x = xv.flatten()
    y = yv.flatten()
    fp = np.vstack((x,y,np.ones((1,len(x)))))
    print(fp.shape)
    
    psource=np.dot(np.linalg.inv(A),fp)
    x_new=np.array(np.round(psource[0,:]),dtype='int')
    y_new=np.array(np.round(psource[1,:]),dtype='int')
    for i in range(len(x_new)):
        if x_new[i]<0 or y_new[i]<0 or x_new[i]>199 or y_new[i]>149:
            image_warped[x[i],y[i]]=0
        else:
            image_warped[x[i],y[i]]=im[x_new[i],y_new[i]]

    
    
    return image_warped
