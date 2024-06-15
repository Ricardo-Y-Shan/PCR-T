"""Triangle Meshes to Point Clouds"""
"""Taken from exercises"""

import numpy as np
import torch 


def sample_point_cloud(vertices, faces, n_points):
    """
    Sample n_points uniformly from the mesh represented by vertices and faces
    :param vertices: Nx3 numpy array of mesh vertices
    :param faces: Mx3 numpy array of mesh faces
    :param n_points: number of points to be sampled
    :return: sampled points, a numpy array of shape (n_points, 3)
    """
    device = vertices.get_device()
    # ###############
    A,B,C= vertices[faces[:,0],:], vertices[faces[:,1],:], vertices[faces[:,2],:]

    areas= 0.5*torch.linalg.norm(torch.cross(B-A,C-A), axis=1)

    
    probs= areas/areas.sum()
    
    
    faces_shape = faces.shape[0]
    

#     random_triangles_to_sample=np.random.choice(faces_shape, size=n_points, p=probs)
    random_triangles_to_sample=torch.multinomial(probs, n_points, replacement=True)

    
    a,b,c= A[random_triangles_to_sample,:], B[random_triangles_to_sample,:], C[random_triangles_to_sample,:]
    # print(torch.cuda.memory_summary())
    # print(torch.cuda.mem_get_info())
    r1 = torch.rand(size=[n_points]).to(device)
    r2 = torch.rand(size=[n_points]).to(device)
    
    u = (1 - torch.sqrt(r1)).unsqueeze(1)
    u = torch.cat([u, u, u], dim=1)
    v = (torch.sqrt(r1) * (1 - r2)).unsqueeze(1)
    v = torch.cat([v, v, v], dim=1)
    w = (torch.sqrt(r1) * r2).unsqueeze(1)
    w = torch.cat([w, w, w], dim=1)

    pts = u * a + v * b + w * c

    return pts.float()
    # ###############
