%clear all
addpath(genpath('./utils/'))
dir_faust = 'D:/Data/MPI-FAUST/training/registrations/';
dir_save = 'D:/shape_completion/data/eigendecomposition/';

shape_idx = '004'
k = 100;
faust_file_name = [dir_faust , 'tr_reg_' , shape_idx , '_downsampled.ply'];
[mesh,~] = plyread(faust_file_name); 
full_mesh = [];
full_mesh.TRIV = cell2mat(mesh.face.vertex_indices) + 1; full_mesh.VERT = [mesh.vertex.x,mesh.vertex.y,mesh.vertex.z];
full_mesh.X = full_mesh.VERT(:,1);
full_mesh.Y = full_mesh.VERT(:,2);
full_mesh.Z = full_mesh.VERT(:,3);
full_mesh.n = size(full_mesh.VERT,1);
full_mesh.m = size(full_mesh.TRIV,1);
xyz_triv = [full_mesh.VERT(full_mesh.TRIV(:,1),:), full_mesh.VERT(full_mesh.TRIV(:,2),:),full_mesh.VERT(full_mesh.TRIV(:,3),:)];    

face_normals = cross(xyz_triv(:,4:6) - xyz_triv(:,1:3),xyz_triv(:,7:9) - xyz_triv(:,4:6));
norm_face_normals = sum(face_normals.^2,2).^0.5;
face_normals = face_normals./repmat(norm_face_normals,1,3);

faces_center_x = (xyz_triv(:,1) + xyz_triv(:,4) + xyz_triv(:,7))/3;
faces_center_y = (xyz_triv(:,2) + xyz_triv(:,5) + xyz_triv(:,8))/3;
faces_center_z = (xyz_triv(:,3) + xyz_triv(:,6) + xyz_triv(:,9))/3;

D = calc_dist_matrix(full_mesh);
grad = calc_grad(full_mesh.VERT,full_mesh.TRIV);
[Phi,S,Lambda,M] = extract_eigen_functions_new(full_mesh,k);
M = full(M); A = diag(M);
S = full(S); 
L = diag(1./diag(M))*S;
adj_VF = full(adjacency_VF(full_mesh.VERT,full_mesh.TRIV));
adj_VV = sparse(1*(adj_VF*adj_VF'>0));
F = full_mesh.TRIV;
V = [full_mesh.X,full_mesh.Y,full_mesh.Z];
save([dir_save,'downsampled_tr_reg_' , shape_idx],'Phi','L','Lambda','adj_VF','V', 'F','A','adj_VV')