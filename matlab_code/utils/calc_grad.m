function grad = calc_grad(vertices,faces)
%INPUT:  vertices - #vertices X 3 matrix of vertices coordinates: X,Y,Z
%        faces - #faces X 3 matrix of composing vertices indices for every face
%OUTPUT: grad - gradient operator, as a 3|F| X |V| matrix
N_faces = size(faces,1);
N_vertices = size(vertices,1);

edges1 = vertices(faces(:,3),:) - vertices(faces(:,2),:);%contains the edges opposite to the 1st vertex in every face
edges2 = vertices(faces(:,1),:) - vertices(faces(:,3),:);%contains the edges opposite to the 2nd vertex in every face
edges3 = vertices(faces(:,2),:) - vertices(faces(:,1),:);%contains the edges opposite to the 3rd vertex in every face

%calculate unit normal vetor to every face
normals = cross(vertices(faces(:,2),:) - vertices(faces(:,1),:),vertices(faces(:,3),:) - vertices(faces(:,2),:),2);
normals = normals./repmat(vecnorm(normals,2,2),1,3);

rot_edges1 = cross(normals,edges1,2); %contains the pi/2 rotated edges opposite to the 1st vertex in every face
rot_edges2 = cross(normals,edges2,2); %contains the pi/2 rotated edges opposite to the 2nd vertex in every face
rot_edges3 = cross(normals,edges3,2); %contains the pi/2 rotated edges opposite to the 3rd vertex in every face

E = sparse([],[],[],3*N_faces,N_vertices,9*N_faces); %The matrix E as defined in discrete_ops.pdf
sizeE = size(E);
E(sub2ind(sizeE,1:3*N_faces,repmat(faces(:,1),3,1)')) = reshape(rot_edges1,3*N_faces,1);
E(sub2ind(sizeE,1:3*N_faces,repmat(faces(:,2),3,1)')) = reshape(rot_edges2,3*N_faces,1);
E(sub2ind(sizeE,1:3*N_faces,repmat(faces(:,3),3,1)')) = reshape(rot_edges3,3*N_faces,1);

farea = faces_area(vertices,faces);
invGF = sparse(1:3*N_faces,1:3*N_faces,repmat(1./farea,3,1),3*N_faces,3*N_faces); %The inverse matrix of G_F as defined in discrete_ops.pdf
grad = 0.5*invGF*E;
end

