%S = load('./shapes/cat0.mat')
addpath(genpath('D:\Tools\export_fig-matlab'))
addpath(genpath('D:\shape_completion\data\'))
S = load('D:\shape_completion\data\faust_projections\range_data\res=100x180/tr_reg_096_001.mat'); 
gt = load('D:\shape_completion\data\faust_projections\range_data\labels/tr_reg_096_001.mat'); 
gt = gt.labels;
S=S.shape;
tmp = S.Z;
S.Z = -S.X;
S.X = tmp;
trisurf(S.TRIV,S.X,S.Y,S.Z,'Facecolor',[192,192,192]/255,'EdgeColor','none'); axis equal; 
hold
[S1,~] = plyread('D:\Data\MPI-FAUST\training\registrations\tr_reg_090.ply'); 
S1.TRIV = cell2mat(S1.face.vertex_indices) + 1; S1.VERT = [S1.vertex.x,S1.vertex.y,S1.vertex.z];

S1.VERT = S1.VERT*90
trisurf(S1.TRIV,S1.VERT(:,1),S1.VERT(:,2),S1.VERT(:,3),'Facecolor',[192,192,192]/255,'EdgeColor','none'); axis equal; 

axis off; set(gcf,'color','none');
%[imageData, alpha] = export_fig('vis4', '-png', '-transparent');

%for i = 1:20
%    p = randi(size(S.X,1))
%    plot3([S.X(p),S1.VERT(gt(p),1)+100],[S.Y(p),S1.VERT(gt(p),2)+100],[S.Z(p),S1.VERT(gt(p),3)],'color','magenta');
%end

[imageData, alpha] = export_fig('full_point_cloud', '-png', '-transparent');
%plywrite('D:\shape_completion\data\faust_projections\range_data\res=100x180/tr_reg_007_005.ply',...
%    S.TRIV,full_shape(gt,1:3))