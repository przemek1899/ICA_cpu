function [ nii_tmp ] = do_segmentation( nii, NCOMPONENTS)


%filepath = 'C:\Users\pteodors\Desktop\nifti_matlab\ds115\sub001\BOLD\task001_run001\bold.nii.gz';
%nii = load_nii('bold.nii.gz');

data = double(nii.img);
data = data(:,:,17:end,:);
for i=1:size(data,4)
    data(:,:,:,i) = smooth3(data(:,:,:,i));
end
[nx,ny,nz,nframes] = size(data);
data_r = reshape(data,nx*ny*nz,nframes);

result_pca = pca(detrend(data_r'),'NumComponents',NCOMPONENTS); 
%result_pca = pca(data_r','NumComponents',NCOMPONENTS);

% start segmentation here
% normalizing values to Z score
pca_zscore = zscore(result_pca);

% applying a threshold
t = 2;
AT = pca_zscore > t;
AT2 = pca_zscore < -t;
result_pca = AT;%pca_zscore.*AT;

% end of segmentation

result_pca = reshape(result_pca,[nx,ny,nz,NCOMPONENTS]);

% co to jest to linijka nizej ???
result_pca = (result_pca-min(result_pca(:)))/(max(result_pca(:))-min(result_pca(:)));
% Plot the result using VIEW_NII.M.
nii_tmp = nii;
nii_tmp.hdr.dime.dim(5) = NCOMPONENTS;
nii_tmp.img = result_pca * (nii_tmp.hdr.dime.glmax-nii_tmp.hdr.dime.glmin)+nii_tmp.hdr.dime.glmin;


end

