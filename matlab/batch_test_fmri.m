%% INITIALIZATION
% How many components do we want to obtain?
NCOMPONENTS = 40;
% Select file to load.
fprintf(1,'Select data file to load...\t');
[filename,filepath] = uigetfile('*.gz');
% Load file.
fprintf(1,'loading, please wait.\n');
try
    nii = load_nii(fullfile(filepath,filename));
catch me
    warning(me.identifier,me.message);
    fprintf(2,'Please, consider using \nreslice_nii(fullfile(filepath,filename),fullfile(filepath,[''r_'' filename]))\n');
    return;
end


%% --- Data selection
% Unfiltered (raw) data. The data should be 4D size [NX,NY,NZ,NFRAMES],
% with NFRAMES being the number of time frames and NX, NY, NZ the number of
% samples along each axis.
data = double(nii.img);
% i tutaj mam dodac smooth3
data = smooth3(data);
[nx,ny,nz,nframes] = size(data);
% Uncomment the lines below to filter the data.
% fprintf(1,'Filtering data...\n');
% filter_vals = bsxfun(@times,gausswin(5,3)*gausswin(5,3)',reshape(gausswin(5,3),1,1,5));
% filter_vals = filter_vals /sum(filter_vals(:));
% data        = convn(data,filter_vals,'same');


%% PCA
% Apply PCA to obtain the principal components.
fprintf(1,'Using PCA to obtain %d principal components...\n',NCOMPONENTS);
% First, reshape the data so that they are size [NVOXELS,NFRAMES]. 
data_r = reshape(data,nx*ny*nz,nframes);
% Use that as the input to PCA. Due to how the PCA.M function is
% implemented, we need to transpose the input matrix.
result_pca = pca(detrend(data_r'),'NumComponents',NCOMPONENTS); 
% result_pca = pca(data_r','NumComponents',NCOMPONENTS); wyzej dodane tylko
% detrend
% Reshape the output to size [NX,NY,NZ,NCOMPONENTS].
result_pca = reshape(result_pca,[nx,ny,nz,NCOMPONENTS]);
result_pca = (result_pca-min(result_pca(:)))/(max(result_pca(:))-min(result_pca(:)));
% Plot the result using VIEW_NII.M.
nii_tmp = nii;
nii_tmp.hdr.dime.dim(5) = NCOMPONENTS;
nii_tmp.img = result_pca * (nii_tmp.hdr.dime.glmax-nii_tmp.hdr.dime.glmin)+nii_tmp.hdr.dime.glmin;
view_nii(nii_tmp);
fprintf(1,'Showing the principal components, press any key to continue.\n');
pause()


%% ICA
% Apply ICA to obtain the independent components.
fprintf(1,'Using ICA to obtain %d independent components...\n',NCOMPONENTS);
% First, reshape the data so that they are size [NVOXELS,NFRAMES]. 
data_r = reshape(data,nx*ny*nz,nframes);
% Use that as the input to ICA. 
result_ica = ica(data_r,'NumComponents',NCOMPONENTS);
reuslt_ica = (result_ica-min(result_ica(:)))/(max(result_ica(:))-min(result_ica(:)));
% Reshape the output to size [NX,NY,NZ,NCOMPONENTS].
result_ica = reshape(result_ica,[nx,ny,nz,NCOMPONENTS]);
% Plot the result using VIEW_NI.M.
nii_tmp2 = nii;
nii_tmp2.hdr.dime.dim(5) = NCOMPONENTS;
nii_tmp2.img = result_ica * (nii_tmp2.hdr.dime.glmax-nii_tmp2.hdr.dime.glmin)+nii_tmp2.hdr.dime.glmin;
view_nii(nii_tmp2);
