function data = generate_data(nvoxels,nframes, rows, cols)

multrows = ceil (nvoxels / rows);
multcols = ceil (nframes / cols);

data = zeros(rows,cols);
tmp  = zscore(rand(rows,cols),0,1);
for m = 1:rows,
    data(m,:) = randn(1,rows)*tmp;
end

data = repmat(data, multrows, multcols);
data = data(1:nvoxels,1:nframes);


% add some noise to the data
for m=1:nvoxels,
    data(m,:) = data(m,:) + randn(1,nframes);
end

filename = strcat(int2str(nvoxels), 'x', int2str(nframes), '.bin');
% write data to binary file
fileID = fopen(filename,'w');
fwrite(fileID,data,'float32');
fclose(fileID);