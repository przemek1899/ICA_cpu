
%% Get the axes that point towards the data.
ha1 = sort(findobj(get(1,'children'),'type','axes'));
ha2 = sort(findobj(get(2,'children'),'type','axes'));
% Get the correct maps.
tmp1 = cell2mat(get(ha1,'position'));
[~,tmp1] = min(sum(tmp1(:,1:2),2));
ha1 = ha1(tmp1);
tmp1 = cell2mat(get(ha2,'position'));
[~,tmp1] = min(sum(tmp1(:,1:2),2));
ha2 = ha2(tmp1);
% Create a new figure and move some axes there.
hf = figure;
set(ha2,'parent',3)
set(ha1,'parent',3)
ha = [ha1;ha2];
% Close the old figures.
% close(1)
% close(2)
% Remove extra values and lines --- keep only the images.
delete(findobj(get(ha(1),'children'),'type','line'))
delete(findobj(get(ha(1),'children'),'type','text'))
delete(findobj(get(ha(2),'children'),'type','line'))
delete(findobj(get(ha(2),'children'),'type','text'))
% Get the thresholded (thresheld?) image and add transparency.
hi = get(ha(1),'children');
im = get(hi,'cdata');
set(hi,'alphaDataMapping','direct','alphaData',im*128);
colormap gray % put in grayscale
% Remove function from the top image.
set(hi,'ButtonDownFcn','');
set(ha,'position',[0.0071    0.0048    0.9893    0.9952])
