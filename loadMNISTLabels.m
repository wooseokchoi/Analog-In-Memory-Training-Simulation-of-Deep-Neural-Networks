function labels = loadMNISTLabels(filename)
%loadMNISTLabels returns a [number of MNIST images]x1 matrix containing
%the labels for the MNIST images

fp = fopen(filename, 'rb');
assert(fp ~= -1, ['Could not open ', filename, '']);

magic = fread(fp, 1, 'int32', 0, 'ieee-be');
assert(magic == 2049, ['Bad magic number in ', filename, '']);

numLabels = fread(fp, 1, 'int32', 0, 'ieee-be');

labels = fread(fp, inf, 'unsigned char');

assert(size(labels,1) == numLabels, 'Mismatch in label count');

fclose(fp);

% change format like [0, 1, 0, 0, 0, 0, 0, 0, 0, 0];
temp=[];
temp_labels=[];
for i=1:length(labels)
    temp=zeros(10,1);
    if labels(i)==0
        temp(10)=1;
    else
        temp(labels(i))=1;
    end
    temp_labels=[temp_labels,temp];
end
labels=temp_labels;
end