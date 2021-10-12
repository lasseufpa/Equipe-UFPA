%Aldebaro.
%Code to resize the digits.
resizing_factor = 1/10;
input_directory = './output_imgs/'; %end with /
output_directory = 'small_images/'; %end with /
mkdir(output_directory)

%read all files
fileList = dir([input_directory '*.png']);

[N,~]=size(fileList);
disp(['Found ' num2str(N) ' files in folder ' input_directory])

for i=1:N
    fileName = [fileList(i).folder '/' fileList(i).name];
    im=imread(fileName);
    final_image = imresize(im, resizing_factor);
    %size(final_image)
    output_filename = [output_directory '/' fileList(i).name];
    imwrite(final_image, output_filename)
    disp(['Wrote ' output_filename])
    %pause
end