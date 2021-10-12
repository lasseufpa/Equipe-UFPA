%Code to generate all combinations of digits.
%You need to run (on Windows is del, on Linux rm):
%del ??100.png
%to remove the images with -0% (minus and 0).
%First two are a and b
%then last three are c, d and e.
%size(im) = 2087        2085    3
should_show_images = 0; %use 1 to display images
output_directory = 'output_imgs/'; %end with /
mkdir(output_directory)

%% Find column where a and b split (same column for c and d)
%im=imread('a1.png');
if 0
    im=imread('all_segments.png');
    imshow(im)
end

split_column = 770;
%im2=im;
%im2(:,1:split_column,:)=0;

%% Find row when ab and cd split
split_row = 1020;
%im2=im;
%im2(1:split_row,:,:)=0;
%imshow(im2)
split_cd = 590;
split_de = 758;

%% Loop over all values
for a=0:9
    afile = ['a' num2str(a) '.png'];
    im=imread(afile);
    final_image = im; %start with a contribution
    for b=0:9
        bfile = ['b' num2str(b) '.png'];
        im=imread(bfile);
        %add b contribution
        final_image(:,split_column:end,:) = im(:,split_column:end,:);
        for c=0:1
            cfile = ['c' num2str(c) '.png'];
            im=imread(cfile);
            %add c contribution
            final_image(split_row:end,1:split_cd,:) = im(split_row:end,1:split_cd,:);
            for d=0:1
                dfile = ['d' num2str(d) '.png'];
                im=imread(dfile);
                %add d contribution
                final_image(split_row:end,split_cd:split_de,:) = im(split_row:end,split_cd:split_de,:);
                for e=0:9
                    efile = ['e' num2str(e) '.png'];
                    im=imread(efile);
                    %add e contribution
                    final_image(split_row:end,split_de:end,:) = im(split_row:end,split_de:end,:);
                    if should_show_images == 1
                        imshow(final_image)
                        drawnow
                    end
                    %convention is to indicate the minus sign with 1 or 0 if positive
                    %below is -18%
                    output_filename = [output_directory num2str(a) num2str(b) ...
                        num2str(c) num2str(d) num2str(e) '.png'];
                    imwrite(final_image, output_filename)
                    disp(['Wrote ' output_filename])
                    %pause
                end
            end
        end
    end    
end