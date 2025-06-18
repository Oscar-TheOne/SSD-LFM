function my_saveastiff(data, path)
options.message = false;
if exist(path,'file')
    delete(path);
end
saveastiff(data,path,options);
end