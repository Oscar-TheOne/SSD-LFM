function line = generate_twisted_bezier_curve(num_control_points, twist_factor, space_size,seed)  
   
    % 设置空间大小  
    x_max=space_size(1);
    y_max=space_size(2);
    z_max=space_size(3);
  
    % 随机生成控制点  
    control_points = seed.* [x_max, y_max, z_max];  
  
    % 应用扭曲变换  
    for i = 1:num_control_points  
        % 计算旋转角度（与索引相关）  
        theta = i * twist_factor * 2 * pi / num_control_points;  
          
        % 创建旋转矩阵（绕z轴旋转）  
        R = [cos(theta), -sin(theta), 0;  
             sin(theta), cos(theta), 0;  
             0, 0, 1];  
          
        % 应用旋转矩阵到控制点（这里只绕z轴旋转，可以根据需要调整）  
        control_points(i, :) = R * control_points(i, :)';  
    end   
    t = linspace(0, 1, 1000); % 参数化曲线的点  
    curve_points = zeros(length(t), 3);  
  
    for i = 1:length(t)  
        % 计算贝塞尔曲线上的点  
        point = control_points(1, :); % 从第一个控制点开始  
        for j = 2:num_control_points  
            point = (1 - t(i)) * point + t(i) * control_points(j, :);    
        end  
        curve_points(i, :) = point;  
    end 
    line=curve_points;
end  