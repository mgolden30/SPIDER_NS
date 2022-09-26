function [f_x, f_y, f_z, f_t] = nonuniform_gradient(f, grid)
  %{
  Estimate gradients of a function on a nonuniform mesh
  %}

  f_x = 0*f;
  f_y = 0*f;
  f_z = 0*f;
  f_t = 0*f;
  
  gridx = grid{1};
  gridy = grid{2};
  gridz = grid{3};
  gridt = grid{4};
  
  s = size(f);
  for i=2:s(1)-1
    x1 = gridx(i-1) - gridx(i); %spacing backwards (negative)
    x2 = gridx(i+1) - gridx(i); %spacing forwards  (positive)
    for j=2:s(2)-1
      y1 = gridy(j-1) - gridy(j); %spacing backwards (negative)
      y2 = gridy(j+1) - gridy(j); %spacing forwards  (positive)
      for k=2:s(3)-1
        z1 = gridz(k-1) - gridz(k); %spacing backwards (negative)
        z2 = gridz(k+1) - gridz(k); %spacing forwards  (positive)
        for l=2:s(4)-1
          t1 = gridt(l-1) - gridt(l); %spacing backwards (negative)
          t2 = gridt(l+1) - gridt(l); %spacing forwards  (positive)

          f_x(i,j,k,l) = (x1*x1*f(i+1,j,k,l) - x2*x2*f(i-1,j,k,l) + (x2*x2 - x1*x1)*f(i,j,k,l))/(x1*x2*(x1-x2));
          f_y(i,j,k,l) = (y1*y1*f(i,j+1,k,l) - y2*y2*f(i,j-1,k,l) + (y2*y2 - y1*y1)*f(i,j,k,l))/(y1*y2*(y1-y2));
          f_z(i,j,k,l) = (z1*z1*f(i,j,k+1,l) - z2*z2*f(i,j,k-1,l) + (z2*z2 - z1*z1)*f(i,j,k,l))/(z1*z2*(z1-z2));
          f_t(i,j,k,l) = (t1*t1*f(i,j,k,l+1) - t2*t2*f(i,j,k,l-1) + (t2*t2 - t1*t1)*f(i,j,k,l))/(t1*t2*(t1-t2));
        end
      end
    end
  end
end