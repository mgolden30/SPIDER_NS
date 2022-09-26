function [nx_xx, nx_xy, nx_yy, ny_xx, ny_xy, ny_yy] = nematic_second_derivative_n( nx, ny, dx_vec )
  nx_xx    = 0*nx;
  nx_xy    = 0*nx;
  nx_yy    = 0*nx;
  ny_xx    = 0*nx;
  ny_xy    = 0*nx;
  ny_yy    = 0*nx;
  
  %sum      = 0*nx_x + 0*ny_y;
  %eighth_term_1 = 0*sum;
  %eighth_term_2 = 0*sum;

  %https://www.mathworks.com/matlabcentral/answers/16996-assign-multiple-variables  
  [dy dx dt] = feval(@(x) x{:}, num2cell(dx_vec));

  dim = size(nx);
  for j = 2:dim(1)-1
    for i = 2:dim(2)-1
      for k = 2:dim(3)-1
        %copy a stencil from nx and ny
        center = [nx(j,i,k) ny(j,i,k)];
        top    = [nx(j,i+1,k) ny(j,i+1,k)];
        bottom = [nx(j,i-1,k) ny(j,i-1,k)];
        left   = [nx(j-1,i,k) ny(j-1,i,k)];
        right  = [nx(j+1,i,k) ny(j+1,i,k)];
  
        tr = [nx(j+1,i+1,k) ny(j+1,i+1,k)];
        tl = [nx(j-1,i+1,k) ny(j-1,i+1,k)];
        br = [nx(j+1,i-1,k) ny(j+1,i-1,k)];
        bl = [nx(j-1,i-1,k) ny(j-1,i-1,k)];

        %Now we will flip vectors as needed.
        %center*top' is a dot product.
        if center*top' < 0
          top = -top;
        end
        if center*bottom' < 0
          bottom = -bottom;
        end
        if center*right' < 0
          right = -right;
        end
        if center*left' < 0
          left = -left;
        end
        if center*tr' < 0
          tr = -tr;
        end
        if center*tl' < 0
          tl = -tl;
        end
        if center*br' < 0
          br = -br;
        end
        if center*bl' < 0
          bl = -bl;
        end
        
        nx_xx(j,i,k) = ( bottom(1) - 2*center(1) + top(1)   )/(dx*dx);
        nx_xy(j,i,k) = ( tr(1) - tl(1) - br(1) + bl(1)      )/(4*dx*dy);
        nx_yy(j,i,k) = ( left(1)   - 2*center(1) + right(1) )/(dy*dy);
        
        ny_xx(j,i,k) = ( bottom(2) - 2*center(2) + top(2)   )/(dx*dx);
        ny_xy(j,i,k) = ( tr(2) - tl(2) - br(2) + bl(2)      )/(4*dx*dy);
        ny_yy(j,i,k) = ( left(2)   - 2*center(2) + right(2) )/(dy*dy);
      end %k
    end %i
  end %j
end %func
