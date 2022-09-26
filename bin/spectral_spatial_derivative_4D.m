function [U_x, U_y, U_z] = spectral_spatial_derivative_4D( U, Lx, Lz )
  %{
  PURPOSE:
  Differentiate data spectrally. Data is 4d, with dimensions 1 and 3
  periodic, dimension 2 is Chebyshev nodes, and 4 is uniform in time. Dont
  worry about time derivatives.
  %}

  s = size(U);
  
  Nx= s(1);
  Ny= s(2);
  Nz= s(3);
  Nt= s(4);
  
  U_x = 0*U;
  U_y = 0*U;
  U_z = 0*U;
  
  kx = 0:(Nx-1): kx(kx > Nx/2) = kx( kx > Nx/2 ) - Nx;
  kz = 0:(Nz-1): kz(kz > Nz/2) = kz( kz > Nz/2 ) - Nz;
  
  for j=1:Ny
    for k=1:Nz
      for t=1:Nt
        U_x(:,j,k,t) = real(ifft( kx.*fft(U(:,j,k,t)) ));
      end
    end
  end
  
  for i=1:Nx
    for k=1:Nz
      for t=1:Nt
        %Use external code for Chebyshev differentiation
        U_y(i,:,k,t) = fchd(U(i,:,k,t));
      end
    end
  end
  
  for i=1:Nx
    for j=1:Ny
      for t=1:Nt
        U_z(i,j,:,t) = real(ifft( kz.*fft(U(i,j,:,t)) ));
      end
    end
  end
  
  U_x = U_x/Lx*2*pi;
  U_z = U_z/Lz*2*pi;
end