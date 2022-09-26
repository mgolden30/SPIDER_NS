function [U_x, U_y, U_z] = spectral_spatial_derivative_3D( U, Lx, Lz )
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
  
  U_x = 0*U;
  U_y = 0*U;
  U_z = 0*U;
  
  kx = 0:(Nx-1); kx(kx > Nx/2) = kx( kx > Nx/2 ) - Nx;
  kz = 0:(Nz-1); kz(kz > Nz/2) = kz( kz > Nz/2 ) - Nz;
  
  for j=1:Ny
    for k=1:Nz
      U_x(:,j,k) = real(ifft( 1i*kx'.*fft(U(:,j,k)) ));
    end
  end
  
  for i=1:Nx
    for k=1:Nz
        %Use external code for Chebyshev differentiation
        U_y(i,:,k) = fchd(U(i,:,k));
    end
  end
  
  for i=1:Nx
    for j=1:Ny
        U_z(i,j,:) = real(ifft( 1i*kz'.*fft(squeeze(U(i,j,:))) ));
    end
  end
  
  U_x = U_x/Lx*2*pi;
  U_z = U_z/Lz*2*pi;
end