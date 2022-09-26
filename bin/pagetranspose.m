function Y = pagetranspose(X)
  s = size(X);
  assert( numel(s)>2 );
  X = reshape(X, [s(1), s(2), prod(s(3:end))] );
  Y = zeros( [s(2), s(1), prod(s(3:end))] );
  for k=1:size(X,3)
    Y(:,:,k) = X(:,:,k)';
  end
  Y = reshape(Y, s);
end