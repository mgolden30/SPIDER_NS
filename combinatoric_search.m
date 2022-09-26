function [models, etas, cs] = combinatoric_search( G, labels, desired_terms )
  %{
  PURPOSE:
  The purpose is reverse regression, in which all combinations of library
  terms are searched for the best models.

  INPUT:
  G - integrated matrix
  labels - cell of strings labeling each term
  desired_terms - integer > 1 
  %}

  [nw, nl] = size(G); %num windows and num library
  big = nchoosek(nl, desired_terms); %Combinations to check
  
  cs     = zeros(big, nl);
  etas   = zeros(big, 1);
  models = cell(big, 1);
  combinations = nchoosek( 1:nl, desired_terms );
  
  for i = 1:big
    combination = combinations(i,:);
    G_restricted = G(:, combination);
    
    [U, S, V] = svd(G_restricted, 'econ');
    c = V(:,end);
    
    cs(i,combination) = c;
    etas(i)   = norm( G_restricted*c )/ max( vecnorm(G_restricted*diag(c)) );
    model_str = "\eta = " + etas(i) + ", ";
    for j=1:desired_terms
      model_str =  model_str + c(j) + " " + labels( combination(j) ) + "  +  ";
    end
    models{i} = model_str;
  
    %If any term in the model is negligible, artificially inflate eta so it
    %gets thrown out.
    %mags = vecnorm(G_restricted*diag(c));
    %if( min(mags)/max(mags) < 1e-2 ) %Check if there is a negligible term
    %  etas(i) = 1; 
    %end
  end
  
  %bad = (etas < 0.001);     %Throw out identities
  %bad = 0*bad;
  %bad = bad | (etas > 0.5); %Throw out if eta is too high. Nonsense relations
  
  %models( bad ) = [];
  %etas( bad )   = [];
  %cs( bad, : )  = [];
  
  [etas, I] = sort( etas );
  cs = cs(I, :);
  models = models(I);
end