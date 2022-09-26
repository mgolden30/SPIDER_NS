function vals = gpu_integrate_nd_nonuniform(data, derivs, grid, corners, size_vec, m, n, mask)
      %{
      PURPOSE:
      I rewrote that was used to integrate 2D+1 data to integrate an
      arbitrary N+1D data. The function is coded with backwards
      compatibility in mind, so that calling gpu_integrate_nd() on 2+1D
      yields the same results as calling gpu_integrate_data(). Therefore, I
      made the following decisions to maintain backwards compatibility:
      
      1. Keep the [dx2 dx1 dx3 dx4 ... dxn dxt]
      2. Integration is done by Riemmann Sum, instructions to switch to the
      trapezoidal rule can be found in the comments.
      3. Maintain the number and order of input arguments.
        
      To improve readability and ease of implementation I have temporarely
      removed the integration by parts functionality for derivatives.

      Future changes will aim to remove persistent variables by
      implementing functional programming for the sake of good practice.
      Reimplementing integration by parts for an arbitrary number of
      derivatives by rewriting gradient3d() function to support gpu use and
      generalize it to an arbitrary number of dimensions.

      I'm still pondering (and discussing, as the final decision is not
      ultimately mine) wether or not it is worth sacrificing backwards
      compatibility for the sake of cleaner code, improved funcionality and
      better performance.
      
      OLD PURPOSE:
      I am rewriting some old code in an attempt to speed up integration
      times.
      I want to accomplish two things with this integration routine:
      1. Cut down on redundancy. I want to call this function a single time
         per library term. Maybe a couple of times if different components 
         of the term can be integrated by parts in various ways.
      2. Leverage GPU acceleration. Sums scale logarithmically in parallel,
         instead of linearly in the sequential case.
      
      INPUT:
      data    - a N-Dimensional matrix containing the data to integrate 
                with Riemann Sums.
    
      derivs  - Deprecated. Used for backward compatibility.
                [1,1] for instance uses integration by parts to integrate 
                two derivatives along the first dimension (could be x or y
                depending on your convention)
    
      grid - a matrix containing the nonuniform gridspacing information

      corners - a Nxnum_windows matrix containing corners of your 
                integration domains.
    
      size_vec- [24 24 24 48] as an example. The size of the n-hypercube 
                you want to integrate over.
      
      m,n     - Powers of the polynomial weights. Use [4 4 4] for
                all when in doubt. 
      
      mask    - an arbitrary function you want to additionally multiply the
                weight by. Use this to get rid of questionable data.
    
      OUTPUT:
      vals - a vector of integrated values.
      %}
    
    % The first index of size is the number of dimensions, the second is
    % the number of windows of integration.
    [cdim,nw] = size(corners);

    dimensions = length(size_vec); % How many variables you are integrating
    dimension = dimensions; %sloppy naming
    
    % Assert all arguments have a coherent number of dimensions.
    assert((dimensions == cdim) && (cdim == ndims(data)), ...
        ['Incoherent dimensionality in arguments when ' ...
        'calling gpu_integrate_nd_nonuniform'])
    
    %This way you can pass a column or row vector.
    d = numel(derivs);
    
    %Make sure the grid matrix is in the correct format
    assert( size(grid,2) == cdim);
    
    if d == 0
        % d == 0 means that we're integrating the data itself, not its
        % derivatives; therefore, no integration by parts is needed.
        
        % Converting from the convension [dx2 dx1 dx3 dx4 ... dxn dxt] to the
        % more intuitive [dx1 dx2 dx3 ... dxn dxt] convension.
        %dx_vec(1:2) = dx_vec(2:-1:1);

        % idx is used to generate the spacing in the integration grid.
        idx = cell(dimensions, 1); % Init. memory to improve performance.
        
        for s = 1:dimensions
            idx{s} = 0:size_vec(s);
        end
        
        % vars provide a way to get the generalized coordinate at a given
        % element of the integration grid. Exemple vars{i}(r) will return 
        % the value of the i-th generalized coordinated at position r, 
        % where r is a position vector.
        vars = cell(1,dimensions); % Init. memory to improve performance.
        
        % The following lines are equivalent to meshgrid for n dimensions.
        [vars{:}] = ndgrid(idx{:});
        
    
        %First move data to the gpu.
        gpu_data = gpuArray(data);
        gpu_mask = gpuArray(mask);
    
        vals = zeros(nw,1); % Initialize memory to improve performance.
        vals = gpuArray(vals);
    
        %Multiply our data by the mask (if given)
        if ~isempty(gpu_mask)
            gpu_data = gpu_data.*gpu_mask;
        end
    
        for i = 1:nw
          % For some reason (ask Matt), our program doesn't seek to integrate
          % the data itself but the data multiplied by a high order (usually 
          % 4) polynomial that vanishes at the boundaries of integration.
    
          %The polynomial weight needs recalculated for every domain sadly.
          polynomial_weight = gpuArray(ones(size_vec+1)); % Init. memory
          
          edges   =  cell(1,dimensions); % nondimensional vectors of gridpoints, shifted and rescaled to be on [0,1]
          lengths = zeros(1,dimensions); % dimensional lengths of the spacetime cube
          spacing1=  cell(1,dimensions);
          spacing2=  cell(1,dimensions);
          for j = 1:dimensions
            grid2 = grid{j};
            edge = grid2( corners(j,i) + (0:(size_vec(j))) ); %read the nonuniform coordinates
            lengths(j) = edge(end) - edge(1);
            edges{j} = (edge - edge(1))/lengths(j);
            
            spacing1_temp      = edges{j} - circshift(edges{j},1);
            spacing1_temp(1)   = 0;
            spacing2_temp      = circshift(edges{j},-1) - edges{j};
            spacing2_temp(end) = 0;
            
            spacing1{j} = spacing1_temp;
            spacing2{j} = spacing2_temp;
          end
          
          vars = cell(1,dimension);
          spa1 = cell(1,dimension);
          spa2 = cell(1,dimension);
          [vars{:}] = ndgrid(edges{:});
          [spa1{:}] = ndgrid(spacing1{:});
          [spa2{:}] = ndgrid(spacing2{:});
          
          for j=1:dimensions
            polynomial_weight = polynomial_weight.*(vars{j}.^m(j)).*(1-vars{j}).^n(j);
            
            %also modify to include trapezoidal rule weights
            polynomial_weight = polynomial_weight.*(spa1{j} + spa2{j})/2;
          end

            % args stores the indexing arguments to gpu_data.
            args = cell(dimensions,1); % Initialize memory
            
            for j = 1:dimensions
                args{j} = corners(j,i) + idx{j};
            end
            
            % Isolate window of integration
            smalldata = gpu_data(args{:}).*polynomial_weight;
            
            % Code is currently implmented using Riemann Sum. To change to 
            % the trapezoidal rule uncomment the while loop.
    
%             while ~(max(size(smalldata))==1)
%                 smalldata = trapz(smalldata);
%             end               
            
            vals(i) = sum(smalldata,'all');
    
        end
        
        % Adjust for scaling.
        scaling = prod( lengths );
        vals = vals*scaling;
        
        % As computations are done asynchronously in the GPU, gather is
        % necessary to assure all computations are complete before 
        % returning the function value.
        vals = gather(vals);
    else
        % If d=/=0, then we resort to integration by parts. Integration by
        % parts is done recursively and doesn't support masks.
        no_mask = isempty(mask) | all(mask == ones(size(mask)));
        if ~no_mask & ndims(data) == 3
            % If 2+1D data just forward to Matt's code
            vals = gpu_integrate_data(data, derivs, dx_vec, corners, size_vec, m, n, mask);
        else
            
            %Need lengths for proper integration by parts
            lengths = zeros(nw,dimensions); % dimensional lengths of the spacetime cube
            for i = 1:nw
              for j = 1:dimensions
                grid2 = grid{j};
                edge = grid2( corners(j,i) + (0:(size_vec(j)))); %read the nonuniform coordinates
                
                lengths(i,j) = max(edge) - min(edge);
              end
            end
            %assert(no_mask, ['Integration by parts currently only ' ...
            %'supports the use of masks for 2+1D data'])
            partial_x = derivs(1);
            m2 = m;
            m2(partial_x) = m2(partial_x) - 1;
            n2 = n;
            n2(partial_x) = n2(partial_x) - 1;
            derivs(1) = [];
            vals1 = -m(partial_x)*gpu_integrate_nd_nonuniform(data, derivs, grid, corners, size_vec, m2, n, mask )./lengths(:,partial_x);
            vals2 =  n(partial_x)*gpu_integrate_nd_nonuniform(data, derivs, grid, corners, size_vec, m, n2, mask )./lengths(:,partial_x);
            vals = vals1 + vals2;
        end
    end
end

