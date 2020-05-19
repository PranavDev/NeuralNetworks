function B = expand(A, S)
if nargin < 2
    error('Size vector must be provided.  See help.');
end

SA = size(A);

if length(SA) ~= length(S)
   error('Length of size vector must equal ndims(A).  See help.')
elseif any(S ~= floor(S))
   error('The size vector must contain integers only.  See help.')
end

T = cell(length(SA), 1);
for ii = length(SA) : -1 : 1
    H = zeros(SA(ii) * S(ii), 1);
    H(1 : S(ii) : SA(ii) * S(ii)) = 1;
    T{ii} = cumsum(H);
end

B = A(T{:});