function outstr = ls_win(pathname)
A = dir(pathname);
B = struct2cell(A);
C = B(1, :);

N = length(C);
%concate C to a single string
outstr = [];
    
for n = 1 : N
    outstr = [outstr, C{n}, '    '];
end

end