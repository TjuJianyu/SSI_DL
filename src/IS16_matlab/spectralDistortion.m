function [SD, meanSD] = spectralDistortion(LSFth, LSFout, fs, N, n0, n1)

SD = zeros(size(LSFth, 1), 1);

for kFrame = 1 : size(LSFth, 1)
    LPCth = lsf2poly(LSFth(kFrame, :))';
    LPCout = lsf2poly(LSFout(kFrame, :))';
%    disp(LPCth)
%    disp(LPCout)
    [freqResponse_th, ~] = freqz(1, LPCth, N, fs);
    [freqResponse_out, ~] = freqz(1, LPCout, N, fs);
    disp(freqResponse_th)
    freqz_th = freqResponse_th(n0 : n1)';
    freqz_out = freqResponse_out(n0 : n1)';
%     freqz_th = freqResponse_th(n0 : n1)'/norm(freqResponse_th(n0 : n1)');
%     freqz_out = freqResponse_out(n0 : n1)'/norm(freqResponse_out(n0 : n1)')';
    absfreqth = abs(freqz_th);
    absoluteRatio = abs(freqz_th)./abs(freqz_out);
    logValue = log10(absoluteRatio.^2);
    bigSum = sum((10*logValue).^2);
    scall = 1/(n1-n0);
    asum = scall*bigSum;
    sd = sqrt( asum);
    
    SD(kFrame) = sd;
end
disp(SD)
meanSD = mean(SD);