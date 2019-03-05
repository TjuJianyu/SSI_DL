function [LSF, gain, residual, frameMatrix, audioFrameLength, fs] = computeLSF(audioPath, order, corrThreshold, downsamplingRate, useGPU)

%order = order/downsamplingRate;
D = dir(audioPath);
LSF = zeros(length(D) - 2, order);
gain = zeros(length(D) - 2, 1);
residual = [];
frameIndex = 0;
frameMatrix = zeros(length(D) - 2, 1);
maxCorr = [];

for kAudio = 1 : length(D) - 2
% for kAudio = length(D) - 2 : -1 : 1
    filenameString = D(kAudio + 2).name;
    filePath = sprintf('%s/%s', audioPath, filenameString);
    [s, fs] = audioread(filePath);
    ss = s;
    fs = fs/downsamplingRate;
    s = decimate(s, downsamplingRate);
    s = s(:, 1);
    s = filter([1 -0.95], 1, s); % pre-emphasis filter
%     s = filter([1 -0.94], [1 -0.94], s); % pre-emphasis filter
%     s = filter([1 -0.42 -0.13], 3.334*[1 -1.39 0.52], s);
    exactAudioFrameLength = fs/60;
    audioFrameLength = floor(exactAudioFrameLength); % 2*fs/60; %%
    signalLength = length(s);
    
%     padded = s;
    padded = [zeros(ceil(audioFrameLength/2),1); s; zeros(ceil(audioFrameLength/2),1)];
    
    for kFrame = 1 : floor(signalLength/exactAudioFrameLength)-1
        startIndex = round((kFrame-1)*exactAudioFrameLength) + 1;
        stopIndex = startIndex + 2*audioFrameLength - 1;
        frame = padded( startIndex : stopIndex).*hamming(2*audioFrameLength);
        frameIndex = frameIndex + 1;
        corr = xcorr(frame); % corr = xcorr(frame, 'unbiased');
        [LPCcoeff, g] = lpc(frame, order);
        LPCcoeff(isnan(LPCcoeff)) = 0;
        gain(frameIndex, :) = g;
        LSF(frameIndex, :) = poly2lsf(LPCcoeff);
        maxCorr = [maxCorr max(corr)];
%         if max(corr) < corrThreshold
%             LSF(frameIndex, :) = zeros(size(LSF(frameIndex, :)));
%         end
        estimatedSignal = filter([0 -LPCcoeff(2 : end)], 1, frame);
        residual = [residual; frame' - estimatedSignal'];
        size_res = size(residual, 2);
    end
    if ~rem(signalLength/exactAudioFrameLength, 1) == 0
        kFrame = kFrame + 1;
        tmp = s(round((kFrame-1)*exactAudioFrameLength) + 1 : end);
%         tmp(1 : ceil(audioFrameLength/2)) = 0;
        frame = tmp.*hamming(length(tmp));
        frameIndex = frameIndex + 1;
        LPCcoeff = lpc(frame, order);
        LPCcoeff(isnan(LPCcoeff)) = 0;
        LSF(frameIndex, :) = poly2lsf(LPCcoeff);
        estimatedSignal = filter([0 -LPCcoeff(2 : end)], 1, frame);
        residual(frameIndex, :) = zeros(1, size_res);
        residual(frameIndex, 1 : length(tmp)) = frame - estimatedSignal;
        length(tmp)
    end
    frameMatrix(kAudio) = kFrame;
%     frameMatrix(length(D) - 2 - kAudio + 1) = kFrame;
    fprintf('Song %d processed \n', kAudio);
    figure; plot(LSF(:,1), '.');
%     figure; plot(LSF(:,12), '.');
%     figure; plot(maxCorr);
end
% Save results
save('LSFresults/mat/LSF12vowels.mat', 'LSF');
save('LSFresults/mat/gain12vowels.mat', 'gain');
save('LSFresults/mat/residual12vowels.mat', 'residual');
save('LSFresults/mat/frameMatrix12vowels.mat', 'frameMatrix');
save('LSFresults/mat/audioFrameLength12vowels.mat', 'audioFrameLength');
save('LSFresults/mat/maxCorr12vowels.mat', 'maxCorr');
% save('LSF12full.mat', 'LSF');
% save('gain12full.mat', 'gain');
% save('residual12full.mat', 'residual');
% save('frameMatrix12full.mat', 'frameMatrix');
% save('audioFrameLength12full.mat', 'audioFrameLength');
% save('maxCorr12full.mat', 'maxCorr');