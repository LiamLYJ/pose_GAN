function [meanPCK, partPCK] = evaluatePCK(GT, PRD, givenThreshold)
    if nargin > 2
    	threshold = givenThreshold;
    else
        threshold = 0.2;
    end
    n = 1;

    for i =1:length(GT)
        gt = GT(:,:,i);
%         gt = gt';
        scale = max(max(gt, [], 1) - min(gt, [], 1) + 1);
%         scale = sqrt(sum((gt(13,:)-gt(14,:)).^2));
        prd = PRD(:,:,i);
%         prd = prd'; 
        dist = sqrt(sum((prd-gt).^2,2));
        tp(:,n) = dist <= threshold * scale;
        
        n = n+1;
    end
    pck = mean(tp, 2)';
    
    partPCK = (pck + pck([6 5 4 3 2 1 12 11 10 9 8 7 14 13]))/2;
    
    % PCK of [Head, Shou, Elbo, Wris, Hip, Knee, Ankle]
    partPCK = partPCK([13,9,8,7,3,2,1]);
    
    meanPCK = mean(partPCK);
    
    fprintf('mean PCK = %.1f\n', meanPCK * 100); 
    fprintf('Keypoints & Head & Shou & Elbo & Wris & Hip  & Knee & Ankle\n');
    fprintf('PCK       '); fprintf('& %.1f ', partPCK * 100); fprintf('\n');
end

