function [meanPDJ, partPDJ] = evaluatePDJ(GT, PRD, givenThreshold)
    if nargin > 2
    	threshold = givenThreshold;
    else
        threshold = 0.2;
    end
    n = 1;
%define torsor as the disant between left hand and right shoulder + right hand and left shoulder /2  
    for i =1:length(GT)
        gt = GT(:,:,i);
%         gt = gt';
        dist = 0.5 * (sqrt(sum((gt(7,:)-gt(10,:)).^2)) + sqrt( sum ((gt(9,:)-gt(12,:)).^2))); 
        
        prd = PRD(:,:,i);
%         prd = prd';
        result = sqrt(sum((prd-gt).^2,2));
        tp(:,n) = result <= threshold * dist;
        
        n = n+1;
    end
    pdj = mean(tp, 2)';
    

    partPDJ = (pdj + pdj([6 5 4 3 2 1 12 11 10 9 8 7 14 13]))/2;

   
    partPDJ = partPDJ([13,9,8,7,3,2,1]);
    
    meanPDJ = mean(partPDJ);
    
    fprintf('mean PDJ = %.1f\n', meanPDJ * 100); 
    fprintf('Keypoints & Head & Shou & Elbo & Wris & Hip  & Knee & Ankle\n');
    fprintf('PDJ       '); fprintf('& %.1f ', partPDJ * 100); fprintf('\n');
end