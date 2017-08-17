function [meanPCP, partPCP] = evaluatePCP(GT, PRD, givenThreshold)
    if nargin > 2
    	threshold = givenThreshold;
    else
        threshold = 0.2;
    end
    n = 1;

    for i =1:length(GT)
        gt = GT(:,:,i);
%         gt = gt';
        dist1 = sqrt(sum((gt(14,:)-gt(13,:)).^2));
        dist2 = sqrt(sum((gt(13,:)-gt(9,:)).^2));
        dist3 = sqrt(sum((gt(9,:)-gt(8,:)).^2));
        dist4 = sqrt(sum((gt(8,:)-gt(7,:)).^2));
        dist5 = sqrt(sum((gt(13,:)-gt(10,:)).^2));
        dist6 = sqrt(sum((gt(10,:)-gt(11,:)).^2));
        dist7 = sqrt(sum((gt(11,:)-gt(12,:)).^2));
        dist8 = sqrt(sum((gt(9,:)-gt(3,:)).^2));
        dist9 = sqrt(sum((gt(3,:)-gt(2,:)).^2));
        dist10 = sqrt(sum((gt(2,:)-gt(1,:)).^2));
        dist11 = sqrt(sum((gt(10,:)-gt(4,:)).^2));
        dist12 = sqrt(sum((gt(4,:)-gt(5,:)).^2));
        dist13 = sqrt(sum((gt(5,:)-gt(6,:)).^2));
        
        
        
        
        prd = PRD(:,:,i);
%         prd = prd';
        result1 = sqrt(sum((gt(1,:) - prd(1,:)).^2));
        result2 = sqrt(sum((gt(2,:) - prd(2,:)).^2));
        result3 = sqrt(sum((gt(3,:) - prd(3,:)).^2));
        result4 = sqrt(sum((gt(4,:) - prd(4,:)).^2));
        result5 = sqrt(sum((gt(5,:) - prd(5,:)).^2));
        result6 = sqrt(sum((gt(6,:) - prd(6,:)).^2));
        result7 = sqrt(sum((gt(7,:) - prd(7,:)).^2));
        result8 = sqrt(sum((gt(8,:) - prd(8,:)).^2));
        result9 = sqrt(sum((gt(9,:) - prd(9,:)).^2));
        result10 = sqrt(sum((gt(10,:) - prd(10,:)).^2));
        result11 = sqrt(sum((gt(11,:) - prd(11,:)).^2));
        result12 = sqrt(sum((gt(12,:) - prd(12,:)).^2));
        result13 = sqrt(sum((gt(13,:) - prd(13,:)).^2));
        result14 = sqrt(sum((gt(14,:) - prd(14,:)).^2));
        
        
        
        tp(1,n) = (result1 <= threshold*dist10);
        tp(2,n) = (result2 <= threshold*dist9) & (result2 <= threshold*dist10); 
        tp(3,n) = (result3 <= threshold*dist9) & (result3 <= threshold*dist8); 
        tp(4,n) = (result4 <= threshold*dist11) & (result4 <= threshold*dist12);
        tp(5,n) = (result5 <= threshold*dist12) & (result5 <= threshold*dist13);
        tp(6,n) = (result6 <= threshold*dist13);
        tp(7,n) = (result7 <= threshold*dist4);
        tp(8,n) = (result8 <= threshold*dist3) & (result8 <= threshold*dist4);
        tp(9,n) = (result9 <= threshold*dist2) & (result9 <= threshold*dist3);
        tp(10,n) = (result10 <= threshold*dist5) & (result10 <= threshold*dist6);
        tp(11,n) = (result11 <= threshold*dist6) & (result11 <= threshold*dist7);
        tp(12,n) = (result12 <= threshold*dist7);
        tp(13,n) = (result13 <= threshold*dist1) & (result13 <= threshold*dist2) & (result13 <= threshold*dist5);
        tp(14,n) = (result14 <= threshold*dist1);
        
        
        n = n+1;
    end
    pcp = mean(tp, 2)';
    

    partPCP = (pcp + pcp([6 5 4 3 2 1 12 11 10 9 8 7 14 13]))/2;

    
    partPCP = partPCP([13,9,8,7,3,2,1]);
    
    meanPCP = mean(partPCP);
    
    fprintf('mean PCP = %.1f\n', meanPCP * 100); 
    fprintf('Keypoints & Head & Shou & Elbo & Wris & Hip  & Knee & Ankle\n');
    fprintf('PCP       '); fprintf('& %.1f ', partPCP * 100); fprintf('\n');
end