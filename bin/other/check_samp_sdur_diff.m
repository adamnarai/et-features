
% samp, sdur, spvel
kernel_sw = 'on';
var_x_name = 'samp';
var_y_name = 'sdur';
x_lim = inf;
y_lim = inf;
spdt_lim = 50;

% var_x_name = 'samp';
% var_y_name = 'sdur';
% x_lim = inf;
% y_lim = inf;
% spdt_lim = inf;

spvel_sub = aeta.s.d.SP2.spvel;
var_x_sub = aeta.s.d.SP2.(var_x_name);
var_y_sub = aeta.s.d.SP2.(var_y_name);
filt_vect = cat(1, spvel_sub{25:48}) > spdt_lim;
var_x_1 = cat(1, var_x_sub{25:48});
var_y_1 = cat(1, var_y_sub{25:48});
var_x_1 = var_x_1(filt_vect);
var_y_1 = var_y_1(filt_vect);

spvel_sub = aeta_2.s.d.SP2.spvel;
var_x_sub = aeta_2.s.d.SP2.(var_x_name);
var_y_sub = aeta_2.s.d.SP2.(var_y_name);
filt_vect = cat(1, spvel_sub{:}) > spdt_lim;
var_x_2 = cat(1, var_x_sub{:});
var_y_2 = cat(1, var_y_sub{:});
var_x_2 = var_x_2(filt_vect);
var_y_2 = var_y_2(filt_vect);

figure
x = [var_x_1; var_x_2];
y = [var_y_1; var_y_2];
group = [repmat({'dys'}, length(var_x_1), 1);...
    repmat({'dys\_contr\_2'}, length(var_x_2), 1)];
scatterhist(x, y, 'group', group, 'kernel', kernel_sw);
title(['Sacc. peak vel. manual th: ', num2str(spdt_lim), char(10),...
    var_x_name, ' median: ', num2str(nanmedian(var_x_1)), ' / ',...
    num2str(nanmedian(var_x_2)), ' (dys / dys\_contr\_2)', char(10),...
    var_y_name, ' median: ', num2str(nanmedian(var_y_1)), ' / ',...
    num2str(nanmedian(var_y_2)), ' (dys / dys\_contr\_2)']);
xlabel(var_x_name);
ylabel(var_y_name);
xlim([0 x_lim]);
ylim([0 y_lim]);

%%
gs = 0;
for i = 1:24
%     subid = subj{i};
%     load(sprintf('ET_results_%s.mat', subid));

    subid = sorder{i};
    load(sprintf('%s_TextLinesWithLeftValidation_correctionType1_limitedToReading.mat', subid));
    
    s = 0;
%     for row = 1:100
    for row = find(corrInfWithoutNaN(:,10) == 2)'
%         test = [[ET_results.saccadeInfo(1,row,:).amplitude]', [ET_results.saccadeInfo(1,row,:).duration]'];
        test = [[ETresults.saccadeInfo(1,row,:).amplitude]', [ETresults.saccadeInfo(1,row,:).duration]'];
        if ~isempty(test)
            sacc_idx = find(test(:,1)<2 & test(:,2)>0.10);
            s = s + length(sacc_idx);
        end
        %     fprintf('%d: %s\n', row, num2str(sacc_idx));
        %     test(sacc_idx,:);
    end
    fprintf('%s sum: %d\n', subid, s);
    gs = gs + s;
end
fprintf('Grand sum: %d\n\n', gs);
% 
% %%
% gs = 0;
% for i = 1:38
%     subid = subj{i};
%     load(sprintf('ET_results_%s.mat', subid));
%     s = 0;
%     for row = 1:100
%         test = [[ET_results.saccadeInfo(1,row,:).amplitude]', [ET_results.saccadeInfo(1,row,:).duration]'];
%         if ~isempty(test)
%             sacc_idx = find(test(:,1)<2 & test(:,2)>0.10);
%             s = s + length(sacc_idx);
%         end
%         %     fprintf('%d: %s\n', row, num2str(sacc_idx));
%         %     test(sacc_idx,:);
%     end
%     fprintf('%s sum: %d\n', subid, s);
%     gs = gs + s;
% end
% fprintf('Grand sum: %d\n\n', gs);
% 
% %%
% figure
% row = 6;
% X = ET_results.data(1,row,1).X;
% hold on
% plot(X)
% plot(ET_results.saccadeIdx(1,row,1).Idx*800);
% hold off

%%
close all
var_x_name = 'samp';
var_y_name = 'spvel';
var_x_sub = aeta.s.d.NS.(var_x_name);
var_y_sub = aeta.s.d.NS.(var_y_name);

for subi = 1:24
    X = var_x_sub{subi};
    Y = var_y_sub{subi};
    figure
    scatterhist(X, Y);
    xlabel(var_x_name);
    ylabel(var_y_name);
    corr(X,Y, 'type', 'Spearman')
end

