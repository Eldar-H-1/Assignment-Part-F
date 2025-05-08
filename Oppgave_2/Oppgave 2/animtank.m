function animtank(block)
%ANIMTANK Animation of water tank system.
%

% Copyright 2021-2023 The MathWorks, Inc.

setup(block)
end
%% Local functions
function setup(block)
%%
block.NumInputPorts  = 1;
block.NumOutputPorts = 0;
block.NumDialogPrms = 1;
block.NumDworks = 0;

% Register as scope block (i.e. will be skipped in codegen, input data will
% be streamed to host during rapid accelerator mode or external mode)
block.SetSimViewingDevice(true);

initializeUI()

block.RegBlockMethod('Update',@updateUI);
block.RegBlockMethod('Outputs',@updateNextHit);
end
%% Local functions
function initializeUI()
%%
global tankdemo %#ok<GVMIS>

% Initialize the figure for use with this simulation
fuzzy_animinit('Tank Demo');
tankdemo = findobj(0,'Name','Tank Demo');

tank1Wid=1;
tank1Ht=2;
tank1Init=0;
setPt=0.5;

tankX=[0 0 1 1]-0.5;
tankY=[1 0 0 1];
% Draw the tank
line(1.1*tankX*tank1Wid+1,tankY*tank1Ht+0.95,'LineWidth',2,'Color','black');
tankX=[0 1 1 0 0]-0.5;
tankY=[0 0 1 1 0];
% Draw the water
waterX=tankX*tank1Wid+1;
waterY=tankY*tank1Init+1;
tank1Hndl=patch(waterX,waterY,'blue','EdgeColor','none');
% Draw the gray wall
waterY([1 2 5])=tank1Ht*[1 1 1]+1;
waterY([3 4])=tank1Init*[1 1]+1;
tank2Hndl=patch(waterX,waterY,[.9 .9 .9],'EdgeColor','none');
% Draw the set point
lineHndl=line([0 0.4],setPt*[1 1]+1,'Color','red','LineWidth',4);

set(tankdemo, ...
    'Color',[.9 .9 .9], ...
    'UserData',[tank1Hndl tank2Hndl lineHndl]);
ax = tankdemo.Children(3);
set(ax, ...
    'XLim',[0 2],'YLim',[0 3.5], ...
    'XColor','black','YColor','black', ...
    'Box','on');
axis equal
xlabel('Water Level Control','Color','black','FontSize',10);
set(get(ax,'XLabel'),'Visible','on')
end

function updateUI(block)
%%
global tankdemo %#ok<GVMIS>

if any(get(0,'Children')==tankdemo)
    if strcmp(get(tankdemo,'Name'),'Tank Demo')
        u = block.InputPort(1).Data;
        % Update tank one level
        tankHndlList = get(tankdemo,'UserData');
        yData = get(tankHndlList(1),'YData');
        yOffset = yData(1);
        yData(3:4) = [1 1]*u(2)+yOffset;
        set(tankHndlList(1),'YData',yData);

        yData = get(tankHndlList(2),'YData');
        yData([3 4]) = [1 1]*u(2)+yOffset;
        set(tankHndlList(2),'YData',yData);

        yData = [1 1]*u(1)+1;
        set(tankHndlList(3),'YData',yData);

        drawnow
    end
end

end

function updateNextHit(block)
%%
% ns stores the number of samples
t = block.CurrentTime;
ts = block.DialogPrm(1).Data;
ns = t/ts; % block.CurrentTime

% This is the time of the next sample hit.
block.NextTimeHit = (1 + floor(ns + 1e-13*(1+ns)))*ts;
end