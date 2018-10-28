% valid actions
actions = [1,2,3,4];
world = 4;
gwinit(world);
s = gwstate();

% map for Q-values(xSize*ySize*actions)
Q = zeros(s.xsize, s.ysize, numel(actions));
episodes  = 10000;

% initial parameters
alpha = 0.4;          % alpha - learning rate
gamma = 0.8;          % gamma - discount factor

initial_epsilon = 0.8;% epsilon - exploration factor
epsilon_goal = 0.3;                    
epsilon = (initial_epsilon - epsilon_goal) / episodes;

%[0.25, 0.25, 0.25, 0.25]
prob_a = (1/numel(actions)) * ones(size(actions));


% main program loop
for i = 1:episodes
    gwinit(world); %get random position in world
    old_state = gwstate();
    
    if(mod(i,20) == 0)
        i
    end
    %Q-function
    while old_state.isterminal == 0
        %select action
        [action, opt_action] = chooseaction(Q, old_state.pos(1), old_state.pos(2), actions, prob_a, initial_epsilon-(i*epsilon));
        new_state = gwaction(action);
        %update value
        if new_state.isvalid == 1
            update = new_state.feedback + gamma * max(Q(new_state.pos(1),new_state.pos(2),action));
            Q(old_state.pos(1),old_state.pos(2),action) = (1-alpha)* Q(old_state.pos(1),old_state.pos(2),action)+ alpha * update;
      
        else
            reward = -0.5;
                update = reward + gamma * max(Q(new_state.pos(1),new_state.pos(2),action));
                 Q(old_state.pos(1),old_state.pos(2),action) = (1-alpha)* Q(old_state.pos(1),old_state.pos(2),action)+ alpha * update;
            %punish invalid moves
            y_dir = old_state.pos(1) - new_state.pos(1); % 1: up, -1: down
            x_dir = old_state.pos(2) - new_state.pos(2); % 1: left, -1: right
            %check if action was deliberate
            up = (y_dir == 1 && action == 2); 
            down = (y_dir == -1 && action == 1);
            left = (x_dir == 1 && action == 4);
            right = (x_dir == -1 && action == 3);
            if up || down || left || right
                %punish
                Q(old_state.pos(1), old_state.pos(2), action) = -inf;
                new_state = old_state;
            else
                %reduce the reward for reaching this tile
                reward = -0.5;
                update = reward + gamma * max(Q(new_state.pos(1),new_state.pos(2),action));
                Q(old_state.pos(1),old_state.pos(2),action) = (1-alpha)* Q(old_state.pos(1),old_state.pos(2),action)+ alpha * update;
            end
        end
        old_state = new_state;
    end
end

gwinit(world);
figure(1);
gwdraw
for x = 1:size(Q,1)
    for y = 1:size(Q,2)
        [temp ,action] = max(Q(x,y,:));
        gwplotarrow([x,y],action);
    end
end 
figure(2);
gwdraw;
[V_star,A_star] = max(Q,[],3);
x = 1:size(V_star,2);
y = 1:size(V_star,1);
for i = x
    for j = y
        gwplotarrow([y(j);x(i)],A_star(j,i))
    end
end
title 'Estimated optimal policy'
figure(2);
imagesc(V_star); title 'V*'; axis image; xlabel Y; ylabel X; colorbar;