using Pkg, InteractiveDynamics, CairoMakie, LinearAlgebra, DataStructures, Agents
using Agents: ABM, DictABM
using Random: Xoshiro

# wolf agent type
@agent struct Wolf(ContinuousAgent{2, Float64})
end

# wolf agent type
@agent struct Sheep(ContinuousAgent{2, Float64})
    health::Float64 # decreases as hunt progresses
    speed::Float64 # same for all sheep except "invalid" sheep
    in_R1::Stack{Wolf} # stack of wolves in R1
    in_C::Stack{Wolf} # stack of wolves in Corona
    in_R2::Stack{Wolf} # stack of wolves in R2
end

# divides nonzero vector by its magnitude
function safe_norm(v)
    return norm(v) == 0 ? [0.0, 0.0] : v / norm(v)
end

# fn updates the position and velocity of a sheep agent after one time step, dt
function animal_step!(agent::Sheep, model)
    
    d_rep = model.d_rep # min. distance at which neighboring sheep start repelling each other
    epsilon = model.epsilon # half the width of the corona
    d_C = model.d_C # distance to the center of the corona

    # number of neighboring sheep exerting interaction forces on agent
    n_att = model.n_att # number of neighboring sheep the agent is attracted to
    n_ali = model.n_ali # number of neighboring sheep the agent aligns with
    n_rep = 0 # counts the number of neighboring sheep within drep of the agent

    # weight parameters
    w_rep_ws = model.w_rep_ws # for repulsive force from wolf
    w_rep_ss = model.w_rep_ss # for repulsive force from other sheep
    w_att = model.w_att # for attractive force from other sheep
    w_ali = model.w_ali # for alignment with other sheep

    # sheep agent wants to move away from wolf agents
    wolf_repulsion = [0, 0] # stores net repulsive force from wolves on sheep

    # Empty and reset stacks
    empty!(agent.in_R1)
    empty!(agent.in_C)
    empty!(agent.in_R2)
    
    # each wolf exerts a repulsive force on the sheep agent
    for wolf in allagents(model)

        if isa(wolf, Wolf)

            dist = norm(agent.pos - wolf.pos)

            # sort wolves by region into corresponding stacks
            if dist < d_C - epsilon
                push!(agent.in_R2, wolf)
            elseif dist < d_C + epsilon
                push!(agent.in_C, wolf)
            else 
                push!(agent.in_R1, wolf)
            end

            # constant to control how fast force exponentially decays
            c = model.c
            
            # sum up the repulsive forces exerted by each wolf on the sheep
            # force is proportional to distance between wolf and sheep
            unit_vector = (agent.pos - wolf.pos) / (dist)

            # force equation is a decaying exponential
            wolf_repulsion += (exp(-c * dist)) * unit_vector

        end

    end

    # sheep agent also experiences external interaction forces from neighboring sheep
    
    sheep_attraction = [0.0, 0.0] # will hold the sum attractive force from neighboring sheep
    sheep_repulsion = [0.0, 0.0] # will hold the sum repulsive force from neighboring sheep
    heading_angle = 0.0 # will hold average alignment angle

    # sheep repulsed by nearby sheep that are too close
    for sheep in allagents(model)

        # find all sheep within distance drep from sheep agent
        if isa(sheep, Sheep) && sheep.id != agent.id
            if norm(sheep.pos - agent.pos) < d_rep

                # add normalized repulsive force vector to total sheep repulsion
                sheep_repulsion += safe_norm(agent.pos - sheep.pos)
                # keep count of total number of sheep that are too close
                n_rep += 1
            end
        end
    end

    # collect all nearby sheep into an array
    neighbors = [i for i in nearby_agents(agent, model, 5.0)]

    # sheep agent attracted to n_att number of neighboring sheep
    # (how to prevent repeated random selection?)
    for i in 1:n_att

        if (length(neighbors) > 0)
            # choose a random neighbor to be attracted to and add attractive force to total
            neighbor = neighbors[rand(1:length(neighbors))]
            sheep_attraction += safe_norm(neighbor.pos - agent.pos)
        end

    end

    # sheep agent aligning with n_ali number of neighboring sheep
    for i in 1:n_ali

        if (length(neighbors) > 0)
            # ultimately have to make an array from the n_att sheep (rn, all of nearby sheep)
            # chose a random neighbor to align with and add alignment force to total
            neighbor = neighbors[rand(1:length(neighbors))]

            phi = atan(neighbor.vel[1], neighbor.vel[2])
            phi = phi < 0 ? phi + 2*pi : phi
            heading_angle += (1/n_ali) * phi

        end
        
    end

    # normalize all forces acting on sheep agent (so all vectors have magnitude = 1)
    wolf_repulsion = safe_norm(wolf_repulsion)
    sheep_repulsion = safe_norm(sheep_repulsion)
    sheep_attraction = safe_norm(sheep_attraction)
    sheep_alignment = [cos(heading_angle), sin(heading_angle)]

    # scale forces by weight parameters and sum together
    net_direction = (w_rep_ws * wolf_repulsion) + (w_rep_ss * sheep_repulsion) + 
                (w_att * sheep_attraction) + (w_ali * sheep_alignment)
    
    # split into net change in direction and prev direction
    agent.vel = safe_norm(net_direction) * agent.speed * agent.health


    if (agent.health >= 1/5000)
        # agent.health -= 1/5000
    end

    # update agent
    move_agent!(agent, model, model.dt)

    if (agent.pos[1] == 0.0 || agent.pos[2] == 0.0 || agent.pos[1] == model.size[1] || agent.pos[2] == model.size[2])
        remove_agent!(agent, model)
    end

end

# fn updates the position and velocity of a wolf agent after one time step, dt
function animal_step!(agent::Wolf, model)

    w_rep_ww = model.w_rep_ww
    wolf_speed = model.wolf_speed
    epsilon = model.epsilon # half the width of the corona
    d_C = model.d_C # distance to the center of the corona

    # constants to control how fast force exponentially decays
    a = model.a
    b = model.b

    wolf_repulsion = [0, 0] # stores net wolf-wolf repulsive force vector on current wolf
    sheep_attraction = [0, 0]

    in_R2 = false
    in_C = false
    min_dist = typemax(Float64)

    closest = agent

    for sheep in allagents(model)

        if isa(sheep, Sheep)

            dist = norm(sheep.pos - agent.pos)

            # summing attractive forces from sheep
            unit_vector = (sheep.pos - agent.pos) / dist
            sheep_attraction += a * exp(-b * dist) * unit_vector

            if (dist < d_C - epsilon)
                in_R2 = true
                if (dist < min_dist)
                    closest = sheep
                    min_dist = dist
                end
            elseif (dist < d_C + epsilon)
                in_C = true
                if (dist < min_dist)
                    closest = sheep
                    min_dist = dist
                end
            end
                
        end
    end

    if (in_R2)
        
        sheep_attraction = [0, 0]
        dist = norm(closest.pos - agent.pos)
        unit_vector = safe_norm(closest.pos - agent.pos)
        sheep_attraction += -a * exp(-b * dist) * unit_vector

    elseif (in_C)

        sheep_attraction = [0, 0]
        
        for wolf in (w for w in closest.in_C if w.id != agent.id)
            
            dist = norm(agent.pos - wolf.pos)
            wolf_repulsion += w_rep_ww * (agent.pos - wolf.pos) / (dist)^2
        
        end
    
    end

    net_direction = sheep_attraction + wolf_repulsion
    agent.vel = safe_norm(net_direction) * wolf_speed
    
    # update model with new velocity after dt (timestep increment for simulation)
    move_agent!(agent, model, model.dt)

end

# this fn initializes our agent-based model for wolf hunt of a single reactive/escaping prey
function initialize(; size,
                    total_wolf, total_sheep,
                    d_rep, w_rep_ww,
                    a, b, c,
                    wolf_speed, sheep_speed,
                    wolf_mass, sheep_mass,
                    n_att, n_ali,
                    w_rep_ws, w_rep_ss, w_att, w_ali,
                    epsilon, d_C,
                    dt, seed)
    space = ContinuousSpace(size; periodic = false)

    properties = Dict{Symbol, Any}(
    :size => size,
    :total_wolf => total_wolf,
    :total_sheep => total_sheep,
    :d_rep => d_rep,
    :w_rep_ww => w_rep_ww,
    :a => a,
    :b => b,
    :c => c,
    :wolf_speed => wolf_speed,
    :sheep_speed => sheep_speed,
    :wolf_mass => wolf_mass,
    :sheep_mass => sheep_mass,
    :n_att => n_att, 
    :n_ali => n_ali,
    :w_rep_ws => w_rep_ws,
    :w_rep_ss => w_rep_ss,
    :w_att => w_att,
    :w_ali => w_ali,
    :epsilon => epsilon,
    :d_C => d_C,
    :dt => dt,
    :in_range => false
)

    rng = Xoshiro(seed)

    model = ABM(Union{Wolf, Sheep}, space;
                        scheduler = Schedulers.ByType((Sheep, Wolf), true),
                        properties=properties,
                        agent_step! = animal_step!,
                        rng=rng,
                        container=Dict)

    # adding wolf agents at random positions in top-left corner with velocity = 0
    for n in 1:(total_wolf)

        rand_pos = [5*rand(rng), size[2] - 5*rand(rng)]
        add_agent!(Wolf, model; pos = rand_pos, vel = (0.0, 0.0))
    end

    # adding sheep agents around the center with velocity = 0
    for n in 1:(total_sheep) - 1
        
        rand_pos = [20 + 5*rand(rng), size[2] - 20 - 5 * rand(rng)]
        add_agent!(Sheep, model; pos = rand_pos, vel = (0.0, 0.0), speed = sheep_speed, health = 1.0,
                    in_R1 = Stack{Wolf}(), in_C = Stack{Wolf}(), in_R2 = Stack{Wolf}())

    end 

    # add a slower sheep
    rand_pos = [20 + 5*rand(rng), size[2] - 20 - 5 * rand(rng)]
    add_agent!(Sheep, model; pos = rand_pos, vel = (0.0, 0.0), speed = 0.5*sheep_speed, health = 1.0,
                in_R1 = Stack{Wolf}(), in_C = Stack{Wolf}(), in_R2 = Stack{Wolf}())

    return model
end

# this fn returns a video simulation of a wolf pack hunting a single reactive/escaping prey
function freactive_hunt_sim(total_wolf = 6, total_sheep = 25, d_rep = 1.0, w_rep_ww = 0.01,
                             a = 1, b = 0.8, c = 0.8,
                             wolf_speed = 2.0, sheep_speed = 1.0, wolf_mass = 1.0, sheep_mass = 1.0, 
                             n_att = 5, n_ali = 2,
                             w_rep_ws = 2, w_rep_ss = 2, w_att = 1.0, w_ali = 0.8,
                             epsilon = 0.5, d_C = 0.5,
                             dt = 0.25, size = (40.0, 40.0), seed = 124, framerate = 5, frames = 40)

    model = initialize(; size, total_wolf, total_sheep,
                       d_rep,
                       w_rep_ww,
                       a, b, c,
                       wolf_speed,
                       sheep_speed,
                       wolf_mass,
                       sheep_mass,
                       n_att, n_ali,
                       w_rep_ws,
                       w_rep_ss,
                       w_att,
                       w_ali,
                       epsilon,
                       d_C,
                       dt, seed)

    # Define color, size, and marker functions
    agent_color(a::Sheep) = :green
    agent_color(a::Wolf) = :blue
    agent_size(a::Sheep) = 10
    agent_size(a::Wolf) = 8
    agent_marker(a::Sheep) = 'o'
    agent_marker(a::Wolf) = :diamond

    # Create the animation
    abmvideo("test_sheep.mp4", model;
    title = "Reactive Sheep Hunt Simulation", framerate, frames, agent_color, agent_size, agent_marker)

end

freactive_hunt_sim(
    6,                      # total_wolf
    60,                     # total_sheep
    1.0,                    # d_rep
    1.0,                    # w_rep_ww
    1.0,                    # a
    2.0,                    # b
    0.8,                    # c
    5.0,                    # wolf_speed
    5.0,                    # sheep_speed
    1.0,                    # wolf_mass
    1.0,                    # sheep_mass
    5,                      # n_att
    3,                      # n_ali
    10.0,                   # w_rep_ws
    13.0,                   # w_rep_ss
    5.0,                    # w_att
    5.0,                    # w_ali
    0.5,                    # epsilon
    5.0,                    # d_C
    0.01,                   # dt
    (100.0, 100.0),         # size of sim space
    120,                     # seed 
    1000,                   # framerate
    5000                    # frames
)

println("success!")