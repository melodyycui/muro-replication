using Pkg, InteractiveDynamics, CairoMakie, Agents, LinearAlgebra
using Random: Xoshiro

# wolf agent type
@agent struct Wolf(ContinuousAgent{2, Float64})
end

# wolf agent type
@agent struct Sheep(ContinuousAgent{2, Float64})
    speed::Float64
end

function safe_norm(v)
    return norm(v) == 0 ? [0.0, 0.0] : v / norm(v)
end

# fn updates the position and velocity of a sheep agent after one time step, dt
function animal_step!(agent::Sheep, model)

    mass_sheep = 1

    # sheep agent wants to move away from wolf agents
    wolf_repulsion = [0, 0] # stores net repulsive force from wolves on sheep
    
    # each wolf exerts a repulsive force on the sheep agent
    for wolf in allagents(model)

        if isa(wolf, Wolf)

            # placeholder values
            b = 0.8
            
            # sum up the repulsive forces exerted by each wolf on the sheep
            # force is proportional to distance between wolf and sheep
            distance_between = norm(agent.pos - wolf.pos)
            # get the direction of the repulsive force
            unit_vector = (agent.pos - wolf.pos) / (distance_between)

            # force equation is a decaying exponential
            wolf_repulsion += (exp(-b * distance_between)) * unit_vector

        end

    end

    # sheep agent also experiences external interaction forces from neighboring sheep

    n_att = 5 # number of neighboring sheep the agent is attracted to
    n_ali = 2 # number of neighboring sheep the agent aligns with
    sheep_attraction = [0, 0] # will hold the sum attractive force from neighboring sheep
    sheep_alignment = [0, 0] # will hold sum alignment force from neighboring sheep


    drep = 1 # min. distance at which neighboring sheep start repelling each other
    n_rep = 0 # counts the number of neighboring sheep within drep of the agent
    sheep_repulsion = [0, 0] # will hold the sum repulsive force from neighboring sheep
    
    # weight parameters
    w_rep_wolf = 2 # for repulsive force from wolf
    w_rep_sheep = 2 # for repulsive force from other sheep
    w_att = 1.0 # for attractive force from other sheep
    w_ali = 0.8 # for alignment with other sheep

    # sheep repulsed by nearby sheep that are too close
    for sheep in allagents(model)

        # find all sheep within distance drep from sheep agent
        if isa(sheep, Sheep) && sheep.id != agent.id
            if norm(sheep.pos - agent.pos) < drep

                # add normalized repulsive force vector to total sheep repulsion
                sheep_repulsion += safe_norm(agent.pos - sheep.pos)
                # keep count of total number of sheep that are too close
                n_rep += 1
            end
        end
    end

    # collect all nearby sheep into an array
    neighbors = [i for i in nearby_agents(agent, model)]

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
            sheep_alignment += safe_norm(neighbor.pos)
        end
        
    end

    # normalize all forces acting on sheep agent
    wolf_repulsion = safe_norm(wolf_repulsion)
    sheep_repulsion = safe_norm(sheep_repulsion)
    sheep_attraction = safe_norm(sheep_attraction)
    sheep_alignment = safe_norm(sheep_alignment)

    # scale forces by weight parameters and sum together
    net_force = (w_rep_wolf * wolf_repulsion) + (w_rep_sheep * sheep_repulsion) 
                + (w_att * sheep_attraction) + (w_ali * sheep_alignment)
    
    # v = v0 + at, a = F/m => v = v0 + F/m * t
    agent.vel += (net_force/mass_sheep)*model.dt

    # direction of velocity is updated, but keep speed the same
    agent.vel = agent.speed * safe_norm(agent.vel)

    # update agent
    move_agent!(agent, model, model.dt)

end

# fn updates the position and velocity of a wolf agent after one time step, dt
function animal_step!(agent::Wolf, model)

    mass_wolf = 1
    ww_repulsion = [0, 0] # stores net wolf-wolf repulsive force vector on current wolf

    # each neighbor wolf exerts repulsive force on current wolf agent
    for neighbor in allagents(model)

        if neighbor.id != agent.id && isa(neighbor, Wolf)
            
            # sum up wolf-wolf repulsive force vectors
            distance_between = norm(agent.pos - neighbor.pos)
            ww_repulsion += model.ww_force_coefficient * (agent.pos - neighbor.pos) / (distance_between)^2

        end

    end

    ws_attraction = [0, 0]
    circling = false

    for sheep in allagents(model)

        if isa(sheep, Sheep)

            current_distance = norm(sheep.pos - agent.pos)

            # once wolf is within a critical distance to the sheep, wolf will maintain that critical distance
            # and orbit around sheep due to repulsion from other wolves
            if (current_distance <= model.min_safe_distance)

                # find direction of wolf to sheep, rotate 90 degrees to find tangential movement direction
                rotation_matrix = [cos(pi/2) sin(pi/2); -sin(pi/2) cos(pi/2)]
                u = rotation_matrix * (agent.pos - sheep.pos)

                # projecting the repulsive force vector onto the tangential vector and multiply by wolf speed
                # to determine the velocity the wolf travels along the circle
                dot_product = dot(u, ww_repulsion)
                proj_u_v = (dot_product / norm(u)^2) * u * model.wolf_chase_speed
                agent.vel = proj_u_v
                circling = true
                break

            else

                # placeholder values
                a = 1
                b = 0.8

                # summing attractive forces from sheep
                unit_vector = (sheep.pos - agent.pos) / current_distance
                ws_attraction += a * exp(-b * current_distance) * unit_vector
            end
        end
    end

    if (!circling)

        acceleration = (ww_repulsion + ws_attraction) / mass_wolf
        agent.vel += acceleration * model.dt
        agent.vel = model.wolf_chase_speed * agent.vel / norm(agent.vel)
    
    end
    
    # update model with new velocity after dt (timestep increment for simulation)
    move_agent!(agent, model, model.dt)
end

# this fn initializes our agent-based model for wolf hunt of a single reactive/escaping prey
function initialize(; total_agents, size,
                    min_safe_distance, ww_force_coefficient,
                    sw_force_coefficient, wolf_chase_speed,
                    dt, seed, center, sheep_speed,)
    space = ContinuousSpace(size; periodic = false)

    properties = Dict{Symbol, Any}(
    :min_safe_distance => min_safe_distance,
    :ww_force_coefficient => ww_force_coefficient,
    :sw_force_coefficient => sw_force_coefficient,
    :wolf_chase_speed => wolf_chase_speed,
    :dt => dt,
    :center => center,
    :sheep_speed => sheep_speed, :in_range => false
)

    rng = Xoshiro(seed)

    model = StandardABM(Union{Wolf, Sheep}, space;
                        properties=properties,
                        agent_step! = animal_step!,
                        rng=rng,
                        container=Vector,
                        scheduler=Schedulers.Randomly())
    
    # placeholder parameters
    total_wolf = 6
    total_sheep = 25

    # adding wolf agents at random positions in top-left corner with velocity = 0
    for n in 1:(total_wolf)

        rand_pos = [2*rand(rng), 18 + 2*rand(rng)]
        add_agent!(Wolf, model; pos = rand_pos, vel = (0.0, 0.0))
    end

    # adding sheep agents around the center with velocity = 0
    for n in 1:(total_sheep)-1
        
        rand_pos = [5 + 10 *rand(rng), 5 + 10*rand(rng)]
        add_agent!(Sheep, model; pos = rand_pos, vel = (0.0, 0.0), speed = 2.0)

    end 

    # add a slower sheep
    rand_pos = [5 + 10 *rand(rng), 5 + 10*rand(rng)]
    add_agent!(Sheep, model; pos = rand_pos, vel = (0.0, 0.0), speed = 1.0)

    return model
end

# this fn returns a video simulation of a wolf pack hunting a single reactive/escaping prey
function freactive_hunt_sim(min_safe_distance=1.0, ww_force_coefficient=0.5, sw_force_coefficient=2.0,
                             wolf_chase_speed=2.0, dt=0.25,
                             total_agents=6, size=(20.0,20.0), seed=125, framerate=15, frames=200,
                             center=[10.0,10.0], sheep_speed=1.0)

    model = initialize(total_agents=total_agents, size=size,
                       min_safe_distance=min_safe_distance,
                       ww_force_coefficient=ww_force_coefficient,
                       sw_force_coefficient=sw_force_coefficient,
                       wolf_chase_speed=wolf_chase_speed,
                       dt=dt, seed=seed, center=center,
                       sheep_speed=sheep_speed)

    # Define color, size, and marker functions
    ac(a::Sheep) = :green
    ac(a::Wolf) = :blue
    as(a::Sheep) = 10
    as(a::Wolf) = 8
    am(a::Sheep) = 'o'
    am(a::Wolf) = :diamond

    # Create the animation
    abmvideo("wolf_hunt.mp4", model;
    title = "Reactive Sheep Hunt Simulation", framerate, frames, ac, as, am)

end

# test with 3 wolves and 1 sheep
freactive_hunt_sim(
    1.0,                    # min_safe_distance
    0.01,                    # ww_force_coefficient
    1.0,                    # sw_force_coefficient
    2.0,                    # wolf_chase_speed
    0.25,                   # dt
    4,                      # total_agents 
    (40.0, 40.0),           # size of sim space
    124,                    # seed 
    3,                     # framerate
    20,                     # frames 
    [20.0, 20.0],           # center of sheep movement
    0.8                     # sheep_speed in m/s
)