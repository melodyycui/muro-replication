using Pkg, InteractiveDynamics, CairoMakie, Agents, LinearAlgebra
using Random: Xoshiro

# wolf agent type
@agent struct Wolf(ContinuousAgent{2, Float64})
end

# wolf agent type
@agent struct Sheep(ContinuousAgent{2, Float64})
end

# fn updates the position and velocity of a sheep agent after one time step, dt
function animal_step!(agent::Sheep, model)

    # sheep agent wants to move away from wolf agents
    ws_repulsion = [0, 0] # stores net repulsive force from wolves on sheep
    
    # each wolf exerts a repulsive force on the sheep agent
    for wolf in allagents(model)

        if wolf.id != agent.id
            
            # sum up the repulsive forces exerted by each wolf on the sheep
            # repulsive force is inversely proportional to distance
            distance_between = norm(agent.pos - wolf.pos)
            ws_repulsion += model.sw_force_coefficient * (agent.pos - wolf.pos) / (distance_between)^2

        end

    end

    # sheep moves in direction of net repulsive force at speed = sheep_speed
    agent.vel = ws_repulsion / norm(ws_repulsion) * model.sheep_speed
    move_agent!(agent, model, model.dt)

end


# fn updates the position and velocity of a wolf agent after one time step, dt
function animal_step!(agent::Wolf, model)

    # identify the sheep in the model
    sheep = nothing
    for a in allagents(model)
        if typeof(a) == Sheep
            sheep = a
            break                       # exit loop once prey is found
        end
    end

    ww_repulsion = [0, 0] # stores net wolf-wolf repulsive force vector on current wolf
    
    # each neighbor wolf exerts repulsive force on current wolf agent
    for neighbor in nearby_agents(agent, model)

        if neighbor.id != agent.id && isa(neighbor, Wolf)
            
            # sum up wolf-wolf repulsive force vectors
            # repulsive force is inversely proportional to distance between wolves
            distance_between = norm(agent.pos - neighbor.pos)
            ww_repulsion += model.ww_force_coefficient * (agent.pos - neighbor.pos) / (distance_between)^2

        end

    end

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

    # if wolf is outside critical distance from sheep, wolf is attracted to sheep
    else

        # wolf moves in direction of sheep due to attractive force exerted by sheep on wolf
        wolf_chase_velocity = (sheep.pos - agent.pos) / norm(sheep.pos - agent.pos) * model.wolf_chase_speed

        # wolf velocity impacted by both attraction to sheep and repulsion to neighboring wolves
        agent.vel = wolf_chase_velocity + ww_repulsion 
    
    end
    
    # update model with new velocity after dt (timestep increment for simulation)
    move_agent!(agent, model, model.dt)
end

function model_step!(model)

    temp_sheep_bary = [0.0, 0.0]
    temp_wolf_bary = [0.0, 0.0]

    # recalculate barycenter
    for a in allagents(model)
        if typeof(a) == Sheep
            temp_sheep_bary += a.pos
        else
            temp_wolf_bary += a.pos
    end

    # will need to be model properties bc that'll be what 
    # wolves/sheep moving towards/away
    sheep_barycenter = temp_sheep_bary / total_sheep
    wolf_barycenter = temp_wolf_bary / total_wolf # need to split total_agents into sep sheep/wolf vars
    dist_to_bary = abs((wolf_barycenter - sheep_barycenter) 
                        / norm(wolf_barycenter - sheep_barycenter))

    if (isnothing(model.target))

        # check if wolf within range
        if (dist_to_bary < dist_init_chase)
            model.in_range = true

            dist_closest_sheep = 1.5 * model.size[1] # distance between wolf barycenter and closest sheep

            # find closest sheep to wolf barycenter
            for a in allagents(model)
                if typeof(a) == Sheep && norm(a.pos - wolf_barycenter) < dist_closest_sheep
                    dist_closest_sheep = norm(sheep.pos - wolf_barycenter)
                    if (dist_closest_sheep < dist_flock_sep)
                        model.target = a
                    end
                end
            end
        end
    end
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
                        agent_step! = animal_step!, model_step!,
                        rng=rng,
                        container=Vector,
                        scheduler=Schedulers.Randomly())

    # adding wolf agents at random positions with velocity = 0
    for n in 1:(total_agents - 1)
        add_agent!(Wolf, model; vel = (0.0, 0.0))
    end 

    # randomly generating a position that does NOT fall near the edges of the space
    rotation_matrix = [cos(pi/2) sin(pi/2); -sin(pi/2) cos(pi/2)]
    rand_pos = [5 + 10 *rand(rng), 5 + 10*rand(rng)] # more central random init position so vid stays in frame

    # adding a single sheep/prey agent at the generated random position with velocity = 0
    add_agent!(Sheep, model; pos = rand_pos, vel = (0.0, 0.0)) # add one sheep
    return model
end

# this fn returns a video simulation of a wolf pack hunting a single reactive/escaping prey
function freactive_hunt_sim(min_safe_distance=1.0, ww_force_coefficient=0.5, sw_force_coefficient=2.0,
                             wolf_chase_speed=1.0, dt=0.25,
                             total_agents=6, size=(20.0,20.0), seed=125, framerate=15, frames=200,
                             center=[10.0,10.0], sheep_speed=0.5)

    model = initialize(total_agents=total_agents, size=size,
                       min_safe_distance=min_safe_distance,
                       ww_force_coefficient=ww_force_coefficient,
                       sw_force_coefficient=sw_force_coefficient,
                       wolf_chase_speed=wolf_chase_speed,
                       dt=dt, seed=seed, center=center,
                       sheep_speed=sheep_speed)

    # Define color, size, and marker functions
    ac(a::Wolf) = :green
    ac(a::Sheep) = :blue
    as(a::Wolf) = 10
    as(a::Sheep) = 13
    am(a::Wolf) = 'o'
    am(a::Sheep) = :diamond

    # Create the animation
    abmvideo("wolf_hunt.mp4", model;
    title = "Reactive Sheep Hunt Simulation", framerate, frames, ac, as, am)

end

# test with 3 wolves and 1 sheep
freactive_hunt_sim(
    1.0,                    # min_safe_distance
    0.5,                    # ww_force_coefficient
    1.0,                    # sw_force_coefficient
    1.0,                    # wolf_chase_speed
    0.25,                   # dt
    4,                      # total_agents 
    (20.0, 20.0),           # size of sim space
    124,                    # seed 
    15,                     # framerate
    200,                    # frames 
    [10.0, 10.0],           # center of sheep movement
    1.3                     # sheep_speed in m/s (found from Jadhav)
)