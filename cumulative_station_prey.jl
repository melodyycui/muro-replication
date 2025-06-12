using Pkg, InteractiveDynamics, CairoMakie, Agents, LinearAlgebra
using Random: Xoshiro

@agent struct Wolf(ContinuousAgent{2, Float64})
end

@agent struct Sheep(ContinuousAgent{2, Float64})
end

# can maybe be put in our slider/hard coded parameters
min_safe_distance = 1.0                 # critical distance at which wolf begins exhibiting encircling behavior
ww_force_coefficient = 0.5              # coefficient of repulsive force exerted by wolf on wolf
sw_force_coefficient = 2                # coefficient of repulsive force exerted by sheep on wolf
sheep_tangent_acceleration = - 0.1     # constant tangential deceleration (slowing down)
dt = 0.25                                  # time step for simulation
rotation_matrix = [cos(pi/2) sin(pi/2); -sin(pi/2) cos(pi/2)]
center = [10.0, 10.0]
sheep_init_speed = 1.0                  # initial tangential speed for sheep

# Function to move a sheep at each time step
function animal_step!(agent::Sheep, model)

    radius = norm(agent.pos - center)
    unit_radial_vector = (center - agent.pos) / radius

    new_speed = norm(agent.vel) + sheep_tangent_acceleration * dt
    centrip_accel = (new_speed^2/radius)* unit_radial_vector

    agent.vel += centrip_accel * dt

    tangential_vector = rotation_matrix * unit_radial_vector
    proj = tangential_vector * dot(agent.vel, tangential_vector)/norm(tangential_vector)
    agent.vel = proj/norm(proj) * new_speed

    move_agent!(agent, model, dt)

end


# Function to move a wolf at each time step
function animal_step!(agent::Wolf, model)

    # identify the sheep in the model
    sheep = nothing
    for a in allagents(model)
        if typeof(a) == Sheep
            sheep = a
            break                       # exit loop once prey is found
        end
    end

    # Model the behavior of the wolf
    current_distance = norm(sheep.pos - agent.pos)
    wolf_encounter_speed = 1.0      # this is an arbitrary speed choice

    # for each neighbor wolf, find repulsive force α distance
    wolf_repulsion = [0, 0]
    for neighbor in nearby_agents(agent, model)

        if neighbor.id != agent.id && isa(neighbor, Wolf)
            
            # sum up the repulsive force vectors to get acceleration
            distance_between = norm(agent.pos - neighbor.pos)
            wolf_repulsion += ww_force_coefficient * (agent.pos - neighbor.pos) / (distance_between)^2

        end

    end
    
    if (current_distance <= min_safe_distance)

        rotation_matrix = [cos(pi/2) sin(pi/2); -sin(pi/2) cos(pi/2)]

        # projecting the repulsive force vector onto the tangential vector
        # to determine the direction the wolf travels along the circle
        u = rotation_matrix * (agent.pos - sheep.pos)
        dot_product = dot(u, wolf_repulsion)
        proj_u_v = (dot_product / norm(u)^2) * u * wolf_encounter_speed

        agent.vel = proj_u_v

    else

        # impact of sheep attraction on distance to sheep
        wolf_encounter_velocity = (sheep.pos - agent.pos) / norm(sheep.pos - agent.pos) * wolf_encounter_speed

        agent.vel = wolf_encounter_velocity + wolf_repulsion ## THINK about how repulsive forces will impact the velocity 
    
    end
    
    move_agent!(agent, model, dt)
end

function initialize(; total_agents = 6, size = (20.0, 20.0), min_safe_distance = 0.1, seed = 125)
    space = ContinuousSpace(size; periodic = false)
    properties = Dict{Symbol, Any}(:min_safe_distance => min_safe_distance)

    rng = Xoshiro(seed)

    model = StandardABM(Union{Wolf, Sheep}, space; properties = properties, agent_step! = animal_step!, rng,
        container = Vector, # agents are not removed, so we use this
        scheduler = Schedulers.Randomly() # all agents are activated once at random
    )

    model.data[:positions] = Vector{Vector{Tuple{Int, Int, NTuple{2, Float64}}}}()  # time series

    for n in 1:(total_agents - 1)
        add_agent!(Wolf, model; vel = (0.0, 0.0))
    end 

    rand_pos = [5 + 10 *rand(rng), 5 + 10*rand(rng)]
    radial_vector = rand_pos - center
    tangential_vector = rotation_matrix * radial_vector
    initial_vel = (tangential_vector/norm(tangential_vector)) * sheep_init_speed

    add_agent!(Sheep, model; pos = rand_pos, vel = initial_vel) # add one sheep
    return model
end



# Attempting to Plot wolf model

# Model step function to record positions
function model_step!(model)
    push!(model.data[:positions], [(a.id, typeof(a), a.pos) for a in allagents(model)])
end

model = initialize()
run!(model, 200)

ac(a::Wolf) = :green
ac(a::Sheep) = :blue
as(a::Wolf) = 10
as(a::Sheep) = 13
am = 'o'
abmvideo("moving_wolf_hunt.mp4", model;
title = "Wolf Hunt Simulation", framerate = 15, frames = 200, ac, as, am)